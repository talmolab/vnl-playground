"""Mouse arm+hand+neck+head imitation (v24 model, STAC v24-native clips).

Unlike v22, v24's forearm_v24.xml has a single fixed root ("Armature", no
joint -- the spine/rib/skull chain down to it is entirely rigid) and purely
anatomically-named rotational joints (shoulder_protraction/internal_rotation/
abduction, elbow_flexion, forearm_supination, wrist_deviation/flexion,
per-digit mcp/pip/dip/tip_flexion, neck_flexion, head_flexion) -- there is no
shoulder-translation slide to snap kinematically, so none of
imitation_arm_hand.py's ik_driven_qpos_idx/multi-root machinery applies here.
Plain MouseImitation (single root, fully muscle-driven, no IK-snapped dims)
is the right base class.

v24 also has no joystick at all (forearm_v24.xml has zero joystick body/
geoms) -- see default_config_v25() below for why that variant isn't
buildable yet.
"""

import glob
import os

import h5py
from ml_collections import config_dict

from vnl_playground.tasks.mouse.consts import (
    JANELIA_MOUSE_V24_XML_PATH,
    MOUSE_REFERENCE_DATA_JANELIA_V24_PATH,
)
from vnl_playground.tasks.mouse.imitation import (
    MouseImitation,
    default_config as imitation_default_config,
)

# Tracked bodies mirror register_v24_mocap.py's KEYPOINT_MODEL_PAIRS (the
# body set STAC actually registered mocap markers against) -- verified
# directly from a real v24 h5's embedded config, not assumed:
# Shoulder->humerus_right, Elbow->ulna_right, Wrist->N_L_C_right (v24's
# wrist-equivalent fused carpal body -- there is no body literally named
# "wrist"), D1_K..D5_K->Phalanx_hand_1_{1..5}_right (proximal knuckle),
# D2_M..D5_M->Phalanx_hand_2_{2..5}_right (middle), D1_T->
# Phalanx_hand_2_1_right (thumb has no separate tip segment, same quirk as
# v22), D2_T..D5_T->Phalanx_hand_3_{2..5}_right (tips).
_TRACKED_BODIES = [
    "humerus_right",  # Shoulder keypoint
    "ulna_right",  # Elbow keypoint
    "N_L_C_right",  # Wrist keypoint / end effector
    "Phalanx_hand_1_1_right",  # D1_K
    "Phalanx_hand_1_2_right",  # D2_K
    "Phalanx_hand_1_3_right",  # D3_K
    "Phalanx_hand_1_4_right",  # D4_K
    "Phalanx_hand_1_5_right",  # D5_K
    "Phalanx_hand_2_2_right",  # D2_M
    "Phalanx_hand_2_3_right",  # D3_M
    "Phalanx_hand_2_4_right",  # D4_M
    "Phalanx_hand_2_5_right",  # D5_M
    "Phalanx_hand_2_1_right",  # D1_T
    "Phalanx_hand_3_2_right",  # D2_T
    "Phalanx_hand_3_3_right",  # D3_T
    "Phalanx_hand_3_4_right",  # D4_T
    "Phalanx_hand_3_5_right",  # D5_T
]

# STAC's own fit config (n_fit_frames in every v24 h5's embedded config, and
# RENDER_FPS) -- verified directly against CFL_35_20240128_trial_0101_ik.h5,
# the one complete trial available as of 2026-07-17.
_NATIVE_N_FRAMES = 126
_NATIVE_MOCAP_HZ = 25


def _complete_clip_indices(data_path: str, n_frames: int = _NATIVE_N_FRAMES):
    """Indices (into the same sorted glob MouseReferenceClips itself uses)
    of clips that have the full native frame count.

    The STAC v24 fitting job is still running as of 2026-07-17: most trials
    on disk are 15-frame in-progress stubs, only one
    (CFL_35_20240128_trial_0101) has the full 126 frames. MouseReferenceClips
    stacks all loaded clips into one array assuming uniform length -- handing
    it a mix of 126- and 15-frame clips raises a stacking error, not just bad
    data. Unlike v22x/v23's keep_clips_idx (a hardcoded list Eric picked by
    reviewing reprojection videos), this is computed dynamically from
    whatever's actually on disk right now, since which trials are complete
    will keep changing as the STAC run progresses -- re-run this (i.e. just
    call default_config_v24() again) rather than trusting a stale list.
    """
    h5_files = sorted(
        glob.glob(os.path.join(str(data_path), "**", "*_ik.h5"), recursive=True)
    )
    keep = []
    for i, path in enumerate(h5_files):
        with h5py.File(path, "r") as f:
            if f["qpos"].shape[0] == n_frames:
                keep.append(i)
    return keep


def default_config() -> config_dict.ConfigDict:
    """v24 arm+hand+neck+head defaults: 52-muscle xml, STAC v24-native clips,
    filtered to whichever trials currently have the full 126 frames."""
    cfg = imitation_default_config()
    cfg.walker_xml_path = JANELIA_MOUSE_V24_XML_PATH
    cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_JANELIA_V24_PATH)
    # Fit natively against this exact kinematic tree (same convention as
    # v22's STAC-v22-native data) -- no v21/v22-style transplant needed.
    cfg.recompute_kinematics = False
    cfg.root_bodies = ("Armature",)
    # forearm_v24.xml's own <option> is integrator="implicitfast", but
    # base.py's compile() only re-applies cfg.integrator when it's not None
    # (default: None = "don't override, keep whatever the arena/walker
    # attach-conflict resolution settled on") -- and that resolution keeps
    # the ARENA's integrator ("RK4", per arena.xml) over the walker's,
    # exactly the same category of bug that made v22 immediately unstable
    # before it was fixed there by forcing integrator="euler" explicitly
    # (see imitation_arm_hand.py's default_config()). Confirmed directly:
    # leaving this at None here produced NaN on the very first random-action
    # step of a smoke test. Force it now, same fix.
    cfg.integrator = "euler"
    # Per Eric 2026-07-17: pair with solver="newton" too, matching v22's
    # exact combo (base.py's own default is solver="cg") rather than mixing
    # a new solver with the ported integrator choice.
    cfg.solver = "newton"
    # Per Eric 2026-07-17: keep the well-tested "2 control steps per mocap
    # frame" ratio established for v22 (see
    # docs/2026-07-17-v22-joystick-camera-fix-and-first-successful-eval.md's
    # "Open follow-up" section) rather than starting over from v1/v3's
    # original 400Hz/16-steps-per-frame pair -- no reason to reintroduce the
    # staircase-target oscillation issue in a brand new model.
    cfg.mocap_hz = _NATIVE_MOCAP_HZ
    cfg.clip_length = None
    cfg.ctrl_dt = 0.02
    cfg.sim_dt = 0.001
    # ppo_params.episode_length in train_mouse_janelia_v24.py must be
    # 252 (= 126 frames / (0.02*25) frames-per-step).
    cfg.tracked_bodies = list(_TRACKED_BODIES)
    cfg.end_effector = "N_L_C_right"
    cfg.keep_clips_idx = _complete_clip_indices(cfg.reference_data_path)
    # No shoulder-translation IK-snap dims exist in this model (see module
    # docstring) -- ik_driven_qpos_idx stays empty/unset (imitation.py's base
    # default), everything is muscle-actuated.
    # njmax/naconmax: v24 has no joystick and, per the muscle-only xml's own
    # header comment, ncon=0 at rest -- but self-contact between digits
    # during muscle-driven motion is still possible (unlike v22x, which has
    # zero possible contacts by construction since no two geoms share a
    # complementary contype/conaffinity pair). Start with v22's proven
    # per-world njmax=512 and re-verify against a real random-action stress
    # test before trusting it, same caution as every other njmax/naconmax
    # value in this codebase.
    cfg.njmax = 512
    cfg.naconmax = 16384
    return cfg


def default_config_v25() -> config_dict.ConfigDict:
    """v25 (v24 arm+hand+joystick): NOT YET BUILDABLE.

    Per Eric 2026-07-17: unlike v22 (where the joystick already existed in
    the raw model and only needed repositioning -- see
    docs/2026-07-17-v22-joystick-camera-fix-and-first-successful-eval.md),
    v24's forearm_v24.xml has zero joystick body/geoms, and there is no
    v24-with-joystick STAC fit or model file anywhere on disk as of this
    writing. Building this requires real joystick geometry sourced from a
    new STAC fit against a v24-plus-joystick model -- not numbers ported
    from v22's unrelated hand geometry. Raises rather than silently
    returning a broken config.
    """
    raise NotImplementedError(
        "v25 (v24 + joystick) is blocked on a v24-with-joystick STAC fit / "
        "model file, which does not exist yet. See this function's "
        "docstring and consts.py's comment above "
        "MOUSE_REFERENCE_DATA_JANELIA_V24_PATH."
    )
