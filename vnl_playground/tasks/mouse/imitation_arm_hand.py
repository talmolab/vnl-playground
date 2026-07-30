"""Mouse arm+hand+joystick imitation (v22 model, STAC v22-native clips).

Generalizes MouseImitationMovingShoulder's "snap some qpos dims kinematically,
actuate the rest via muscles" mechanism from a leading qpos slice to an
arbitrary set of indices: v22's qpos layout is
`[x_slide, y_slide, sh_tx, sh_ty, sh_tz, sh_rotation, ...]`, so the IK-driven
shoulder-translation dims (sh_tx/ty/tz) sit at indices 2-4, *after* the
joystick's x_slide/y_slide (indices 0-1), which are left fully physically
simulated (driven only by hand-joystick contact, per Eric 2026-07-16).
"""

import glob
import os
from typing import Any, Dict, Optional, Union

import brax.math as brax_math
import h5py
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.mouse import contact_presets
from vnl_playground.tasks.mouse.consts import (
    janelia_scott_v1_xml_path,
    janelia_scott_v2_xml_path,
    JANELIA_MOUSE_ARM_HAND_V22_RAW_JOYSTICK_XML_PATH,
    JANELIA_MOUSE_ARM_HAND_V22_XML_PATH,
    JANELIA_MOUSE_ARM_HAND_V22X_XML_PATH,
    JANELIA_MOUSE_ARM_HAND_V23_XML_PATH,
    JANELIA_MOUSE_ARM_HAND_SCOTT_V1_XML_PATH,
    JANELIA_MOUSE_ARM_HAND_V25_XML_PATH,
    MOUSE_REFERENCE_DATA_JANELIA_V22_PATH,
    MOUSE_REFERENCE_DATA_JANELIA_V22X_PATH,
    MOUSE_REFERENCE_DATA_JANELIA_V25_PATH,
)
from vnl_playground.tasks.mouse.imitation import (
    MouseImitation,
    default_config as imitation_default_config,
    _registry as _parent_registry,
)
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.reward_registry import RewardRegistry

# Tracked bodies mirror the STAC v21 fit's KEYPOINT_MODEL_PAIRS (the body set
# the mocap markers were actually registered against), deduplicating the
# joystick (both js_base and js_ball map to the same "joystick" body).
_TRACKED_BODIES = [
    "humerus",  # Shoulder keypoint
    "ulna",  # Elbow keypoint
    "wrist",  # Wrist keypoint / end effector
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
    "joystick",  # js_base / js_ball
]

# qpos/qvel indices snapped from the STAC reference every step (shoulder
# translation). Everything else is muscle-actuated by the policy. The
# joystick's x_slide/y_slide (indices 0-1) are deliberately NOT in this list —
# they stay physically simulated via hand-joystick contact.
_IK_DRIVEN_QPOS_IDX = [2, 3, 4]  # sh_tx, sh_ty, sh_tz

# v22x (no joystick): sh_tx/ty/tz shift to indices 0-2 since x_slide/y_slide
# (previously 0-1) no longer exist in this model's qpos layout.
_IK_DRIVEN_QPOS_IDX_V22X = [0, 1, 2]  # sh_tx, sh_ty, sh_tz

# v25: matches KEYPOINT_MODEL_PAIRS in the v24 STAC fit config exactly.
_TRACKED_BODIES_V25 = [
    "humerus_right",
    "ulna_right",
    "N_L_C_right",
    "Phalanx_hand_1_1_right",
    "Phalanx_hand_1_2_right",
    "Phalanx_hand_1_3_right",
    "Phalanx_hand_1_4_right",
    "Phalanx_hand_1_5_right",
    "Phalanx_hand_2_2_right",
    "Phalanx_hand_2_3_right",
    "Phalanx_hand_2_4_right",
    "Phalanx_hand_2_5_right",
    "Phalanx_hand_2_1_right",
    "Phalanx_hand_3_2_right",
    "Phalanx_hand_3_3_right",
    "Phalanx_hand_3_4_right",
    "Phalanx_hand_3_5_right",
]

# v25 joystick-contact reward geometry: the model's fixed set of 15 possible
# contact pairs (5 hand grip geoms x 3 joystick geoms, reduced 2026-07-21
# from an original 10 grip geoms -- every other geom pair is either
# non-colliding or excluded via contype/conaffinity, see
# default_config_v25()'s docstring). Each entry is (geom_name, name of the
# body it's welded to *as recorded in reference_clips' xpos/xquat*, i.e. the
# pre-"-mouse"-suffix name) -- reference clips only store body-level
# xpos/xquat, not per-geom, so the reference-side geom position has to be
# reconstructed as body_xpos + rotate(local_geom_offset, body_xquat) rather
# than read directly like the sim side (data.geom_xpos, already computed by
# mj_forward).
_JOYSTICK_GEOMS_V25 = [
    ("circular_base", "joystick_base"),
    ("joystick_geom", "joystick"),
    ("joystick_ball", "joystick"),
]
_GRIP_BODIES_V25 = [
    # Reduced from 10 to 5 (2026-07-21, per Eric): drop the thumb entirely
    # (Metacarpal_hand_1_right + its tip Phalanx_hand_2_1_right) and 3 of
    # the 4 remaining metacarpals, keeping one ("palm") representative plus
    # the 4 non-thumb fingertips. Halves the joystick contact pairs
    # (10x3 -> 5x3), matching the contype/conaffinity=0 disabling of the
    # same geoms in the walker XML -- keep in sync with that file.
    "Metacarpal_hand_3_right",  # palm representative
    "Phalanx_hand_3_2_right",
    "Phalanx_hand_3_3_right",
    "Phalanx_hand_3_4_right",
    "Phalanx_hand_3_5_right",
]
_GRIP_GEOMS_V25 = [(name + "_col", name) for name in _GRIP_BODIES_V25]


def default_config() -> config_dict.ConfigDict:
    """v22 arm+hand defaults: 35-muscle xml + STAC v22-native clips."""
    cfg = imitation_default_config()
    # Per Eric 2026-07-17: disable the pose_error early-termination criterion
    # (base default: terminate if ||target.joints - qpos|| > 3.0) -- episodes
    # were ending immediately under an untrained/early policy (100% of eval
    # episodes hit pose_error in the v22x baseline eval), cutting off training
    # signal before the policy has a chance to learn. nan_termination stays.
    del cfg.termination_criteria["pose_error"]
    cfg.walker_xml_path = JANELIA_MOUSE_ARM_HAND_V22_XML_PATH
    cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_JANELIA_V22_PATH)
    # This data is fit natively against the same kinematic tree as
    # walker_xml_path (verified: stored xpos matches our model's own FK to
    # 0.0000mm) -- no recompute needed, unlike the old v21-transplant data.
    cfg.recompute_kinematics = False
    # v22 has two disconnected roots under worldbody: "shoulder_base" (arm,
    # carries sh_tx/ty/tz) and "joystick_base" (manipulandum, carries
    # x_slide/y_slide). "clavicle" -- the single root the base add_mouse()
    # default assumes -- is a *child* of shoulder_base here, not a root, so
    # both real roots must be listed or their joints get silently dropped.
    cfg.root_bodies = ("shoulder_base", "joystick_base")
    # The _contacts xml's joystick geoms are cylinders colliding against hand
    # mesh geoms; MJX's pure-JAX collision pipeline doesn't implement
    # cylinder-vs-mesh (raises NotImplementedError at mjx.put_model), but the
    # Warp backend does.
    cfg.mujoco_impl = "warp"
    # v22's own <option> is integrator="Euler" solver="Newton", but
    # MouseBaseEnv.compile() always re-applies cfg.solver (base default:
    # "cg") after compiling, and the arena/walker attach-conflict resolution
    # separately keeps the arena's integrator="RK4" over the walker's. The
    # Warp MJX backend's RK4 kernel errors on this model's geom types, so
    # match what the XML actually intends here.
    cfg.solver = "newton"
    cfg.integrator = "euler"
    # imitation.py's base default (clip_length=50, mocap_hz=200) was tuned
    # for the old single-animal dataset, not this one. Verified from the
    # original registered-mocap trial's own reprojection video metadata
    # (fps=25.0, duration=5.04s, 126 frames -- 126/25=5.04 exactly): this
    # STAC v22-native data is natively 25Hz, 126 frames/clip. clip_length=None uses
    # MouseReferenceClips' native frame count (126) instead of truncating.
    cfg.mocap_hz = 25
    cfg.clip_length = None
    # ctrl_dt=0.02, sim_dt=0.001 (20 substeps/control step) -- 2026-07-17,
    # per Eric: switch from v1/v3's proven 0.0025/0.00125 pair (400Hz
    # control, 16 control steps per mocap frame) to 2 control steps per
    # mocap frame instead, to fix the staircase-target artifact confirmed
    # in the track-a-smoke run (checkpoints/janelia-v22-arm-hand-20260717-
    # 060110-track-a-smoke/5242880.mp4): imitation.py's _get_cur_frame()
    # does floor(data.time * mocap_hz + start_frame), so at ctrl_dt=0.0025
    # with mocap_hz=25 the reference target held for ~16 consecutive
    # control steps with nothing changing in between, and the policy was
    # free to chatter at ~400Hz between updates -- clearly visible as
    # fast oscillation in that video despite otherwise-plausible
    # arm+joystick behavior. ctrl_dt=0.02 (0.04s mocap frame / 2) restores
    # v1's own ~2-steps-per-frame ratio (v1 used mocap_hz=200,
    # ctrl_dt=0.0025 -> 0.005/0.0025=2 steps/frame). sim_dt=0.001 (20
    # substeps) chosen over 0.00125 (16 substeps) deliberately: a prior
    # same-day test found sim_dt=0.00125 paired with a ctrl_dt=0.02-like
    # rate showed a 24.2% NaN rate, while sim_dt=0.001 had one clean eval
    # at this ctrl_dt -- still UNPROVEN AT FULL SCALE (that was one eval,
    # not a full run), so re-verify NaN rate at scale before trusting this
    # combination for a long run, same caution as the pair it replaces.
    # ppo_params.episode_length in train_mouse_janelia_arm_hand.py must be
    # 252 (= 126 frames / (0.02*25) frames-per-step), not 2016.
    cfg.ctrl_dt = 0.02
    cfg.sim_dt = 0.001
    cfg.ik_driven_qpos_idx = list(_IK_DRIVEN_QPOS_IDX)
    cfg.tracked_bodies = list(_TRACKED_BODIES)
    cfg.end_effector = "wrist"
    # mjx.make_data()'s Warp-backend auto-heuristic sizes naconmax/njmax for
    # a SINGLE world (nworld=1) -- with 0 collision geoms it guesses
    # naconmax=16 total, shared across the WHOLE vmapped training batch, not
    # per env. That undersized shared buffer, not mesh-vs-primitive collision
    # cost, is what actually produced the "broadphase overflow"/"nefc
    # overflow" warnings during v22 training (490k+ over one run). Set
    # explicitly, sized for num_envs=4096 training (see
    # train_mouse_janelia_arm_hand.py): njmax is PER WORLD (no "nefc
    # overflow" observed up to 512 over a full 252-step/4096-env randomized-
    # action rollout -- do NOT raise this casually, the dense per-world
    # constraint solve scales ~njmax^2 * nworld, so njmax=1024 alone blew up
    # to an 86.88GiB allocation and OOM'd); naconmax is the TOTAL
    # broadphase-candidate budget across all 4096 worlds (observed peak
    # needed ~63.5k during that same rollout -- fully random +-1 torques on
    # all 35 muscles at every step is a worst-case adversarial stress test,
    # wilder than early real PPO exploration). Set ~1.5x that peak for
    # margin; contact structs scale far more cheaply than njmax does.
    cfg.njmax = 512
    cfg.naconmax = 16384
    # Per Eric 2026-07-17: only these 6 trials (of the 15 available under
    # MOUSE_REFERENCE_DATA_JANELIA_V22_PATH), same selection as v22x's
    # keep_clips_idx (identified by reviewing
    # /root/vast/eric/stac-mjx/v22_all_trials_3d_view/*_3d_fixedcam.mp4 for
    # each candidate trial -- see docs/2026-07-17-v23-handoff.md for the
    # full trial-name <-> index mapping). Indices are into
    # sorted(glob("**/*_ik.h5")) over MOUSE_REFERENCE_DATA_JANELIA_V22_PATH;
    # re-verified directly against that directory's current contents
    # (15 entries, same trial names/order as the doc) on 2026-07-17 --
    # re-run the same glob+sort if trials are ever added/removed there,
    # since this list is position-based, not name-based.
    cfg.keep_clips_idx = [1, 2, 3, 4, 13, 14]
    return cfg


def default_config_no_joystick() -> config_dict.ConfigDict:
    """v22x: same arm+hand model and STAC data, joystick removed entirely.

    Isolates arm+hand reach-tracking from hand-joystick contact dynamics
    (buffer overflow, contact-geometry mismatch) to validate tracking on its
    own before revisiting the joystick in v23 with simplified collision
    geoms. Reuses MouseImitationArmHand unchanged -- nothing in that class
    hardcodes "joystick" by name, it's all config-driven.
    """
    cfg = default_config()
    cfg.walker_xml_path = JANELIA_MOUSE_ARM_HAND_V22X_XML_PATH
    cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_JANELIA_V22X_PATH)
    # This data is fit natively against v22's arm/hand kinematic tree, but
    # v22x is missing one body (no joystick) that the original data's
    # xpos/names_xpos included -- must recompute from the real v22x model.
    cfg.recompute_kinematics = True
    # Single root now -- "shoulder_base" is the model's only top-level body
    # under worldbody once joystick_base is gone.
    cfg.root_bodies = ("shoulder_base",)
    # Reverted back to "jax" 2026-07-17 (second reversal -- see git history).
    # Full story: "jax" was the original v22x choice, blamed for a 49-min
    # apparent hang and switched to "warp" -- but that attempt also had
    # njmax=8 (wrong, see below) at the same time, so backend choice was
    # never cleanly isolated from that bug. "warp" then hit a SEPARATE,
    # confirmed bug: mjx's warp glue hardcodes graph_mode=GraphMode.WARP
    # (mujoco/mjx/warp/ffi.py, not exposed via the public mjx.step() API),
    # and eval_env's long-lived cached jit_step (used both by brax's own
    # periodic eval AND policy_params_fn's video rollout, see
    # train_mouse_janelia_arm_hand.py) crashed on a later call with
    # "RuntimeError: Warp error: unknown stream" after ~11 min of real
    # training happened in between -- confirmed via a real crash log, not
    # inferred. Directly reproduced the SAME call pattern under "jax"
    # (2016-step rollout, 200 unrelated GPU ops simulating the training loop,
    # another 2016-step rollout on the same cached jit_step) with zero
    # crashes, since "jax" never touches Warp's CUDA graph capture at all.
    # "jax" also now confirmed to work correctly on this contact-free model
    # (mjx.put_model(m, impl="jax") succeeds -- v22 needed "warp" only for
    # joystick cylinder-vs-mesh collision, which doesn't exist here).
    # Rejected alternative: giving eval_env a different impl than env would
    # avoid the compile-time cost below, but eval_env is literally what
    # brax's train_fn(eval_env=...) uses for its own official eval metrics
    # -- evaluating under different physics than training happens under is
    # a real correctness risk, not worth it just to dodge a one-time compile
    # cost. Known real cost of "jax": compile time. Same A/B test as before
    # (batch=64, reset+first-step): impl="jax" 87.9s+75.4s=163.3s vs
    # impl="warp" 26.7s+3.35s=30.0s -- ~5.4x slower to compile, though this
    # is a BOUNDED one-time cost against an hours-long 500M-step run, not an
    # actual hang (which is what it looked like the first time, confounded
    # by njmax=8 making everything ambiguous). Full-scale (num_envs=4096,
    # 2-GPU pmap) compile time not yet measured -- verify before trusting
    # this at real scale, not just at the batch sizes tested so far.
    cfg.mujoco_impl = "jax"
    # njmax/naconmax: v22 (with joystick) needs njmax=512/naconmax=16384 (see
    # default_config() above); v22x has ZERO possible contacts (no two geoms
    # share a complementary contype/conaffinity bitmask pair -- verified:
    # ncon=0 at rest AND across 20 random in-range poses). But njmax is NOT
    # just contacts -- it's total nefc (active constraints), which also
    # counts joint-limit constraints, and every joint here is limited=true.
    # njmax=8 (a first guess that only accounted for contacts, wrongly
    # treating "zero contacts" as "zero constraints") caused 6.4M "nefc
    # overflow" warnings in a 19-minute real run once a policy actually moved
    # joints into their limits (observed overflow up to njmax=25 needed).
    # Silently dropped constraints there means joint limits weren't always
    # being enforced -- a real correctness bug, not just log spam. Re-verified
    # this value directly against a live training-shaped rollout (not just a
    # handful of random poses) before trusting it a second time.
    cfg.njmax = 64
    cfg.naconmax = 64
    cfg.ik_driven_qpos_idx = list(_IK_DRIVEN_QPOS_IDX_V22X)
    cfg.tracked_bodies = [b for b in _TRACKED_BODIES if b != "joystick"]
    # Per Eric 2026-07-17: only these 6 trials (of the 15 available under
    # MOUSE_REFERENCE_DATA_JANELIA_V22X_PATH), identified by reviewing
    # /root/vast/eric/stac-mjx/v22_all_trials_3d_view/*_3d_fixedcam.mp4 for
    # each candidate trial. Indices are into sorted(glob("**/*_ik.h5")) over
    # that data directory -- re-verify with the same glob+sort if trials are
    # ever added/removed there, since this list is position-based, not
    # name-based (see docs/2026-07-17-v23-handoff.md for the full trial-name
    # <-> index mapping this was computed from).
    cfg.keep_clips_idx = [1, 2, 3, 4, 13, 14]
    return cfg


def default_config_v23() -> config_dict.ConfigDict:
    """v23: same v22 arm+hand+joystick model/data, but the 38 hand/wrist bone
    collision geoms are simplified primitives (ellipsoid/capsule) instead of
    raw meshes -- meant to fix the "broadphase overflow" contact-buffer issue
    (490k+ warnings over one v22 training run), since mesh-vs-primitive
    collision generates far more broadphase candidate pairs per touching pair
    than primitive-vs-primitive. Kinematic tree, qpos layout, and reference
    data are identical to v22 -- only the collision geoms changed.
    """
    cfg = default_config()
    cfg.walker_xml_path = JANELIA_MOUSE_ARM_HAND_V23_XML_PATH
    return cfg


def default_config_raw_joystick() -> config_dict.ConfigDict:
    """Raw-joystick + fixed-camera variant: identical to default_config()
    (mesh collision geoms -- the config that already trained cleanly to
    500M steps, reward -49 -> 305) except the joystick_base
    position-translation patch is reverted to the raw stac-mjx-synced
    pos/quat (same revert as v23, without v23's ellipsoid collision swap,
    which introduced its own ccd_iterations stalling). Adds the fixed
    diagnostic camera for legible eval renders despite the joystick's real
    ~38.6-degree shaft tilt.
    """
    cfg = default_config()
    cfg.walker_xml_path = JANELIA_MOUSE_ARM_HAND_V22_RAW_JOYSTICK_XML_PATH
    return cfg


# STAC v25's own fit config (n_fit_frames in every v25 h5's embedded config)
# -- verified directly against CFL_35_20240128_trial_0101_ik.h5.
_NATIVE_N_FRAMES_V25 = 126


def _complete_clip_indices_v25(data_path: str, n_frames: int = _NATIVE_N_FRAMES_V25):
    """Indices (into the same sorted glob MouseReferenceClips itself uses) of
    v25 trials that have the full native frame count.

    Same rationale as imitation_v24.py's _complete_clip_indices(): the STAC
    v25 fitting job may still be adding trials (as of 2026-07-17,
    CFL_35_20240128_trial_0001 has only a _fit.h5, no _ik.h5 yet, so it's
    already excluded by MouseReferenceClips' *_ik.h5 glob), but a trial could
    in principle land with a partial/in-progress _ik.h5 too -- filter
    defensively by frame count rather than trusting a hardcoded index list.
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


def default_config_v25() -> config_dict.ConfigDict:
    """v25: v24 arm+hand+neck+head rig plus a joystick, STAC v25-native clips.

    Contacts are deliberately minimal, per Eric 2026-07-17: of the ~40
    ellipsoid collision proxies covering every wrist/hand bone, only 5 are
    left contact-enabled as of 2026-07-21 (contype/conaffinity unchanged from
    the rest of the hand's contype=4/conaffinity=8 vs. the joystick's
    contype=8/conaffinity=4 pairing) and converted from ellipsoid to sphere:
    one "palm" representative (Metacarpal_hand_3_right_col) and the 4
    non-thumb digit tips (Phalanx_hand_3_{2..5}_right_col). The thumb
    (Metacarpal_hand_1_right_col + its tip Phalanx_hand_2_1_right_col) and
    the other 3 metacarpals were disabled 2026-07-21 (per Eric, halving the
    contact-pair count 10x3->5x3 for training throughput; originally all 5
    metacarpals + 5 digit tips were enabled, see git history for that
    version). Every other wrist/hand collision geom (proximal/middle
    phalanges, carpals, sesamoids) has contype/conaffinity zeroed --
    non-colliding. Sphere radius per geom is the mean of the original
    ellipsoid's 3 semi-axes, except the 5 kept-active geoms, which were
    enlarged 2026-07-21 toward each bone's true mesh bounding-sphere radius
    (model.geom_rbound) instead -- the semi-axis-mean spheres were only
    ~35-45% of the visible bone's true size, which was never noticed until
    a collision-only render made it obvious. Paired with the joystick's own
    capsule shaft/base (+ one sphere ball-tip) geoms, every remaining
    contact pair is sphere-sphere or sphere-capsule -- the two simplest
    (fully analytic, no GJK) collision functions in MJX's pure-JAX pipeline
    (see collision_driver.py's FunctionKey table), not the ellipsoid-vs-*
    pairs the raw STAC-synced collision geoms would have used.
    """
    cfg = imitation_default_config()
    del cfg.termination_criteria["pose_error"]
    cfg.walker_xml_path = JANELIA_MOUSE_ARM_HAND_V25_XML_PATH
    cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_JANELIA_V25_PATH)
    # Per Eric 2026-07-19: STAC-fit xpos is already correct; recompute was introducing the error, not fixing one.
    cfg.recompute_kinematics = False
    cfg.root_bodies = ("Armature", "joystick_base")
    cfg.mujoco_impl = "jax"
    cfg.solver = "newton"
    cfg.integrator = "euler"
    # The XML's own <option iterations="30" ls_iterations="30"/> is a
    # deliberate convergence-margin bump for the joystick/fingertip contacts,
    # but MouseBaseEnv.compile() unconditionally re-applies cfg.iterations/
    # cfg.ls_iterations after attach (base.py default_config(): 6/6) --same
    # "parent wins" override that already bit integrator/solver above. Must
    # be set here explicitly or the XML's own value is silently discarded.
    # Lowered 30 -> 15 (2026-07-21, per Eric, for training throughput) now
    # that grip contacts are down to 5 geoms (was 10) -- half the contact
    # pairs (15 vs 30) should converge in fewer iterations. Watch NaN rate /
    # "iterations insufficient" warnings after this change; raise back
    # toward 30 if either shows up.
    cfg.iterations = 15
    cfg.ls_iterations = 15
    cfg.mocap_hz = 25
    cfg.clip_length = None
    cfg.ctrl_dt = 0.02
    cfg.sim_dt = 0.001
    cfg.ik_driven_qpos_idx = []
    cfg.tracked_bodies = list(_TRACKED_BODIES_V25)
    cfg.end_effector = "N_L_C_right"
    # x_slide, y_slide -- the joystick's own 2 translational dofs, always
    # qpos[0:2] (verified against names_qpos 2026-07-20). See
    # _joystick_pos_reward: dedicated, highly-weighted tracking of just these
    # 2 dims, since the generic "joints" reward's single L2 norm over all 27
    # qpos dims drowns the joystick's ~1cm-scale error out among arm/hand
    # joints with much wider natural ranges.
    cfg.joystick_qpos_idx = [0, 1]
    cfg.njmax = 256
    cfg.naconmax = 512
    cfg.keep_clips_idx = _complete_clip_indices_v25(cfg.reference_data_path)
    # Reward for making contact with the joystick when the reference/IK data
    # says contact should be happening -- see _joystick_contact_reward.
    # contact_threshold: hard gate on the reference-side surface-to-surface
    # clearance (grip-geom surface to joystick-geom surface, minimum over all
    # 30 pairs). Calibrated 2026-07-20 directly from all 6 clips' reference
    # data (756 frames total): that clearance is already <=0 (interpenetrating,
    # i.e. definitely gripping) on 66% of frames and <=1mm on 82%, so 1mm
    # comfortably separates "should be gripping" frames from the rest without
    # being so tight that STAC-fit noise flips the gate near the boundary.
    # weight/exp_scale start small relative to bodies_pos/wrist_pos (0.1 each,
    # exp_scale 0.1) since, unlike those, there's no principled starting point
    # yet for this term -- raise both if the policy ignores it once training.
    # weight scaled 0.1 -> 0.0291 alongside joints/joints_vel/bodies_pos/
    # wrist_pos below (2026-07-21, per Eric) -- see joystick_pos's comment
    # for the full rebalance math.
    cfg.reward_terms["joystick_contact"] = {
        "weight": 0.0291,
        "exp_scale": 0.002,
        "contact_threshold": 0.001,
    }
    # Dedicated joystick position/joint-angle tracking (per Eric 2026-07-20:
    # the policy wasn't moving the joystick to the right place -- contact
    # alone doesn't reward *where* it ends up, and the generic "joints"
    # reward's combined 27-dim L2 norm barely notices the joystick's own
    # ~1cm-scale error next to the rest of the body). exp_scale=0.002 (2mm)
    # against the joystick's +-6mm travel range gives a real gradient across
    # that whole range instead of saturating near 0 or 1 everywhere; weight
    # well above wrist_pos/bodies_pos (now downweighted below) so tracking
    # the joystick itself dominates over precisely matching arm/hand pose.
    # Weight raised 2.0 -> 6.0 (2026-07-21, per Eric): 94M steps of training
    # (janelia-v25-arm-hand-joystick-20260720-192714-v25-joystick-pos-reweight)
    # showed joystick_contact learned fine (reward rose from ~0.4 to ~19-20,
    # near its own ceiling) but joystick_pos stayed completely flat the
    # entire run (episode reward for this term: 358.6 at step 0 vs 362.3 at
    # step 94.4M, no trend) even as overall episode_reward climbed 767->925
    # -- entirely driven by the unrelated "joints" term (397->550). The
    # policy learned to touch the joystick but never learned to move it to
    # the right place. Raising this weight well above "joints" (5.0) so
    # actually displacing the joystick competes with, not loses to, general
    # arm/hand pose tracking.
    cfg.reward_terms["joystick_pos"] = {
        "weight": 6.0,
        "exp_scale": 0.002,
    }
    # Reward-scale rebalance (2026-07-21, per Eric): raising joystick_pos
    # 2.0->6.0 alone would inflate the per-step reward ceiling from 7.64 to
    # 11.64 (sum of bodies_pos+joints+joints_vel+joystick_contact+
    # joystick_pos+wrist_pos), making this run's episode_reward numbers not
    # directly comparable to janelia-v25-arm-hand-joystick-20260720-192714's
    # (episode_reward 767->925 over 94M steps). Scaled every OTHER term down
    # by (7.64-6.0)/(7.64-2.0) = 1.64/5.64 ~= 0.2908 so the new ceiling
    # matches the old one (6.0 + 0.2908*5.64 ~= 7.64) with joystick_pos's
    # weight itself left exactly at the intended 6.0. This is a real
    # tradeoff, not a free lunch: "joints" (general 27-dim arm/hand pose
    # tracking, the term that drove ALL of the old run's reward growth) goes
    # 5.0->1.454, a 71% cut -- watch for the arm/hand pose quality
    # regressing, not just whether joystick_pos finally moves.
    cfg.reward_terms["joints"]["weight"] = 1.454
    cfg.reward_terms["joints_vel"]["weight"] = 0.1454
    cfg.reward_terms["wrist_pos"]["weight"] = 0.0058
    cfg.reward_terms["bodies_pos"]["weight"] = 0.0058
    return cfg


def default_config_scott_v1(variant: str = "mixed") -> config_dict.ConfigDict:
    """scott_v1: the v25 rig with a 19-geom anatomical hand and hard contact.

    Three things change together relative to v25, so this is deliberately NOT
    a clean A/B against it:

    1. **Collision geometry.** 19 contact-enabled hand bones (every metacarpal
       and phalanx) as per-bone minimum-volume primitives -- 6 ellipsoid + 13
       capsule -- instead of v25's 5 spheres sized as the arithmetic mean of
       three ellipsoid semi-axes. 57 contact pairs, not 15. See
       analysis/2026-07-28-scott-v1-hand-collision-ellipsoids/README.md.

    2. **Contact parameters.** `contact_preset="harder"`. At v25's shipped
       parameters the kinematic probe measured press transmission 0.024 and
       1.1 mN of push -- a 19-geom anatomical hand buys literally nothing
       until the contact is stiffened, so the geometry work cannot pay off
       without this. See contact_presets.py.

    3. **Reward exp_scales.** Retuned so that some term has gradient at every
       competence level; see the reward_terms block below.

    `joints_vel` is dropped outright (weight 0). It contributed exactly 0.0
    reward and 0.0 gradient in every v25 run -- `exp_scale=0.2` against a
    median error of 33.6 rad/s is `exp(-14148)` -- and rescaling it would be
    worse than deleting it, because the target itself is wrong: the STAC v25
    h5s' stored `qvel` is 14-40x the derivative of the `qpos` stored in the
    same file (median 33.87 vs 1.17 rad/s). Measured, not inferred. Nothing
    else consumes reference qvel here (`qvel_init="zeros"`, and
    `_get_imitation_target` carries joint-angle and body-position deltas only),
    so the blast radius is this one term.
    """
    cfg = default_config_v25()
    cfg.walker_xml_path = janelia_scott_v1_xml_path(variant)

    # -- contact ------------------------------------------------------------
    # "harder" = gap 0 + solimp dmax 0.999 on all 22 grip/joystick geoms, a
    # direct negative solref at stiffness_mult x the shipped k, and the same
    # applied to the joystick's own slide limits (otherwise those become the
    # weakest link and the stick is squeezed through its own +-6 mm range).
    #
    # k=30x with sim_dt=0.25 ms is the probe's measured sweet spot: 62 mN of
    # push against the ~54 mN the joystick's return spring actually needs, with
    # 10 mN of ringing. k=3000x reaches a nominally better transmission (0.964
    # vs 0.920) but delivers 1036 mN -- 19x what the spring needs -- oscillating
    # at +-687 mN, and halving the timestep does not fix it (687 -> 674 mN), so
    # it is genuine stiff-contact ringing rather than discretisation. k=3000x
    # also needs sim_dt=0.01 ms, i.e. 2000 physics substeps per control step
    # against the 80 used here -- a ~25x throughput cost on top of everything.
    cfg.contact_preset = "harder"
    cfg.contact_stiffness_mult = 30.0
    # 0.25 ms is the largest timestep at which k=30x is stable (the probe found
    # each ~5x in stiffness needs ~5x smaller dt). ctrl_dt is unchanged at
    # 0.02 s, so this is 80 substeps per control step, up from v25's 20.
    cfg.sim_dt = 0.00025

    # -- constraint buffers -------------------------------------------------
    # v25's njmax=256/naconmax=512 were sized for 15 contact pairs; scott_v1
    # has 57, all of which are always present in MJX's jax pipeline (the pair
    # list is static -- verified: contact.geom is identical across poses).
    # Sized from a measured random-action rollout, not guessed; undersizing
    # silently drops constraints, which is a correctness bug rather than log
    # spam (see the v22x njmax=8 history in default_config_no_joystick).
    cfg.njmax = 512
    cfg.naconmax = 1024

    # -- reward exp_scales --------------------------------------------------
    # Every tracking term is w*exp(-(d/s)^2/2), so `s` alone decides whether a
    # term has usable gradient: at d/s -> 0 it is saturated (improving gains
    # nothing), at d/s >> 1 it is dead (invisible to PPO), and the signal
    # |d*dr/dd|/w peaks at d/s = sqrt(2). Measured on the 519M v25 checkpoint
    # (6 clips x 252 steps) and on Eric's from-scratch probe, v25's scales put
    # every term in one of the two dead regimes:
    #
    #   term          s      r/w at cold | zero-action | expert
    #   joystick_pos  2 mm   0.73 | 0.72 | 0.80   <- saturated; doing nothing
    #                                               already scores 91% of what
    #                                               the trained policy scores
    #   joints        0.2    0.00 | 0.00 | 0.37   <- exp(-35) from a cold start:
    #                                               nothing to climb
    #   wrist_pos     0.1    0.99 | 1.00 | 1.00   <- a constant
    #   bodies_pos    0.1    0.91 | 1.00 | 1.00   <- a constant
    #
    # The scales below form a staircase instead, so something always has
    # gradient: wrist_pos/bodies_pos are cold-start shaping (live early,
    # saturate late), joints/joystick_pos carry late precision. Predicted
    # effect, from the same measurement: the zero-action floor drops from 57%
    # of the reward ceiling to 16%. Weights are deliberately left alone.
    cfg.reward_terms["joystick_pos"]["exp_scale"] = 0.00075
    cfg.reward_terms["joints"]["exp_scale"] = 0.8
    cfg.reward_terms["wrist_pos"]["exp_scale"] = 0.005
    cfg.reward_terms["bodies_pos"]["exp_scale"] = 0.03
    cfg.reward_terms["joints_vel"]["weight"] = 0.0
    # Raised 0.01 -> 0.02. Tension worth recording: control_cost was the *only*
    # term with a live gradient in the from-scratch v25 probe, and the policy
    # optimised it by going limp and backing away from the joystick (clearance
    # 7.0 -> 8.3 mm over 78M steps). Raising it is defensible only because the
    # tracking terms above now carry real gradient; if a run goes limp again
    # this is the first thing to reverse.
    cfg.reward_terms["control_cost"]["weight"] = 0.02
    # Recalibrated against the exact reference clearance computed by
    # MouseImitationArmHandScottV1 (mj_geomDistance over all 57 pairs), which
    # is a different quantity from v25's isotropic-radius proxy -- see that
    # class's _joystick_contact_reward. 0 mm = "the reference hand is touching
    # or inside the joystick", which holds on ~77% of reference frames.
    cfg.reward_terms["joystick_contact"]["exp_scale"] = 0.00075
    cfg.reward_terms["joystick_contact"]["contact_threshold"] = 0.0
    return cfg


def default_config_scott_v2(variant: str = "wrist_forearm") -> config_dict.ConfigDict:
    """scott_v2: scott_v1 plus contact at the wrist and the wrist->elbow bones.

    **Exactly one thing changes from scott_v1: the set of contacting geoms.**
    Every reward scale, weight, contact parameter, timestep and the entire
    kinematic tree are inherited untouched, and the added geoms are
    `density="0"` so body mass and inertia are bit-identical (asserted at emit
    time on all 51 bodies). scott_v1 vs scott_v2 is therefore a clean
    single-factor comparison, which scott_v1 vs v25 deliberately was not.

    What it fixes, measured on the a5 rollout at 293.6M steps (6 clips,
    8,091 contact events -- see
    analysis/2026-07-29-scott-v2-wrist-forearm-contact-geoms/README.md):

    * The wrist and forearm proxies exist in the v1 XML with `contype=0`, and
      the trained policy parks them *inside* the joystick -- the carpal block
      on 46.9% of frames (deepest 1.73 mm), the distal ulna on 26.7%
      (1.92 mm), the distal radius on 9.2% (2.09 mm). Broken down by target,
      the wrist is inside the 2 mm **ball** on 47.1% of all frames and inside
      the stem on only 7.7%: the hand wraps the ball so tightly that the carpal
      block occupies it. That grasp is not physically realisable.

    What v2 is NOT predicted to fix, contrary to an earlier version of this
    docstring. The claim that the pass-through is *why* the policy pushes the
    stem (21.9% of delivered impulse, 58.7% of it from `Metacarpal_hand_2` at
    12.4 mm) was asserted without a test and is **refuted**: on a5's own
    trajectory P(a v2 geom is inside the joystick | stem contact is carrying
    force) is 0.086 against 0.742 given ball contact only -- 0.12x, where >2x
    was predicted. When the hand reaches down to the stem the wrist is 1.46 mm
    clear. In pose space the two classes are indistinguishable: 72.9% of
    stem-touching poses stay v2-feasible against 70.3% of ball-touching ones.
    Whether v2 changes where the hand pushes is an open question for the b1
    training arm, not a prediction.

    Checked, so the fix is not worse than the problem: v2 does not hold the
    hand off the ball. 70.3% of ball-touching poses in the reference
    neighbourhood stay feasible, and the deepest achievable ball press is
    unchanged at -2.364 mm (13,608 poses).

    Worth being explicit about what this does *not* fix. The joystick is
    translation-only (two slide joints with return springs on `joystick_base`;
    the `joystick` body has no joints), so a push on the stem moves the stick
    exactly as much as the same push on the ball. Contact height buys no
    leverage here, and v2 is not expected to increase force transmission. What
    it removes is a set of hand placements that no real forelimb could reach.

    Buffers are inherited at njmax=512 / naconmax=1024, which carried ~39%
    headroom over v1's measured peak of 57 contacts / 369 constraints per
    world. v2 adds 9 possible pairs (57 -> 66); re-measure before trusting the
    headroom under a different backend, since Warp treats naconmax as a batch
    total rather than a per-world budget.
    """
    cfg = default_config_scott_v1()
    cfg.walker_xml_path = janelia_scott_v2_xml_path(variant)
    return cfg


# Build a registry that inherits all parent entries, then override specific ones.
_registry = RewardRegistry()
_registry.rewards.update(_parent_registry.rewards)
_registry.terminations.update(_parent_registry.terminations)


class MouseImitationArmHand(MouseImitation):
    """MouseImitation variant that snaps shoulder-translation qpos/qvel dims
    to IK every step, leaving the joystick and the rest of the arm+hand chain
    fully muscle/contact-driven."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list, dict]]] = None,
        clips: Optional[MouseReferenceClips] = None,
    ) -> None:
        super().__init__(config, config_overrides, clips)

        ik_idx = list(self._config.ik_driven_qpos_idx)
        self._ik_idx = jp.array(ik_idx, dtype=int)
        nq = self.mjx_model.nq
        self._non_ik_idx = jp.array(
            [i for i in range(nq) if i not in ik_idx], dtype=int
        )

        if ik_idx:
            # Sanity: reference qpos at the IK-driven indices stays inside
            # those joints' own XML range (queried from the model, not
            # hardcoded -- v22's sh_tx/ty/tz range differs from the old
            # moving-shoulder model's slide joints).
            jnt_range = np.asarray(self._mj_model.jnt_range)[ik_idx]
            ref_vals = np.asarray(self.reference_clips.qpos[:, :, ik_idx])
            lo, hi = jnt_range[:, 0], jnt_range[:, 1]
            tol = 1e-4
            within = (ref_vals >= (lo - tol)) & (ref_vals <= (hi + tol))
            assert within.all(), (
                f"IK-driven qpos indices {ik_idx} exceed their XML joint "
                f"range. range={jnt_range.tolist()}, "
                f"observed min/max={ref_vals.min(axis=(0, 1)).tolist()}/"
                f"{ref_vals.max(axis=(0, 1)).tolist()}."
            )

        # Only set up for models that actually have joystick_contact enabled
        # (v25) -- v22/v22x/v23 don't have this reward term in their configs
        # and, for v22x specifically, don't have a joystick body at all.
        if "joystick_contact" in self._config.reward_terms:
            def _geom_setup(geoms):
                geom_ids, local_pos, radii, ref_body_names = [], [], [], []
                for geom_name, ref_body_name in geoms:
                    geom_id = self._mj_model.geom(geom_name + self._suffix).id
                    geom_ids.append(geom_id)
                    local_pos.append(self._mj_model.geom_pos[geom_id])
                    # size[0] is the radius for both sphere and capsule geoms
                    # (every joystick/grip geom here is one or the other) --
                    # treating the capsule shaft as a sphere of its own
                    # radius is a coarse approximation, fine for a distance
                    # proxy (see default_config_v25()'s reward_terms comment).
                    radii.append(self._mj_model.geom_size[geom_id][0])
                    ref_body_names.append(ref_body_name)
                return (
                    jp.array(geom_ids, dtype=int),
                    jp.array(np.stack(local_pos)),
                    jp.array(radii),
                    ref_body_names,
                )

            (
                self._joystick_geom_ids,
                self._joystick_geom_local_pos,
                self._joystick_geom_radii,
                self._joystick_ref_bodies,
            ) = _geom_setup(_JOYSTICK_GEOMS_V25)
            (
                self._grip_geom_ids,
                self._grip_geom_local_pos,
                self._grip_geom_radii,
                self._grip_ref_bodies,
            ) = _geom_setup(_GRIP_GEOMS_V25)

        if "joystick_pos" in self._config.reward_terms:
            self._joystick_qpos_idx = jp.array(
                self._config.joystick_qpos_idx, dtype=int
            )

    def _override_ik_dims(self, data: mjx.Data, info: Dict[str, Any]) -> mjx.Data:
        """Snap the IK-driven qpos/qvel indices to the reference for the current frame."""
        if self._ik_idx.shape[0] == 0:
            return data
        cur_frame = self._get_cur_frame(data, info)
        last_valid = self._clip_length() - 1
        cur_frame_clamped = jp.minimum(cur_frame, last_valid)
        ref = self.reference_clips.at(
            clip=info["reference_clip"], frame=cur_frame_clamped
        )
        data = data.replace(
            qpos=data.qpos.at[self._ik_idx].set(ref.qpos[self._ik_idx]),
            qvel=data.qvel.at[self._ik_idx].set(ref.qvel[self._ik_idx]),
        )
        # Refresh xpos/xquat so downstream consumers (rewards, obs) see the
        # snapped pose in world coordinates.
        data = mjx.forward(self.mjx_model, data)
        return data

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        state = super().reset(rng, clip_idx, start_frame)
        # Reference qpos already matches at start, but the snap is idempotent
        # and guards against any rounding drift.
        data = self._override_ik_dims(state.data, state.info)
        obs = self._get_obs(data, state.info)
        return state.replace(data=data, obs=obs)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        # Snap IK-driven dims BEFORE computing obs/reward/termination so that
        # every downstream consumer sees the kinematically correct pose.
        data = self._override_ik_dims(data, info)

        last_valid_frame = self._clip_length() - self._config.reference_length - 1
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jp.astype(truncated, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        terminated = self._is_done(data, info, state.metrics)
        done = jp.logical_or(terminated, info["truncated"])
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        current_frame = self._get_cur_frame(data, info)
        state.metrics["current_frame"] = jp.astype(current_frame, float)
        return state

    # ---- Reward / termination overrides: mask IK-driven indices ----

    @_registry.reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        distance = jp.linalg.norm(
            target.joints[self._non_ik_idx] - data.qpos[self._non_ik_idx]
        )
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_registry.reward("joints_vel")
    def _joints_vel_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        distance = jp.linalg.norm(
            target.joints_velocity[self._non_ik_idx] - data.qvel[self._non_ik_idx]
        )
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    @_registry.reward("joystick_pos")
    def _joystick_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """Dedicated tracking of the joystick's own qpos dims (x_slide,
        y_slide) against the reference -- see default_config_v25()'s
        reward_terms comment for why this needs to be separate from the
        generic "joints" reward rather than relying on that norm alone."""
        target = self._get_current_target(data, info)
        distance = jp.linalg.norm(
            target.joints[self._joystick_qpos_idx] - data.qpos[self._joystick_qpos_idx]
        )
        metrics["joystick_pos_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joystick_pos"] = reward
        return reward

    def _geom_world_pos(self, geom_ids, local_pos, ref_bodies, target=None, data=None):
        """World position of each geom in `geom_ids`.

        Sim side (`data` given): read straight from `data.geom_xpos`, already
        computed by mj_forward -- no need to redo the body->geom transform
        ourselves. Reference side (`target` given): reference_clips only
        stores body-level xpos/xquat, not per-geom, so reconstruct each
        geom's world position as body_xpos + rotate(local_offset, body_xquat).
        """
        if data is not None:
            return data.geom_xpos[geom_ids]
        return jp.stack(
            [
                target.body_xpos(body) + brax_math.rotate(local, target.body_xquat(body))
                for body, local in zip(ref_bodies, local_pos)
            ]
        )

    @_registry.reward("joystick_contact")
    def _joystick_contact_reward(
        self, data, info, metrics, weight, exp_scale, contact_threshold
    ) -> float:
        """Reward contact with the joystick where the reference/IK data says
        contact should be happening.

        Hard-gated (per Eric 2026-07-20): the reward is exactly 0 on frames
        where the reference data's own hand-to-joystick clearance says the
        real mouse isn't gripping, and only turns on (scaled by how close the
        sim's own clearance is to actually touching) on frames where it says
        it should be. See default_config_v25() for how contact_threshold was
        calibrated and why weight/exp_scale are starting points, not tuned.
        """
        target = self._get_current_target(data, info)

        # Combined radius per (grip, joystick) geom pair, so "clearance" below
        # is surface-to-surface (negative = interpenetrating), matching how
        # contact_threshold was calibrated -- not raw center-to-center
        # distance, which would be offset by ~1-3mm of geom radius and never
        # approach the threshold.
        combined_radii = (
            self._grip_geom_radii[:, None] + self._joystick_geom_radii[None, :]
        )

        ref_joystick_pos = self._geom_world_pos(
            self._joystick_geom_ids, self._joystick_geom_local_pos,
            self._joystick_ref_bodies, target=target,
        )
        ref_grip_pos = self._geom_world_pos(
            self._grip_geom_ids, self._grip_geom_local_pos,
            self._grip_ref_bodies, target=target,
        )
        ref_center_dist = jp.linalg.norm(
            ref_grip_pos[:, None, :] - ref_joystick_pos[None, :, :], axis=-1
        )
        ref_clearance = jp.min(ref_center_dist - combined_radii)
        should_contact = ref_clearance < contact_threshold

        sim_joystick_pos = self._geom_world_pos(
            self._joystick_geom_ids, self._joystick_geom_local_pos,
            self._joystick_ref_bodies, data=data,
        )
        sim_grip_pos = self._geom_world_pos(
            self._grip_geom_ids, self._grip_geom_local_pos,
            self._grip_ref_bodies, data=data,
        )
        sim_center_dist = jp.linalg.norm(
            sim_grip_pos[:, None, :] - sim_joystick_pos[None, :, :], axis=-1
        )
        sim_clearance = jp.min(sim_center_dist - combined_radii)

        metrics["joystick_contact_ref_clearance"] = ref_clearance
        metrics["joystick_contact_sim_clearance"] = sim_clearance
        metrics["joystick_should_contact"] = jp.astype(should_contact, float)
        reward = jp.where(
            should_contact, weight * jp.exp(-((sim_clearance / exp_scale) ** 2) / 2), 0.0
        )
        metrics["rewards/joystick_contact"] = reward
        return reward

    @_registry.termination("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        pose_error = jp.linalg.norm(
            target.joints[self._non_ik_idx] - data.qpos[self._non_ik_idx]
        )
        return pose_error > max_l2_error


def _mjx_contact(data):
    """`data.contact` across mujoco-mjx versions.

    3.4 moved it behind `data._impl` and deprecated the direct attribute; the
    direct one still works but warns on every call, which is unusable inside a
    reward evaluated every step.
    """
    impl = getattr(data, "_impl", None)
    return impl.contact if impl is not None else data.contact


# scott_v1 inherits every arm+hand reward/termination and overrides only
# joystick_contact below.
_scott_v1_registry = RewardRegistry()
_scott_v1_registry.rewards.update(_registry.rewards)
_scott_v1_registry.terminations.update(_registry.terminations)


class MouseImitationArmHandScottV1(MouseImitationArmHand):
    """scott_v1 variant: exact contact clearance instead of a radius proxy.

    v25's `_joystick_contact_reward` builds surface-to-surface clearance as
    `center_distance - (r_grip + r_joystick)`, reading each radius as
    `geom_size[id][0]`. That is only correct when every geom is a sphere.
    scott_v1's grip proxies are 6 ellipsoids and 13 capsules, where
    `geom_size[0]` is the *longest semi-axis* (3-5x the other two) or a capsule
    radius with its length ignored -- so the v25 term would be silently wrong
    on every pair, in a direction that varies per bone and per pose.

    Both sides are replaced with exact geometry:

    * **Sim side** -- `data.contact.dist`, MJX's own signed distance for each
      pair, exact for any geom type. MJX's jax pipeline enumerates a *static*
      pair list (verified: `contact.geom` is identical across poses), so the
      relevant entries can be located once at construction.

      Cross-checked against `mj_geomDistance` over 12 random poses x 57 pairs:
      the two agree to 0.052 mm max (0.001 mm median) at separations under
      2 mm, and diverge by up to 4.9 mm only beyond 5 mm, and only on the
      ellipsoid pairs that go through MJX's iterative SDF collider rather than
      an analytic one. The reward is `exp(-(d/0.75mm)^2/2)`, already ~0 by
      2 mm, so the disagreement lives entirely where the term is flat.

    * **Reference side** (the hard gate) -- precomputed once at construction
      with `mj_geomDistance`, which is exact for any pair. The reference
      clearance depends only on the clip data and never on the sim, so there
      is no reason to approximate it at all, let alone every step.

    The sim clearance is also clamped at 0. v25's term is an *even* function of
    clearance: it peaks at exactly touching and decays for interpenetration, so
    pressing into the joystick scored the same as hovering the same distance
    away. Over the 519M-step v25 run the grip deepened (min clearance -0.26 ->
    -0.56 mm) and this term's reward *fell* (0.0220 -> 0.0197) -- it was
    penalising the policy for gripping harder. Clamping makes "touching" and
    "pressing" score alike, which is the weakest change that removes the
    perverse gradient.
    """

    _registry = _scott_v1_registry

    def __init__(self, config=None, config_overrides=None, clips=None):
        super().__init__(
            config if config is not None else default_config_scott_v1(),
            config_overrides,
            clips,
        )
        if "joystick_contact" not in self._config.reward_terms:
            return

        m = self._mj_model
        grip = np.flatnonzero(m.geom_contype == contact_presets.GRIP_CONTYPE)
        joystick = np.flatnonzero(m.geom_contype == contact_presets.JOYSTICK_CONTYPE)
        if grip.size == 0 or joystick.size == 0:
            raise ValueError(
                "scott_v1 needs geoms carrying contype "
                f"{contact_presets.GRIP_CONTYPE} (grip) and "
                f"{contact_presets.JOYSTICK_CONTYPE} (joystick); found "
                f"{grip.size} and {joystick.size}."
            )
        self._grip_geom_ids_exact = grip
        self._joystick_geom_ids_exact = joystick

        self._contact_row_idx = self._locate_contact_rows(grip, joystick)
        self._ref_clearance_table = jp.asarray(
            self._precompute_reference_clearance(grip, joystick)
        )

    def _locate_contact_rows(self, grip, joystick):
        """Rows of MJX's static contact array that are grip<->joystick pairs.

        Runs one `mjx.forward` at construction to read the pair list. That list
        is fixed by `mjx.put_model`, not by the pose, so doing it once is
        correct -- and the alternative (reproducing MJX's broadphase pair
        enumeration here) would silently rot the first time it changed
        upstream.
        """
        data = mjx.make_data(
            self._mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        pairs = np.asarray(_mjx_contact(mjx.forward(self.mjx_model, data)).geom)
        grip_set, joystick_set = set(grip.tolist()), set(joystick.tolist())
        rows = np.flatnonzero(
            [
                (a in grip_set and b in joystick_set)
                or (a in joystick_set and b in grip_set)
                for a, b in pairs
            ]
        )
        if rows.size == 0:
            raise ValueError(
                "no grip<->joystick pair appears in MJX's contact list; the "
                "contact reward would be reading unrelated geoms."
            )
        return jp.asarray(rows, dtype=int)

    def _precompute_reference_clearance(self, grip, joystick):
        """Exact min surface-to-surface clearance at every reference pose.

        Shape (n_clips, n_frames), metres, negative = the reference hand is
        inside the joystick. Costs one `mj_forward` + `n_grip * n_joystick`
        `mj_geomDistance` calls per reference frame, once, at construction.
        """
        import mujoco  # local: only needed on the CPU-side setup path

        m, data = self._mj_model, mujoco.MjData(self._mj_model)
        qpos = np.asarray(self.reference_clips.qpos)
        n_clips, n_frames = qpos.shape[0], qpos.shape[1]
        out = np.empty((n_clips, n_frames), dtype=np.float32)
        # distmax only bounds the search; anything past it is far enough that
        # the reward is flat there anyway.
        distmax = 1.0
        for c in range(n_clips):
            for f in range(n_frames):
                data.qpos[:] = qpos[c, f]
                mujoco.mj_forward(m, data)
                out[c, f] = min(
                    mujoco.mj_geomDistance(m, data, int(g), int(j), distmax, None)
                    for g in grip
                    for j in joystick
                )
        return out

    @_scott_v1_registry.reward("joystick_contact")
    def _joystick_contact_reward(
        self, data, info, metrics, weight, exp_scale, contact_threshold
    ) -> float:
        """Reward touching the joystick on frames where the reference does.

        Hard-gated on the *reference*'s own clearance, so the term is exactly 0
        on frames where the real mouse was not gripping and cannot be farmed by
        simply holding the joystick the whole episode.
        """
        cur_frame = jp.minimum(
            self._get_cur_frame(data, info), self._clip_length() - 1
        )
        ref_clearance = self._ref_clearance_table[info["reference_clip"], cur_frame]
        should_contact = ref_clearance < contact_threshold

        # Clamped at 0: pressing scores the same as just touching, rather than
        # decaying back down the far side of a Gaussian centred on "barely in
        # contact" (see the class docstring).
        sim_clearance = jp.min(_mjx_contact(data).dist[self._contact_row_idx])
        gated_clearance = jp.maximum(sim_clearance, 0.0)

        metrics["joystick_contact_ref_clearance"] = ref_clearance
        metrics["joystick_contact_sim_clearance"] = sim_clearance
        metrics["joystick_should_contact"] = jp.astype(should_contact, float)
        reward = jp.where(
            should_contact,
            weight * jp.exp(-((gated_clearance / exp_scale) ** 2) / 2),
            0.0,
        )
        metrics["rewards/joystick_contact"] = reward
        return reward
