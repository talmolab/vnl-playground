"""Defines mouse constants (filesystem paths only).

Model-specific constants (body names, joint names, tracked bodies) should be
specified via YAML config files for flexibility in parameter sweeps.
See vnl_playground/config/mouse_imitation.yaml for an example.
"""

import os
from typing import Optional

from etils import epath

MOUSE_PATH = epath.Path(__file__).parent

# --- Janelia bone meshes (deliberately NOT committed) ------------------------
#
# The Janelia mouse model's ~107 bone .obj meshes are an unreleased asset and
# have never been committed to this repo (verified 2026-07-28 across every ref
# in history). The v22-v26 XMLs therefore carry an absolute meshdir pointing
# into Eric's container -- /root/vast/eric/janelia_model/{v22,v24} -- which
# resolves there and nowhere else, so a fresh clone fails to compile the model
# with no explanation of why.
#
# `janelia_mesh_dir()` supplies the path at load time instead; MouseBaseEnv's
# _load_walker_spec() applies it. Keeping the meshes out of git is a hard
# requirement, so the in-repo location below is gitignored: the meshes can sit
# in the source tree for convenience without any way to commit them by accident.
JANELIA_MESH_DIR_ENV = "JANELIA_MODEL_DIR"
JANELIA_MESH_DIR_LOCAL = MOUSE_PATH / "xmls" / "assets" / "janelia_model_v24"


def janelia_mesh_dir() -> Optional[str]:
    """Where the Janelia bone meshes live on this machine.

    Resolution order, first hit wins:
      1. $JANELIA_MODEL_DIR -- explicit override, for meshes kept elsewhere
         (a NAS mount, a shared scratch dir).
      2. JANELIA_MESH_DIR_LOCAL -- the gitignored in-repo drop-in.
      3. None -- caller keeps the XML's own meshdir untouched.

    Returns None rather than raising so that Eric's container (where the XML's
    absolute meshdir does resolve) and the non-Janelia mouse XMLs (which use
    relative meshdirs) are both unaffected.
    """
    env_dir = os.environ.get(JANELIA_MESH_DIR_ENV)
    if env_dir:
        return env_dir
    if JANELIA_MESH_DIR_LOCAL.exists():
        return str(JANELIA_MESH_DIR_LOCAL)
    return None

MOUSE_XML_PATH = MOUSE_PATH / "xmls" / "akira_muscle.xml"
JANELIA_MOUSE_XML_PATH = MOUSE_PATH / "xmls" / "mouse_forelimb_right.xml"
JANELIA_AKIRA_XML_PATH = MOUSE_PATH / "xmls" / "janelia_akira.xml"
MOUSE_ARENA_XML_PATH = MOUSE_PATH / "xmls" / "arena.xml"
MOUSE_REFERENCE_DATA_PATH = MOUSE_PATH / "reference_data"
JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_moving_shoulder_ik.xml"
)
MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH = (
    MOUSE_PATH / "reference_data_moving_shoulder"
)

# v22 arm+hand+joystick model (35 muscles, 27 joints) and its STAC v22-native
# reference data. This points at a vnl-playground-local PATCHED copy, not the
# shared reference model at /root/vast/eric/janelia_model/v22/ -- the patch
# only corrects the joystick_base position/orientation/geom sizes (synced
# from stac-mjx's mouse_forelimb_right_janelia_arm_hand_v22_stac.xml, the
# model STAC actually fit against, after stac-mjx fixed a mocap-registration
# bug on 2026-07-16). Every other body in this file is byte-identical to the
# shared reference model. Never edit the shared reference model directly;
# edit this copy instead.
#
# 2026-07-17: points at mouse_forelimb_right_janelia_arm_hand_v22_final_camera.xml,
# NOT mouse_forelimb_right_janelia_arm_hand_v22_contacts_patched.xml directly
# -- it's the same file (same untilted, hand-shifted joystick_base that
# trained cleanly to 500M steps, reward -49 -> 305, 0% NaN, per
# logs/janelia_v22_arm_hand_full_run.log) plus one addition: a fixed
# diagnostic camera (arm_joystick_fixed_view), re-derived for this file's
# untilted joystick pose. Per Eric: keep the joystick untilted (matches every
# run that's actually trained), only fix the camera -- do not point this at
# _raw_joystick_camera.xml, whose camera is rolled to compensate for a
# ~38.6-degree tilt this file does not have.
#
# Reference data points directly at stac-mjx's own output directory (not a
# vnl-playground-local copy) since it's genuine STAC-v22-native qpos/xpos
# (fit directly against v22, not transplanted from v21) and may grow as more
# trials finish there -- as of 2026-07-16 only the 5 CFL_35 trials are done;
# CFL_36/37 (10 more trials) may land later. The old
# /root/vast/eric/refined_STACed_data_v22/ (this repo's own
# scripts/convert_stac_v21_to_v22.py output, transplanting v21's STAC fit
# into v22 by name-matching qpos) is superseded -- that whole approach
# assumed v21's qpos values meant the same thing in v22, which turned out
# not to fully hold; this data doesn't need any such transplant since it's
# fit natively against v22.
JANELIA_MOUSE_ARM_HAND_V22_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_janelia_arm_hand_v22_final_camera.xml"
)
MOUSE_REFERENCE_DATA_JANELIA_V22_PATH = epath.Path(
    "/root/vast/eric/stac-mjx/refined_STACed_data_v22"
)

# v22x, 2026-07-17: v22 arm+hand with the joystick removed entirely (nq 27
# -> 25), to validate arm+hand reach-tracking on its own before tackling
# hand-joystick contact dynamics again in v23. Reference data is this repo's
# own conversion (scripts/convert_stac_v22_drop_joystick.py) of the v22-native
# STAC data with the joystick's x_slide/y_slide columns dropped by name --
# root-level, outside the repo, matching the established convention.
JANELIA_MOUSE_ARM_HAND_V22X_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_janelia_arm_hand_v22x_no_joystick.xml"
)
MOUSE_REFERENCE_DATA_JANELIA_V22X_PATH = epath.Path(
    "/root/vast/eric/refined_STACed_data_v22x_no_joystick"
)

# v23, 2026-07-17: v22 arm+hand+joystick with simplified primitive collision
# geoms (ellipsoid/capsule, matched by mesh name from a reference full-body
# rig Eric provided) replacing the raw bone meshes as collision geoms on all
# 38 wrist/hand bones -- mesh-vs-primitive collision generates far more
# broadphase candidate contacts per touching pair than primitive-vs-primitive,
# which was the likely real driver of the "broadphase overflow" warnings
# (490k+ over one v22 training run) rather than which bones were enabled.
# Visual bone meshes are now non-colliding (contype=0/conaffinity=0); the new
# "*_col" primitives carry the joystick-collision flags instead. Reuses v22's
# reference data (same arm/hand kinematic tree, joystick unaffected).
JANELIA_MOUSE_ARM_HAND_V23_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_janelia_arm_hand_v23_simplified_collision.xml"
)

# v22 raw-joystick + fixed camera, 2026-07-17: same mesh collision geoms as
# JANELIA_MOUSE_ARM_HAND_V22_XML_PATH (the config that already trained
# cleanly to 500M steps, reward -49 -> 305, per
# logs/janelia_v22_arm_hand_full_run.log -- the broadphase/nefc overflow
# warnings there were benign, not a training blocker), but with the
# joystick_base position-translation patch reverted back to the raw
# stac-mjx-synced pos/quat (same revert as v23, minus v23's collision-geom
# swap) and the fixed diagnostic camera (arm_joystick_fixed_view) added for
# legible eval renders despite the joystick's ~38.6-degree shaft tilt.
JANELIA_MOUSE_ARM_HAND_V22_RAW_JOYSTICK_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_janelia_arm_hand_v22_raw_joystick_camera.xml"
)

# v24 arm+hand+neck+head model (52 muscles, 25 joints), no joystick -- a
# structurally different rebuild from v22/v23 (single fixed root "Armature",
# anatomically-named rotational joints, no shoulder-translation slides, so
# none of v22's multi-root/ik_driven_qpos_idx machinery applies here). Points
# at a vnl-playground-local muscle-only copy, not
# /root/vast/eric/janelia_model/v24/forearm_v24.xml directly -- that source
# file also ships position/velocity/motor actuators (nu=127 total) alongside
# the 52 muscles, which base.py's action_size (=mjx_model.nu, no group
# filtering) would expose to the policy as direct joint torque/PD control in
# addition to muscles. This copy strips those, keeping only the 52 muscle
# actuators (nu=52), consistent with how v22/akira treat muscle-driven
# imitation. See the file's own header comment for the full body/joint
# structure (verified via mj_forward, not assumed from XML text).
JANELIA_MOUSE_V24_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_janelia_v24_muscle_only.xml"
)

JANELIA_MOUSE_ARM_HAND_V25_XML_PATH = (
    MOUSE_PATH / "xmls" / "mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml"
)
# STAC v25-native reference data (v24 arm+hand+neck+head rig + joystick,
# fit directly against mouse_forelimb_right_janelia_arm_hand_v25_stac.xml --
# same kinematic tree as JANELIA_MOUSE_ARM_HAND_V25_XML_PATH except
# joystick_base's pos, same "vnl-playground copy patches the joystick
# position" pattern as v22, see imitation_arm_hand.py's default_config_v25()).
# As of 2026-07-17, 6 trials have a complete *_ik.h5 (126 frames each);
# CFL_35_20240128_trial_0001 is still fit-only (_fit.h5, no _ik.h5 yet) and
# is skipped automatically by MouseReferenceClips' *_ik.h5 glob.
MOUSE_REFERENCE_DATA_JANELIA_V25_PATH = epath.Path(
    os.environ.get(
        "VNL_STAC_V25_DIR", "/root/vast/eric/stac-mjx/refined_STACed_data_v25"
    )
)

# scott_v1 (2026-07-29): the v25 rig with its 5 hand-sphere grip geoms replaced
# by 19 per-bone minimum-volume primitives -- 6 ellipsoid + 13 capsule, the
# "mixed" variant, emitted from the v25 template by
# analysis/2026-07-28-scott-v1-hand-collision-ellipsoids/scripts/emit_xml.py.
# Kinematic tree, qpos layout and STAC clips are identical to v25 -- only the
# collision geoms differ -- so MOUSE_REFERENCE_DATA_JANELIA_V25_PATH is reused
# unchanged. The version counter is independent of Eric's v22->v26 line: this
# is scott_v1, not "v27".
def janelia_scott_v1_xml_path(variant: str = "mixed"):
    """Walker XML for a scott_v1 collision variant.

    The three variants differ only in which primitive each hand bone's proxy
    is, and that choice has a real cost in MJX's pure-JAX pipeline: the
    joystick is two capsules and a sphere, so capsule/sphere pairs take the
    closed-form `collision_primitive` path while every *ellipsoid* pair falls
    through to `collision_sdf`, which runs 10 gradient-descent steps with a
    10-point line search and an autodiff gradient per pair.

        variant     ellipsoid grip geoms   SDF pairs (of 57)
        capsule                        0                   0
        mixed                          6                  18
        ellipsoid                     19                  57

    The kinematic contact probe found capsule and mixed within a few percent of
    each other on every transmission measure -- they differ on 6 of 19 geoms
    and none of them is the palm -- so `capsule` is the cheap escape hatch if
    the SDF cost dominates.
    """
    if variant not in ("mixed", "capsule", "ellipsoid"):
        raise ValueError(
            f"unknown scott_v1 variant {variant!r}; expected one of "
            "'mixed', 'capsule', 'ellipsoid'"
        )
    return (
        MOUSE_PATH
        / "xmls"
        / f"mouse_forelimb_right_janelia_scott_v1_{variant}_arm_hand_joystick.xml"
    )


JANELIA_MOUSE_ARM_HAND_SCOTT_V1_XML_PATH = janelia_scott_v1_xml_path("mixed")

# STAC v24-native reference data. As of 2026-07-17 the STAC v24 fitting job is
# still running -- only 6 of the eventual trial set exist on disk, and only
# CFL_35_20240128_trial_0101 has the full native 126 frames; the other 5 are
# 15-frame in-progress stubs. imitation_v24.py filters to full-length trials
# dynamically (not a hardcoded index list like v22x/v23's keep_clips_idx)
# since more trials will complete over time.
MOUSE_REFERENCE_DATA_JANELIA_V24_PATH = epath.Path(
    "/root/vast/eric/stac-mjx/refined_STACed_data_v24"
)
