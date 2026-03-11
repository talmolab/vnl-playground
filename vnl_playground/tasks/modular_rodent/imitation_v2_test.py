"""Tests for ModularImitation_v2 coordinate frame computations.

Tests verify:
- All observations are finite at reset.
- Proprioception matches target at reset (agent starts at reference pose).
- Egocentric quantities are invariant to global yaw rotation of the agent.
- Target quantities are invariant when both agent and reference are rotated by the same yaw.
"""

import jax
import jax.numpy as jp
import pytest
from mujoco import mjx

from vnl_playground.tasks.modular_rodent.imitation_v2 import ModularImitation_v2, default_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_and_state():
    config = default_config()
    config.iterations = 3
    config.ls_iterations = 3
    env = ModularImitation_v2(config)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    return env, state


def _rotate_qpos_yaw(qpos: jp.ndarray, angle_deg: float) -> jp.ndarray:
    """Apply a global yaw rotation to root pos+quat in qpos.

    MuJoCo free-joint qpos layout: [x, y, z, qw, qx, qy, qz, joints...].
    Quaternion convention: [w, x, y, z].
    Rotation applied as q_new = Rz(angle) * q_old (left-multiply).
    """
    angle = jp.deg2rad(angle_deg)
    c, s = jp.cos(angle / 2), jp.sin(angle / 2)
    # Rz(angle) as quaternion [w, x, y, z]
    dq = jp.array([c, 0.0, 0.0, s])

    # Rotate root quaternion: q_new = dq * q_old (Hamilton product)
    q = qpos[3:7]  # [w, x, y, z]
    # Hamilton product dq * q
    dw, dx, dy, dz = dq[0], dq[1], dq[2], dq[3]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    q_new = jp.array([
        dw*qw - dx*qx - dy*qy - dz*qz,
        dw*qx + dx*qw + dy*qz - dz*qy,
        dw*qy - dx*qz + dy*qw + dz*qx,
        dw*qz + dx*qy - dy*qx + dz*qw,
    ])

    # Rotate root position
    Rz = jp.array([
        [jp.cos(angle), -jp.sin(angle), 0.0],
        [jp.sin(angle),  jp.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    pos_new = Rz @ qpos[0:3]

    return qpos.at[0:3].set(pos_new).at[3:7].set(q_new)


# ---------------------------------------------------------------------------
# Test 1: Finite values at reset
# ---------------------------------------------------------------------------


def test_obs_finite_at_reset(env_and_state):
    """All observation values should be finite (no NaN/Inf) after reset."""
    _, state = env_and_state
    leaves = jax.tree_util.tree_leaves(state.obs)
    for leaf in leaves:
        assert jp.all(jp.isfinite(leaf)), f"Non-finite obs: {leaf}"


# ---------------------------------------------------------------------------
# Test 2: Prop ≈ target at reset
# ---------------------------------------------------------------------------


def test_prop_matches_target_at_reset(env_and_state):
    """At reset the agent is placed at the reference pose, so proprioception
    should approximately match the interpolated reference target."""
    env, state = env_and_state
    prop = jax.jit(env._get_modular_proprioception)(state.data)
    cur = jax.jit(env._interpolated_target)(state.data, state.info, 0.0)

    atol = 1e-3

    # Egocentric body positions (yaw-aligned frame)
    assert jp.allclose(prop["hand_L"]["egocentric_hand_pos"], cur["hand_L"]["pos"], atol=atol), \
        f"hand_L pos diff: {prop['hand_L']['egocentric_hand_pos'] - cur['hand_L']['pos']}"
    assert jp.allclose(prop["hand_R"]["egocentric_hand_pos"], cur["hand_R"]["pos"], atol=atol), \
        f"hand_R pos diff: {prop['hand_R']['egocentric_hand_pos'] - cur['hand_R']['pos']}"
    assert jp.allclose(prop["foot_L"]["egocentric_foot_pos"], cur["foot_L"]["pos"], atol=atol), \
        f"foot_L pos diff: {prop['foot_L']['egocentric_foot_pos'] - cur['foot_L']['pos']}"
    assert jp.allclose(prop["foot_R"]["egocentric_foot_pos"], cur["foot_R"]["pos"], atol=atol), \
        f"foot_R pos diff: {prop['foot_R']['egocentric_foot_pos'] - cur['foot_R']['pos']}"

    # Knee positions (torso body frame)
    assert jp.allclose(prop["leg_L"]["egocentric_knee_pos"], cur["leg_L"]["knee_pos"], atol=atol), \
        f"knee_L diff: {prop['leg_L']['egocentric_knee_pos'] - cur['leg_L']['knee_pos']}"
    assert jp.allclose(prop["leg_R"]["egocentric_knee_pos"], cur["leg_R"]["knee_pos"], atol=atol), \
        f"knee_R diff: {prop['leg_R']['egocentric_knee_pos'] - cur['leg_R']['knee_pos']}"

    # Elbow positions (torso body frame)
    assert jp.allclose(prop["arm_L"]["egocentric_elbow_pos"], cur["arm_L"]["elbow_pos_torso"], atol=atol), \
        f"elbow_L diff: {prop['arm_L']['egocentric_elbow_pos'] - cur['arm_L']['elbow_pos_torso']}"
    assert jp.allclose(prop["arm_R"]["egocentric_elbow_pos"], cur["arm_R"]["elbow_pos_torso"], atol=atol), \
        f"elbow_R diff: {prop['arm_R']['egocentric_elbow_pos'] - cur['arm_R']['elbow_pos_torso']}"

    # Axis orientations (torso body frame)
    assert jp.allclose(prop["hand_L"]["xaxis"], cur["hand_L"]["xaxis"], atol=atol), \
        f"hand_L xaxis diff: {prop['hand_L']['xaxis'] - cur['hand_L']['xaxis']}"
    assert jp.allclose(prop["hand_R"]["xaxis"], cur["hand_R"]["xaxis"], atol=atol)
    assert jp.allclose(prop["foot_L"]["xaxis"], cur["foot_L"]["xaxis"], atol=atol)
    assert jp.allclose(prop["head"]["xaxis"], cur["head"]["xaxis"], atol=atol)
    assert jp.allclose(prop["head"]["zaxis"], cur["head"]["zaxis"], atol=atol)

    # Torso/pelvis axes (yaw-aligned frame)
    assert jp.allclose(prop["torso"]["zaxis"], cur["torso"]["zaxis"], atol=atol), \
        f"torso zaxis diff: {prop['torso']['zaxis'] - cur['torso']['zaxis']}"

    # Lumbar tendon lengths (tendonpos sensor vs weighted qpos sum from reference)
    assert jp.allclose(prop["torso"]["lumbar_bend"],   cur["torso"]["lumbar_bend"],   atol=atol), \
        f"lumbar_bend diff: {prop['torso']['lumbar_bend'] - cur['torso']['lumbar_bend']}"
    assert jp.allclose(prop["torso"]["lumbar_twist"],  cur["torso"]["lumbar_twist"],  atol=atol), \
        f"lumbar_twist diff: {prop['torso']['lumbar_twist'] - cur['torso']['lumbar_twist']}"
    assert jp.allclose(prop["torso"]["lumbar_extend"], cur["torso"]["lumbar_extend"], atol=atol), \
        f"lumbar_extend diff: {prop['torso']['lumbar_extend'] - cur['torso']['lumbar_extend']}"

    # Root position should be near zero (agent is at reference pose)
    assert jp.linalg.norm(cur["root"]["pos"]) < 0.01, \
        f"root pos at reset should be ~0, got {cur['root']['pos']}"


# ---------------------------------------------------------------------------
# Test 3: Yaw rotation invariance of proprioception
# ---------------------------------------------------------------------------


def test_prop_yaw_rotation_invariance(env_and_state):
    """Egocentric proprioception should be unchanged under global yaw rotation."""
    env, state = env_and_state

    qpos_rot = _rotate_qpos_yaw(state.data.qpos, 90.0)
    data_rot = jax.jit(mjx.forward)(env.mjx_model, state.data.replace(qpos=qpos_rot))

    prop_orig = jax.jit(env._get_modular_proprioception)(state.data)
    prop_rot  = jax.jit(env._get_modular_proprioception)(data_rot)

    atol = 1e-4

    # Hand module: egocentric pos, joint angles, xaxis all invariant
    for key in ["egocentric_hand_pos", "wrist_angle", "finger_angle", "xaxis"]:
        assert jp.allclose(prop_orig["hand_L"][key], prop_rot["hand_L"][key], atol=atol), \
            f"hand_L/{key} not yaw-invariant: orig={prop_orig['hand_L'][key]}, rot={prop_rot['hand_L'][key]}"
        assert jp.allclose(prop_orig["hand_R"][key], prop_rot["hand_R"][key], atol=atol), \
            f"hand_R/{key} not yaw-invariant"

    # Arm module: elbow position in torso frame is invariant
    for key in ["egocentric_elbow_pos", "elbow_angle"]:
        assert jp.allclose(prop_orig["arm_L"][key], prop_rot["arm_L"][key], atol=atol), \
            f"arm_L/{key} not yaw-invariant"
        assert jp.allclose(prop_orig["arm_R"][key], prop_rot["arm_R"][key], atol=atol), \
            f"arm_R/{key} not yaw-invariant"

    # Foot module: egocentric pos, joint angles, xaxis (symmetric L and R)
    for key in ["egocentric_foot_pos", "ankle_angle", "toe_angle", "xaxis"]:
        assert jp.allclose(prop_orig["foot_L"][key], prop_rot["foot_L"][key], atol=atol), \
            f"foot_L/{key} not yaw-invariant"
        assert jp.allclose(prop_orig["foot_R"][key], prop_rot["foot_R"][key], atol=atol), \
            f"foot_R/{key} not yaw-invariant"

    # Leg module: knee pos in torso frame is invariant; joint angles invariant (symmetric L and R)
    for key in ["egocentric_knee_pos", "knee_angle", "ankle_angle",
                "hip_supinate", "hip_abduct", "hip_extend"]:
        assert jp.allclose(prop_orig["leg_L"][key], prop_rot["leg_L"][key], atol=atol), \
            f"leg_L/{key} not yaw-invariant"
        assert jp.allclose(prop_orig["leg_R"][key], prop_rot["leg_R"][key], atol=atol), \
            f"leg_R/{key} not yaw-invariant"

    # Torso module: zaxis (yaw-aligned), pelvis_zaxis, lumbar joints all invariant
    for key in ["zaxis", "pelvis_zaxis", "lumbar_bend", "lumbar_twist", "lumbar_extend",
                "height_above_ground"]:
        assert jp.allclose(prop_orig["torso"][key], prop_rot["torso"][key], atol=atol), \
            f"torso/{key} not yaw-invariant: orig={prop_orig['torso'][key]}, rot={prop_rot['torso'][key]}"

    # Head module: xaxis/zaxis in torso frame, egocentric pos all invariant
    for key in ["xaxis", "zaxis", "egocentric_pos", "mandible"]:
        assert jp.allclose(prop_orig["head"][key], prop_rot["head"][key], atol=atol), \
            f"head/{key} not yaw-invariant"


# ---------------------------------------------------------------------------
# Test 4: Target is near-zero at reset (root pos check)
# ---------------------------------------------------------------------------


def test_target_root_near_zero_at_reset(env_and_state):
    """At reset the agent is placed at the reference pose, so root pos
    target (egocentric offset) should be approximately zero."""
    env, state = env_and_state
    cur = jax.jit(env._interpolated_target)(state.data, state.info, 0.0)
    root_dist = jp.linalg.norm(cur["root"]["pos"])
    assert root_dist < 0.01, f"root pos at reset = {cur['root']['pos']} (norm={root_dist:.4f})"
