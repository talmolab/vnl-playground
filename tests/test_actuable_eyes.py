"""Tests for actuable eye cameras feature.

Validates that when actuable_eyes=True, the RunGapVision environment:
- Adds eye mount bodies with hinge joints on the skull
- Adds torque actuators for independent eye yaw control
- Places cameras on the mount bodies
- Preserves proprioception shape (eye state excluded from decoder input)
- Includes eye state in task_obs
"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from vnl_playground.tasks.rodent import run_gap_vision
from vnl_playground.tasks.wrappers import HighLevelWrapper


def _make_actuable_eye_env() -> run_gap_vision.RunGapVision:
    """Helper: create a RunGapVision env with actuable eyes enabled."""
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.actuable_eyes = True
    cfg.mujoco_impl = "warp"
    env = run_gap_vision.RunGapVision(config=cfg)
    return env


def _make_fixed_eye_env() -> run_gap_vision.RunGapVision:
    """Helper: create a RunGapVision env with fixed binocular cameras."""
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.actuable_eyes = False
    cfg.mujoco_impl = "warp"
    env = run_gap_vision.RunGapVision(config=cfg)
    return env


# ---------- Task 1: Config parameters ----------


def test_actuable_eyes_config_defaults():
    """Config params exist with correct defaults."""
    cfg = run_gap_vision.default_config()
    assert cfg.actuable_eyes is False
    assert cfg.eye_yaw_range == pytest.approx(0.698)
    assert cfg.eye_force_range == pytest.approx(0.01)
    assert cfg.eye_damping == pytest.approx(0.001)
    assert cfg.eye_stiffness == pytest.approx(0.0)


# ---------- Task 2: Eye joints, actuators, cameras ----------


def test_eye_joints_exist_in_model():
    """Eye yaw joints should be present after init with actuable_eyes=True."""
    env = _make_actuable_eye_env()
    m = env.mj_model
    suffix = env._suffix

    left_id = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, f"eye_left_yaw{suffix}"
    )
    right_id = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, f"eye_right_yaw{suffix}"
    )
    assert left_id >= 0, "eye_left_yaw joint not found"
    assert right_id >= 0, "eye_right_yaw joint not found"


def test_eye_actuators_exist_in_model():
    """Eye actuators should be appended at the end of the actuator list."""
    env = _make_actuable_eye_env()
    m = env.mj_model
    suffix = env._suffix

    left_id = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"eye_left_yaw{suffix}"
    )
    right_id = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"eye_right_yaw{suffix}"
    )
    assert left_id >= 0, "eye_left_yaw actuator not found"
    assert right_id >= 0, "eye_right_yaw actuator not found"

    # They should be the last two actuators
    assert left_id == m.nu - 2, f"Left eye actuator at {left_id}, expected {m.nu - 2}"
    assert right_id == m.nu - 1, f"Right eye actuator at {right_id}, expected {m.nu - 1}"


def test_eye_cameras_exist_on_mount_bodies():
    """Actuated cameras should exist and be attached to mount bodies."""
    env = _make_actuable_eye_env()
    m = env.mj_model
    suffix = env._suffix

    for side in ["left", "right"]:
        cam_name = f"eye_{side}_actuated{suffix}"
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        assert cam_id >= 0, f"Camera '{cam_name}' not found"

        body_name = f"eye_{side}_mount{suffix}"
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert body_id >= 0, f"Body '{body_name}' not found"

        # Camera should be on the mount body
        assert m.cam_bodyid[cam_id] == body_id, (
            f"Camera '{cam_name}' is on body {m.cam_bodyid[cam_id]}, "
            f"expected body {body_id} ('{body_name}')"
        )


def test_action_size_increased_by_two():
    """Action size should be original + 2 (one per eye actuator)."""
    fixed_env = _make_fixed_eye_env()
    actuable_env = _make_actuable_eye_env()

    assert actuable_env.action_size == fixed_env.action_size + 2


def test_n_eye_actuators_stored():
    """n_eye_actuators property should return 2 for actuable, 0 for fixed."""
    actuable_env = _make_actuable_eye_env()
    assert actuable_env.n_eye_actuators == 2

    fixed_env = _make_fixed_eye_env()
    assert fixed_env.n_eye_actuators == 0


def test_camera_names_point_to_actuated_cameras():
    """Config camera names should be updated to the actuated camera names."""
    env = _make_actuable_eye_env()
    suffix = env._suffix

    assert env._config.left_camera_name == f"eye_left_actuated{suffix}"
    assert env._config.right_camera_name == f"eye_right_actuated{suffix}"


# ---------- Task 3: Proprioception masking ----------


def test_proprioception_shape_unchanged():
    """Proprioceptive obs size should be the same with/without actuable eyes."""
    fixed_env = _make_fixed_eye_env()
    actuable_env = _make_actuable_eye_env()

    assert actuable_env.proprioceptive_obs_size == fixed_env.proprioceptive_obs_size


# ---------- Task 4: Eye state in observations ----------


def test_task_obs_includes_eye_state():
    """task_obs should be 4 larger: +2 from prev_action (wider) and +2 eye angles."""
    fixed_env = _make_fixed_eye_env()
    actuable_env = _make_actuable_eye_env()

    fixed_state = jax.jit(fixed_env.reset)(jax.random.PRNGKey(0))
    actuable_state = jax.jit(actuable_env.reset)(jax.random.PRNGKey(0))

    fixed_task_obs = fixed_state.obs["state"]["task_obs"]
    actuable_task_obs = actuable_state.obs["state"]["task_obs"]

    # +2 from prev_action being wider (38 + 2 = 40), +2 from eye angles
    assert actuable_task_obs.shape[0] == fixed_task_obs.shape[0] + 4


def test_eye_joint_angles_at_zero_on_reset():
    """Eye joint angles should be zero at reset."""
    env = _make_actuable_eye_env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    eye_angles = state.data.qpos[env._eye_qpos_indices]
    np.testing.assert_allclose(eye_angles, 0.0, atol=1e-6)


# ---------- Task 5: HighLevelWrapper eye action bypass ----------


def test_highlevel_wrapper_action_size_with_eyes():
    """HighLevelWrapper action_size should include eye actuators."""
    env = _make_actuable_eye_env()
    body_nu = env.action_size - env.n_eye_actuators

    def mock_decoder(x):
        return jp.zeros(body_nu), {}

    latent_size = 64
    hlw = HighLevelWrapper(
        env=env,
        decoder_inference_fn=mock_decoder,
        latent_size=latent_size,
        n_eye_actuators=env.n_eye_actuators,
        pass_vision=True,
        pass_task_obs=True,
    )
    assert hlw.action_size == latent_size + env.n_eye_actuators


def test_highlevel_wrapper_action_size_without_eyes():
    """HighLevelWrapper action_size should be latent_size when no eye actuators."""
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.mujoco_impl = "warp"
    env = run_gap_vision.RunGapVision(config=cfg)
    body_nu = env.action_size

    def mock_decoder(x):
        return jp.zeros(body_nu), {}

    latent_size = 64
    hlw = HighLevelWrapper(
        env=env,
        decoder_inference_fn=mock_decoder,
        latent_size=latent_size,
        pass_vision=True,
        pass_task_obs=True,
    )
    assert hlw.action_size == latent_size


# ---------- Task 6: Task registration ----------


def test_actuable_eyes_task_registered():
    """RodentRunGapActuableEyes should be in the task registry."""
    from vnl_playground import tasks
    cfg = tasks.get_default_config("RodentRunGapActuableEyes")
    assert cfg.actuable_eyes is True
    assert cfg.binocular is True


# ---------- Task 7: Integration smoke tests ----------


@pytest.mark.slow
def test_smoke_actuable_eyes_reset_step():
    """Env should reset and step without errors. Eye joints should respond to control."""
    env = _make_actuable_eye_env()
    rng = jax.random.PRNGKey(42)

    # Reset
    state = jax.jit(env.reset)(rng)
    assert state.obs is not None

    # Verify observation shapes
    task_obs = state.obs["state"]["task_obs"]
    vision = state.obs["state"]["vision"]
    assert task_obs.ndim == 1
    assert vision.shape == env.vision_shape

    # Step with eye actuators at max outward
    action = jp.zeros(env.action_size)
    action = action.at[-2:].set(1.0)

    step_fn = jax.jit(env.step)
    state2 = step_fn(state, action)
    assert state2.obs is not None


@pytest.mark.slow
def test_smoke_eye_angles_change_with_sustained_control():
    """Sustained eye control should produce measurable joint angle change."""
    env = _make_actuable_eye_env()
    rng = jax.random.PRNGKey(42)

    state = jax.jit(env.reset)(rng)

    # Apply sustained outward eye control for 10 steps
    action = jp.zeros(env.action_size)
    action = action.at[-2:].set(1.0)

    step_fn = jax.jit(env.step)
    for _ in range(10):
        state = step_fn(state, action)

    # Eye joint angles should be noticeably non-zero after 10 steps of torque
    eye_angles = state.obs["state"]["task_obs"][-2:]
    assert jp.any(jp.abs(eye_angles) > 0.001), (
        f"Eye angles too small after sustained control: {eye_angles}"
    )
