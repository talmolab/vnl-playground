"""Tests for configurable eye camera angle offset.

Validates that _configure_eye_cameras() correctly yaws the eye cameras
by checking the compiled model's camera look directions (cam_mat0).
"""

import mujoco
import numpy as np
import pytest
import jax

from vnl_playground.tasks.rodent import run_gap_vision


def test_default_config_has_eye_angle_offset():
    """eye_angle_offset should exist in default config with value 0.2 (backward compat)."""
    cfg = run_gap_vision.default_config()
    assert "eye_angle_offset" in cfg
    assert cfg.eye_angle_offset == 0.2


def _make_vision_env(eye_angle_offset: float) -> run_gap_vision.RunGapVision:
    """Helper: create a RunGapVision env with a given eye angle offset."""
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.eye_angle_offset = eye_angle_offset
    cfg.mujoco_impl = "warp"
    env = run_gap_vision.RunGapVision(config=cfg)
    return env


def _get_camera_look_dir(mj_model, cam_name):
    """Get the look direction of a camera from the compiled model.

    MuJoCo camera convention: -Z axis is the look direction.
    cam_mat0 is the 3x3 rotation matrix (row-major) mapping camera frame to world.
    """
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    assert cam_id >= 0, f"Camera '{cam_name}' not found in model"
    R = mj_model.cam_mat0[cam_id].reshape(3, 3)
    look = R @ np.array([0.0, 0.0, -1.0])
    return look


def _get_camera_up_dir(mj_model, cam_name):
    """Get the up direction of a camera from the compiled model."""
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R = mj_model.cam_mat0[cam_id].reshape(3, 3)
    up = R @ np.array([0.0, -1.0, 0.0])  # MuJoCo: -Y_cam = up
    return up


def test_eye_cameras_yawed_with_custom_offset():
    """Camera look directions should be yawed by the configured offset."""
    offset = 0.35  # 20° offset → 40° overlap
    env = _make_vision_env(offset)

    look_left = _get_camera_look_dir(env.mj_model, "eye_left-rodent")
    look_right = _get_camera_look_dir(env.mj_model, "eye_right-rodent")

    # Left eye should be yawed left (+Y in skull frame)
    expected_left = np.array([np.cos(offset), np.sin(offset), 0.0])
    expected_right = np.array([np.cos(offset), -np.sin(offset), 0.0])

    np.testing.assert_allclose(look_left, expected_left, atol=1e-4)
    np.testing.assert_allclose(look_right, expected_right, atol=1e-4)

    # Up vectors should have no roll (pointing along skull -Z)
    up_left = _get_camera_up_dir(env.mj_model, "eye_left-rodent")
    up_right = _get_camera_up_dir(env.mj_model, "eye_right-rodent")
    np.testing.assert_allclose(up_left, [0, 0, -1], atol=1e-4)
    np.testing.assert_allclose(up_right, [0, 0, -1], atol=1e-4)


def test_default_offset_looks_correct():
    """Default offset=0.2 should yaw eyes by ~11.5° with no roll."""
    env = _make_vision_env(0.2)

    look_left = _get_camera_look_dir(env.mj_model, "eye_left-rodent")
    look_right = _get_camera_look_dir(env.mj_model, "eye_right-rodent")

    # Left eye yawed 0.2 rad left, right eye yawed 0.2 rad right
    np.testing.assert_allclose(
        look_left, [np.cos(0.2), np.sin(0.2), 0.0], atol=1e-4
    )
    np.testing.assert_allclose(
        look_right, [np.cos(0.2), -np.sin(0.2), 0.0], atol=1e-4
    )


def test_zero_offset_makes_eyes_look_straight_ahead():
    """offset=0 should make both eyes look exactly like egocentric (forward)."""
    env = _make_vision_env(0.0)

    look_ego = _get_camera_look_dir(env.mj_model, "egocentric-rodent")
    look_left = _get_camera_look_dir(env.mj_model, "eye_left-rodent")
    look_right = _get_camera_look_dir(env.mj_model, "eye_right-rodent")

    np.testing.assert_allclose(look_left, look_ego, atol=1e-4)
    np.testing.assert_allclose(look_right, look_ego, atol=1e-4)

    # All up vectors should be [0, 0, -1] (no roll)
    for name in ["egocentric-rodent", "eye_left-rodent", "eye_right-rodent"]:
        up = _get_camera_up_dir(env.mj_model, name)
        np.testing.assert_allclose(up, [0, 0, -1], atol=1e-4)


def test_config_override_eye_angle_offset():
    """eye_angle_offset should be overridable via config_overrides (simulates env_args)."""
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.mujoco_impl = "warp"
    env = run_gap_vision.RunGapVision(
        config=cfg,
        config_overrides={"eye_angle_offset": 0.5},
    )
    assert env._config.eye_angle_offset == 0.5

    look_left = _get_camera_look_dir(env.mj_model, "eye_left-rodent")
    np.testing.assert_allclose(
        look_left, [np.cos(0.5), np.sin(0.5), 0.0], atol=1e-4
    )


def test_binocular_default_config_has_eye_angle_offset():
    """The registered binocular config should include eye_angle_offset."""
    from vnl_playground.tasks import _cfgs
    cfg = _cfgs["RodentRunGapBinocularVision"]()
    assert "eye_angle_offset" in cfg
    assert cfg.eye_angle_offset == 0.2
    assert cfg.binocular is True


def test_invalid_eye_angle_offset_raises():
    """Negative or >π/2 offsets should raise ValueError."""
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.mujoco_impl = "warp"

    cfg.eye_angle_offset = -0.1
    with pytest.raises(ValueError, match="eye_angle_offset must be in"):
        run_gap_vision.RunGapVision(config=cfg)

    cfg.eye_angle_offset = 2.0
    with pytest.raises(ValueError, match="eye_angle_offset must be in"):
        run_gap_vision.RunGapVision(config=cfg)


@pytest.mark.slow
def test_smoke_binocular_env_with_custom_overlap():
    """Env should reset without errors for representative overlap values."""
    for offset in [0.0, 0.2, 0.698]:
        env = _make_vision_env(offset)
        rng = jax.random.PRNGKey(42)
        state = jax.jit(env.reset)(rng)
        assert state.obs is not None, f"Failed for offset={offset}"
