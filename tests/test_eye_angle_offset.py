"""Tests for configurable eye camera angle offset."""

import numpy as np
import pytest
from vnl_playground.tasks.rodent import run_gap_vision


def test_default_config_has_eye_angle_offset():
    """eye_angle_offset should exist in default config with value 0.2 (backward compat)."""
    cfg = run_gap_vision.default_config()
    assert "eye_angle_offset" in cfg
    assert cfg.eye_angle_offset == 0.2


import jax


def _make_vision_env(eye_angle_offset: float) -> run_gap_vision.RunGapVision:
    """Helper: create a RunGapVision env with a given eye angle offset."""
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.eye_angle_offset = eye_angle_offset
    cfg.mujoco_impl = "warp"
    env = run_gap_vision.RunGapVision(config=cfg)
    return env


def test_eye_cameras_modified_with_custom_offset():
    """Camera euler angles should reflect the configured eye_angle_offset."""
    offset = 0.35  # 20° offset → 40° overlap
    env = _make_vision_env(offset)

    suffix = env._suffix  # "-rodent"
    left_cam = None
    right_cam = None
    for cam in env._spec.cameras:
        if cam.name == f"eye_left{suffix}":
            left_cam = cam
        elif cam.name == f"eye_right{suffix}":
            right_cam = cam

    assert left_cam is not None, "eye_left camera not found in spec"
    assert right_cam is not None, "eye_right camera not found in spec"

    expected_left_z = -np.pi / 2 + offset
    expected_right_z = -np.pi / 2 - offset

    np.testing.assert_allclose(left_cam.alt.euler[2], expected_left_z, atol=1e-6)
    np.testing.assert_allclose(right_cam.alt.euler[2], expected_right_z, atol=1e-6)


def test_default_offset_matches_original_xml():
    """Default offset=0.2 should reproduce the original XML camera angles."""
    env = _make_vision_env(0.2)
    suffix = env._suffix

    for cam in env._spec.cameras:
        if cam.name == f"eye_left{suffix}":
            np.testing.assert_allclose(
                cam.alt.euler[2], -np.pi / 2 + 0.2, atol=1e-6
            )
        elif cam.name == f"eye_right{suffix}":
            np.testing.assert_allclose(
                cam.alt.euler[2], -np.pi / 2 - 0.2, atol=1e-6
            )


def test_zero_offset_makes_eyes_look_straight_ahead():
    """offset=0 should make both eyes point the same direction as egocentric."""
    env = _make_vision_env(0.0)
    suffix = env._suffix

    ego_euler_z = None
    left_euler_z = None
    right_euler_z = None
    for cam in env._spec.cameras:
        if cam.name == f"egocentric{suffix}":
            ego_euler_z = cam.alt.euler[2]
        elif cam.name == f"eye_left{suffix}":
            left_euler_z = cam.alt.euler[2]
        elif cam.name == f"eye_right{suffix}":
            right_euler_z = cam.alt.euler[2]

    np.testing.assert_allclose(left_euler_z, ego_euler_z, atol=1e-6)
    np.testing.assert_allclose(right_euler_z, ego_euler_z, atol=1e-6)


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

    suffix = env._suffix
    for cam in env._spec.cameras:
        if cam.name == f"eye_left{suffix}":
            np.testing.assert_allclose(
                cam.alt.euler[2], -np.pi / 2 + 0.5, atol=1e-6
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
