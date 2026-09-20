"""Eye-camera yaw offset: assert on the cameras the env actually renders from.

``RunGapVision.__init__`` only builds the servo-held yaw mount when the
configured offset differs from 0.2 (see ``run_gap_vision.py:155-162``)::

    offset = self._config.get("eye_angle_offset", 0.2)
    if abs(offset - 0.2) > 1e-6:
        self._configure_eye_cameras(offset)

So the two regimes are the opposite of what "0.2 is the default eye offset"
might suggest at a glance:

- ``offset == 0.2`` (the ``default_config()`` value): the gate is False. No
  mount body, hinge joint, or actuator is built; the env keeps rendering
  from the untouched XML cameras ``eye_left-rodent`` / ``eye_right-rodent``,
  and ``action_size`` is unaffected.
- ``offset != 0.2``, INCLUDING ``0.0``: the gate is True.
  ``_configure_eye_cameras`` builds an ``eye_{side}_fixed_mount`` body, a
  hinge joint ``eye_{side}_yaw_fixed`` held at the offset by a position
  servo, and a camera ``eye_{side}_yawed``. ``self._config.{left,right}
  _camera_name`` are updated to ``eye_{side}_yawed-rodent`` and the model is
  recompiled with 2 extra actuators. ``self._n_fixed_eye_actuators = 2`` is
  then subtracted back out of ``action_size`` (``run_gap_vision.py:266-269``)
  so the policy's action dimensionality does not change -- the servo holds
  the offset passively, the policy never drives it.

``cam_mat0`` (the qpos0 pose) never shows the yaw; the tests below park the
joint at its equilibrium (== the configured offset, since the servo's bias
term is tuned so ``ctrl=0`` holds ``qpos=offset``) and read ``cam_xmat``.
"""

import mujoco
import numpy as np
import pytest

from vnl_playground.tasks.rodent import run_gap_vision


def _env(offset: float) -> run_gap_vision.RunGapVision:
    cfg = run_gap_vision.default_config()
    cfg.binocular = True
    cfg.eye_angle_offset = offset
    cfg.mujoco_impl = "warp"
    return run_gap_vision.RunGapVision(config=cfg)


def _look_at_equilibrium(
    env: run_gap_vision.RunGapVision, offset: float, cam_name: str
) -> np.ndarray:
    """World-frame look vector of ``cam_name`` with any yaw servo joint
    parked at its equilibrium (== offset)."""
    m = env.mj_model
    suffix = env._suffix
    d = mujoco.MjData(m)
    for side in ("left", "right"):
        jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"eye_{side}_yaw_fixed{suffix}"
        )
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = offset
    mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    assert cid >= 0, f"camera {cam_name!r} not in model"
    return d.cam_xmat[cid].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])


def test_default_config_has_eye_angle_offset():
    """``default_config().eye_angle_offset`` is 0.2 -- the value at which the
    servo-mount gate is False (``run_gap_vision.py:90``)."""
    assert run_gap_vision.default_config().eye_angle_offset == 0.2


@pytest.mark.gpu
def test_default_offset_uses_xml_cameras_no_servo_mount():
    """``offset == 0.2`` (default): the gate is False, so no mount body,
    hinge joint, or actuator is built, and the env keeps the XML
    ``eye_left-rodent`` / ``eye_right-rodent`` cameras."""
    env = _env(0.2)
    suffix = env._suffix
    assert env._config.left_camera_name == f"eye_left{suffix}"
    assert env._config.right_camera_name == f"eye_right{suffix}"

    m = env.mj_model
    for side in ("left", "right"):
        assert (
            mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, f"eye_{side}_yaw_fixed{suffix}"
            )
            == -1
        )
        assert (
            mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"eye_{side}_yaw_fixed{suffix}"
            )
            == -1
        )
        assert (
            mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_CAMERA, f"eye_{side}_yawed{suffix}"
            )
            == -1
        )
    assert getattr(env, "_n_fixed_eye_actuators", 0) == 0


@pytest.mark.gpu
def test_boundary_flips_offset_gate():
    """``abs(offset - 0.2) > 1e-6`` is the gate: 0.2 stays on the XML
    cameras, but 0.2 + 2e-6 already crosses into the servo-mount regime."""
    at_default = _env(0.2)
    just_past = _env(0.2 + 2e-6)
    suffix = at_default._suffix

    assert at_default._config.left_camera_name == f"eye_left{suffix}"
    assert just_past._config.left_camera_name == f"eye_left_yawed{suffix}"


@pytest.mark.gpu
def test_zero_offset_eyes_look_straight_ahead():
    """``offset == 0.0`` crosses the gate (0.0 != 0.2) and gives a sharper
    assertion than a generic nonzero offset: both eyes are held at the same
    yaw (0), so they must look in exactly the same direction -- maximum
    overlap, per the ``_configure_eye_cameras`` docstring."""
    env = _env(0.0)
    suffix = env._suffix
    assert env._config.left_camera_name == f"eye_left_yawed{suffix}"
    assert env._config.right_camera_name == f"eye_right_yawed{suffix}"

    left = _look_at_equilibrium(env, 0.0, env._config.left_camera_name)
    right = _look_at_equilibrium(env, 0.0, env._config.right_camera_name)
    np.testing.assert_allclose(left, right, atol=1e-4)
    np.testing.assert_allclose(left, [1.0, 0.0, 0.0], atol=1e-4)


@pytest.mark.gpu
def test_nonzero_offset_look_vectors_mirror():
    """``offset == 0.35`` (comfortably inside ``(0, pi/2)``): the two yawed
    cameras are mirror images about the forward axis, matching the mount
    joint axis convention (``left axis=[0,0,1]``, ``right axis=[0,0,-1]``)."""
    offset = 0.35
    env = _env(offset)
    suffix = env._suffix
    assert env._config.left_camera_name == f"eye_left_yawed{suffix}"
    assert env._config.right_camera_name == f"eye_right_yawed{suffix}"

    left = _look_at_equilibrium(env, offset, env._config.left_camera_name)
    right = _look_at_equilibrium(env, offset, env._config.right_camera_name)
    np.testing.assert_allclose(left, [np.cos(offset), np.sin(offset), 0.0], atol=1e-4)
    np.testing.assert_allclose(right, [np.cos(offset), -np.sin(offset), 0.0], atol=1e-4)


@pytest.mark.gpu
def test_action_size_invariant_across_offset_regimes():
    """The servo actuators the ``offset != 0.2`` branch adds are not meant
    to be policy-controlled (``run_gap_vision.py:171-173``):
    ``_n_fixed_eye_actuators`` is subtracted back out of ``action_size``
    (line 269), so raw actuator count ``nu`` changes but ``action_size``
    does not. This is the opposite of the ``actuable_eyes=True`` path
    (see ``test_actuable_eyes.py::test_action_size_increased_by_two``),
    where the extra actuators ARE policy-controlled and DO grow
    ``action_size``.
    """
    baseline = _env(0.2)
    yawed = _env(0.35)

    assert yawed.mj_model.nu == baseline.mj_model.nu + 2
    assert yawed.action_size == baseline.action_size
