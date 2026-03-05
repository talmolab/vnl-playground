"""Integration tests for ModularRodentEnv geometry using the real MuJoCo model.

These tests load the actual XML model and run MJX forward kinematics to verify:
1. torso_yawmat extracts the correct yaw-only rotation matrix.
2. body_relpos is invariant under global yaw rotations of the scene.
3. MJX FRAMEPOS sensors (reftype="xbody" refname="torso") match the expected
   formula: R_torso.T @ (body_pos - torso_pos).
4. MJX FRAMEXAXIS sensors match: R_torso.T @ R_body[:, 0].

These tests are marked 'integration' and excluded from the default pytest run.
Run explicitly with:
    pytest -m integration vnl_playground/tasks/modular_rodent/integration_test.py -v
"""

import brax.math
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx
from mujoco_playground._src.mjx_env import get_sensor_data

from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config


# ---------------------------------------------------------------------------
# Shared fixture — load env once per class
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_and_data():
    """Load ModularImitation and return (env, initial_data).

    Uses minimal solver iterations for speed. The fixture is module-scoped so
    the expensive JIT compilation and forward pass happen only once.
    """
    config = default_config()
    config.iterations = 3
    config.ls_iterations = 3
    env = ModularImitation(config)
    state = jax.jit(env.reset)(jax.random.key(0))
    return env, state.data


def _apply_world_yaw(env, data, delta_yaw: float):
    """Apply an additional world-frame yaw rotation to the torso freejoint.

    Multiplies the freejoint quaternion (qpos[3:7]) by Rz(delta_yaw) on the
    left, then runs MJX forward kinematics to update all derived quantities.
    """
    q_yaw = jp.array(
        [np.cos(delta_yaw / 2), 0.0, 0.0, np.sin(delta_yaw / 2)], dtype=jp.float32
    )
    q_cur = data.qpos[3:7]
    q_new = brax.math.quat_mul(q_yaw, q_cur)
    new_qpos = data.qpos.at[3:7].set(q_new)
    return jax.jit(mjx.forward)(env._mjx_model, data.replace(qpos=new_qpos))


# ---------------------------------------------------------------------------
# TestTorsoYawmatReal — validity with real model data
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTorsoYawmatReal:
    """torso_yawmat must return a valid yaw-only rotation matrix on real data."""

    def test_is_orthogonal(self, env_and_data):
        """R @ R.T == identity (orthogonality)."""
        env, data = env_and_data
        mat = np.array(env.torso_yawmat(data))
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-5)

    def test_determinant_is_one(self, env_and_data):
        """det(R) == 1 (proper rotation, not reflection)."""
        env, data = env_and_data
        mat = np.array(env.torso_yawmat(data))
        np.testing.assert_allclose(np.linalg.det(mat), 1.0, atol=1e-5)

    def test_z_axis_preserved(self, env_and_data):
        """Yaw-only matrix must not tilt the z-axis."""
        env, data = env_and_data
        mat = np.array(env.torso_yawmat(data))
        np.testing.assert_allclose(mat[2, :], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(mat[:, 2], [0.0, 0.0, 1.0], atol=1e-5)

    def test_extracted_yaw_matches_xmat(self, env_and_data):
        """The yaw extracted from xmat must equal the angle baked into the matrix."""
        env, data = env_and_data
        torso_id = env._mj_model.body("torso").id
        xmat = np.array(data.xmat[torso_id]).reshape(3, 3)
        expected_yaw = np.arctan2(xmat[1, 0], xmat[0, 0])
        mat = np.array(env.torso_yawmat(data))
        np.testing.assert_allclose(mat[0, 0], np.cos(expected_yaw), atol=1e-5)
        np.testing.assert_allclose(mat[0, 1], np.sin(expected_yaw), atol=1e-5)
        np.testing.assert_allclose(mat[1, 0], -np.sin(expected_yaw), atol=1e-5)
        np.testing.assert_allclose(mat[1, 1], np.cos(expected_yaw), atol=1e-5)

    def test_yaw_changes_after_rotation(self, env_and_data):
        """Applying Rz(90°) to the freejoint increases the extracted yaw by 90°."""
        env, data = env_and_data
        torso_id = env._mj_model.body("torso").id

        xmat0 = np.array(data.xmat[torso_id]).reshape(3, 3)
        yaw0 = np.arctan2(xmat0[1, 0], xmat0[0, 0])

        data_rot = _apply_world_yaw(env, data, np.pi / 2)
        xmat1 = np.array(data_rot.xmat[torso_id]).reshape(3, 3)
        yaw1 = np.arctan2(xmat1[1, 0], xmat1[0, 0])

        delta = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]
        np.testing.assert_allclose(delta, np.pi / 2, atol=1e-4)


# ---------------------------------------------------------------------------
# TestBodyRelposInvariance — global yaw must not change egocentric positions
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBodyRelposInvariance:
    """body_relpos should give identical results regardless of global torso yaw.

    Applying a world-frame yaw rotation to the freejoint changes world-frame body
    positions but must not change the yaw-aligned egocentric positions.
    """

    BODIES = ["hand_L", "hand_R", "foot_L", "foot_R"]
    DELTA_YAWS = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -np.pi / 2]

    def _check_invariant(self, env, data, delta_yaw):
        data_rot = _apply_world_yaw(env, data, delta_yaw)
        for body in self.BODIES:
            r0 = np.array(env.body_relpos(data, body))
            r1 = np.array(env.body_relpos(data_rot, body))
            np.testing.assert_allclose(
                r0, r1, atol=1e-4,
                err_msg=(
                    f"body_relpos({body}) changed after Rz({np.degrees(delta_yaw):.0f}°): "
                    f"before={r0}, after={r1}"
                ),
            )

    def test_invariant_45deg(self, env_and_data):
        env, data = env_and_data
        self._check_invariant(env, data, np.pi / 4)

    def test_invariant_90deg(self, env_and_data):
        env, data = env_and_data
        self._check_invariant(env, data, np.pi / 2)

    def test_invariant_180deg(self, env_and_data):
        env, data = env_and_data
        self._check_invariant(env, data, np.pi)

    def test_invariant_neg90deg(self, env_and_data):
        env, data = env_and_data
        self._check_invariant(env, data, -np.pi / 2)


# ---------------------------------------------------------------------------
# TestMjxSensorFrames — verify MJX computes sensor frames correctly
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMjxSensorFrames:
    """Verify MJX sensor values match the expected reference-frame formulas.

    FRAMEPOS with reftype="xbody": R_ref.T @ (obj_xpos - ref_xpos)
    FRAMEXAXIS with reftype="xbody": R_ref.T @ R_obj[:, 0]
    """

    def _get_torso_frame(self, env, data):
        torso_id = env._mj_model.body("torso").id
        R_torso = np.array(data.xmat[torso_id]).reshape(3, 3)
        torso_pos = np.array(data.xpos[torso_id])
        return R_torso, torso_pos

    def test_framepos_elbow_L(self, env_and_data):
        """arm_L/egocentric_elbow_pos matches R_torso.T @ (lower_arm_L - torso)."""
        env, data = env_and_data
        m = env._mj_model
        R_torso, torso_pos = self._get_torso_frame(env, data)

        lower_arm_id = m.body("lower_arm_L").id
        lower_arm_pos = np.array(data.xpos[lower_arm_id])
        expected = R_torso.T @ (lower_arm_pos - torso_pos)

        sensor_val = np.array(get_sensor_data(m, data, "arm_L/egocentric_elbow_pos"))
        np.testing.assert_allclose(sensor_val, expected, atol=1e-5)

    def test_framepos_elbow_R(self, env_and_data):
        """arm_R/egocentric_elbow_pos matches R_torso.T @ (lower_arm_R - torso)."""
        env, data = env_and_data
        m = env._mj_model
        R_torso, torso_pos = self._get_torso_frame(env, data)

        lower_arm_id = m.body("lower_arm_R").id
        lower_arm_pos = np.array(data.xpos[lower_arm_id])
        expected = R_torso.T @ (lower_arm_pos - torso_pos)

        sensor_val = np.array(get_sensor_data(m, data, "arm_R/egocentric_elbow_pos"))
        np.testing.assert_allclose(sensor_val, expected, atol=1e-5)

    def test_framexaxis_hand_L(self, env_and_data):
        """hand_L/xaxis matches R_torso.T @ R_hand_L[:, 0]."""
        env, data = env_and_data
        m = env._mj_model
        R_torso, _ = self._get_torso_frame(env, data)

        hand_L_id = m.body("hand_L").id
        R_hand = np.array(data.xmat[hand_L_id]).reshape(3, 3)
        expected = R_torso.T @ R_hand[:, 0]

        sensor_val = np.array(get_sensor_data(m, data, "hand_L/xaxis"))
        np.testing.assert_allclose(sensor_val, expected, atol=1e-5)

    def test_framexaxis_hand_R(self, env_and_data):
        """hand_R/xaxis matches R_torso.T @ R_hand_R[:, 0]."""
        env, data = env_and_data
        m = env._mj_model
        R_torso, _ = self._get_torso_frame(env, data)

        hand_R_id = m.body("hand_R").id
        R_hand = np.array(data.xmat[hand_R_id]).reshape(3, 3)
        expected = R_torso.T @ R_hand[:, 0]

        sensor_val = np.array(get_sensor_data(m, data, "hand_R/xaxis"))
        np.testing.assert_allclose(sensor_val, expected, atol=1e-5)

    def test_framexaxis_foot_L(self, env_and_data):
        """foot_L/xaxis matches R_torso.T @ R_foot_L[:, 0]."""
        env, data = env_and_data
        m = env._mj_model
        R_torso, _ = self._get_torso_frame(env, data)

        foot_L_id = m.body("foot_L").id
        R_foot = np.array(data.xmat[foot_L_id]).reshape(3, 3)
        expected = R_torso.T @ R_foot[:, 0]

        sensor_val = np.array(get_sensor_data(m, data, "foot_L/xaxis"))
        np.testing.assert_allclose(sensor_val, expected, atol=1e-5)

    def test_framexaxis_foot_R(self, env_and_data):
        """foot_R/xaxis matches R_torso.T @ R_foot_R[:, 0]."""
        env, data = env_and_data
        m = env._mj_model
        R_torso, _ = self._get_torso_frame(env, data)

        foot_R_id = m.body("foot_R").id
        R_foot = np.array(data.xmat[foot_R_id]).reshape(3, 3)
        expected = R_torso.T @ R_foot[:, 0]

        sensor_val = np.array(get_sensor_data(m, data, "foot_R/xaxis"))
        np.testing.assert_allclose(sensor_val, expected, atol=1e-5)

    def test_sensor_invariant_under_yaw(self, env_and_data):
        """Sensor values (framepos, framexaxis) are invariant under global yaw rotation.

        Since sensors are computed in the torso body frame, rotating the whole
        scene by a global yaw should not change any sensor value.
        """
        env, data = env_and_data
        m = env._mj_model
        data_rot = _apply_world_yaw(env, data, np.pi / 2)

        for sensor_name in [
            "arm_L/egocentric_elbow_pos",
            "arm_R/egocentric_elbow_pos",
            "hand_L/xaxis",
            "hand_R/xaxis",
            "foot_L/xaxis",
            "foot_R/xaxis",
        ]:
            v0 = np.array(get_sensor_data(m, data, sensor_name))
            v1 = np.array(get_sensor_data(m, data_rot, sensor_name))
            np.testing.assert_allclose(
                v0, v1, atol=1e-4,
                err_msg=f"Sensor {sensor_name} changed after Rz(90°): {v0} → {v1}",
            )
