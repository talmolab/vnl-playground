"""Geometry tests for ModularRodentEnv coordinate frame helpers and _sub_quat.

Tests verify:
- torso_yawmat returns Rz(-θ) (world-to-egocentric)
- body_relpos correctly transforms world-frame positions to egocentric
- _sub_quat matches Julia's subQuat: conj(current) * target → 3D angular velocity

All tests use lightweight mock objects — no MJX model load required.
"""

import types

import jax.numpy as jp
import numpy as np
import pytest

from vnl_playground.tasks.modular_rodent.base import ModularRodentEnv
from vnl_playground.tasks.modular_rodent.imitation import _sub_quat


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _ConcreteEnv(ModularRodentEnv):
    """Minimal concrete subclass — only coordinate frame helpers are exercised."""

    def reset(self, rng):
        raise NotImplementedError

    def step(self, state, action):
        raise NotImplementedError


def _make_env(xmat_rows, xpos_rows):
    """Create a _ConcreteEnv with mock model/data, bypassing __init__.

    Args:
        xmat_rows: list/array of shape (n_bodies, 9) — flat row-major rotation
                   matrices (MuJoCo xmat convention).
        xpos_rows: list/array of shape (n_bodies, 3) — body positions.

    Body ids: torso=0, test body=1.
    """
    env = object.__new__(_ConcreteEnv)  # skip __init__
    env._mj_model = types.SimpleNamespace(
        body=lambda name: types.SimpleNamespace(id=0 if name == "torso" else 1)
    )
    data = types.SimpleNamespace(
        xmat=jp.array(xmat_rows, dtype=jp.float32),
        xpos=jp.array(xpos_rows, dtype=jp.float32),
    )
    return env, data


def _rz(theta):
    """Rz(theta) as flat row-major (9,) array (MuJoCo xmat convention).

    Rz(theta) rotates by theta around the z-axis (body-to-world rotation
    for a body facing direction theta in the horizontal plane).
    """
    c, s = np.cos(theta), np.sin(theta)
    # Row-major flattening of [[c,-s,0],[s,c,0],[0,0,1]]
    return np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _rz_neg(theta):
    """Rz(-theta) as a 3x3 numpy array (expected world-to-egocentric matrix)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rz_ry(theta, phi):
    """Rz(theta) @ Ry(phi) as flat row-major (9,) array."""
    cz, sz = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(phi), np.sin(phi)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return (Rz @ Ry).flatten().astype(np.float32)


def _rz_rx(theta, phi):
    """Rz(theta) @ Rx(phi) as flat row-major (9,) array."""
    cz, sz = np.cos(theta), np.sin(theta)
    cx, sx = np.cos(phi), np.sin(phi)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return (Rz @ Rx).flatten().astype(np.float32)


_DUMMY_XMAT = np.zeros(9, dtype=np.float32)
_DUMMY_XPOS = np.zeros(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# TestTorsoYawmat: verify torso_yawmat returns Rz(-theta)
# ---------------------------------------------------------------------------


class TestTorsoYawmat:
    """torso_yawmat should return a world-to-egocentric rotation matrix Rz(-theta)."""

    def _check(self, xmat_flat, expected_3x3):
        env, data = _make_env(
            [xmat_flat, _DUMMY_XMAT], [_DUMMY_XPOS, _DUMMY_XPOS]
        )
        result = np.array(env.torso_yawmat(data))
        np.testing.assert_allclose(result, expected_3x3, atol=1e-5)

    def test_yaw_zero(self):
        """At yaw=0 the matrix should be identity."""
        self._check(_rz(0.0), np.eye(3, dtype=np.float32))

    def test_yaw_90(self):
        """At yaw=90° the result should be Rz(-90°) = [[0,1,0],[-1,0,0],[0,0,1]]."""
        self._check(_rz(np.pi / 2), _rz_neg(np.pi / 2))

    def test_yaw_minus90(self):
        """At yaw=-90° the result should be Rz(+90°) = [[0,-1,0],[1,0,0],[0,0,1]]."""
        self._check(_rz(-np.pi / 2), _rz_neg(-np.pi / 2))

    def test_yaw_45(self):
        """At yaw=45° the result should be Rz(-45°)."""
        self._check(_rz(np.pi / 4), _rz_neg(np.pi / 4))

    def test_yaw_180(self):
        """At yaw=180° the result should be Rz(-180°) = [[-1,0,0],[0,-1,0],[0,0,1]]."""
        self._check(_rz(np.pi), _rz_neg(np.pi))

    def test_yaw_with_pitch(self):
        """Yaw extraction is robust to pitch (rotation around y-axis)."""
        # Rz(90°) @ Ry(30°) — body at 90° yaw with 30° pitch
        # The x-axis of this body in world is still mostly pointing in +y direction
        # so yaw should still be extracted as ~90°
        self._check(_rz_ry(np.pi / 2, np.pi / 6), _rz_neg(np.pi / 2))

    def test_yaw_with_roll(self):
        """Yaw extraction is robust to roll (rotation around x-axis)."""
        # Rz(90°) @ Rx(30°)
        # The x-axis of this body is unchanged by Rx, so yaw is still 90°
        self._check(_rz_rx(np.pi / 2, np.pi / 6), _rz_neg(np.pi / 2))


# ---------------------------------------------------------------------------
# TestBodyRelpos: verify body_relpos gives correct egocentric positions
# ---------------------------------------------------------------------------


class TestBodyRelpos:
    """body_relpos should give the body position in the yaw-aligned egocentric frame.

    Convention (matching Julia):
    - Egocentric +x = forward (direction torso faces)
    - Egocentric +y = left
    - z is absolute height (torso z not subtracted)
    """

    def _check(self, torso_xmat, torso_xpos, body_xpos, expected_relpos):
        env, data = _make_env(
            [torso_xmat, _DUMMY_XMAT],
            [torso_xpos, body_xpos],
        )
        # Override data.xpos[1] to be the body position
        data = types.SimpleNamespace(
            xmat=jp.array([torso_xmat, _DUMMY_XMAT], dtype=jp.float32),
            xpos=jp.array([torso_xpos, body_xpos], dtype=jp.float32),
        )
        result = np.array(env.body_relpos(data, "body"))  # body_id=1
        np.testing.assert_allclose(result, expected_relpos, atol=1e-5)

    def test_in_front_yaw0(self):
        """At yaw=0, body 1m ahead in world → [1,0,0] egocentric."""
        self._check(_rz(0.0), [0, 0, 0], [1, 0, 0], [1, 0, 0])

    def test_in_front_yaw90(self):
        """At yaw=90°, body 1m ahead in world (=[0,1,0]) → [1,0,0] egocentric."""
        self._check(_rz(np.pi / 2), [0, 0, 0], [0, 1, 0], [1, 0, 0])

    def test_in_front_yaw45(self):
        """At yaw=45°, body 1m ahead in world (=[cos45,sin45,0]) → [1,0,0] egocentric."""
        c45 = np.cos(np.pi / 4)
        self._check(_rz(np.pi / 4), [0, 0, 0], [c45, c45, 0], [1, 0, 0])

    def test_right_yaw0(self):
        """At yaw=0, body 1m to the right in world (=[0,-1,0]) → [0,-1,0] egocentric."""
        self._check(_rz(0.0), [0, 0, 0], [0, -1, 0], [0, -1, 0])

    def test_right_yaw90(self):
        """At yaw=90°, body to the right in world (=[1,0,0]) → [0,-1,0] egocentric."""
        self._check(_rz(np.pi / 2), [0, 0, 0], [1, 0, 0], [0, -1, 0])

    def test_left_yaw90(self):
        """At yaw=90°, body to the left in world (=[-1,0,0]) → [0,1,0] egocentric."""
        self._check(_rz(np.pi / 2), [0, 0, 0], [-1, 0, 0], [0, 1, 0])

    def test_height_preserved(self):
        """Z-component (absolute height) is preserved regardless of torso height."""
        # torso at [5,3,0.1], body at [6,3,0.5], yaw=0
        # torso_xy = [5,3,0], body-torso_xy = [1,0,0.5], Rz(0)@[1,0,0.5] = [1,0,0.5]
        self._check(_rz(0.0), [5, 3, 0.1], [6, 3, 0.5], [1, 0, 0.5])

    def test_height_yaw90(self):
        """Height is preserved even at non-zero yaw."""
        # torso at origin z=0.1, body at [0,1,0.3], yaw=90°
        # body-torso_xy = [0,1,0.3], Rz(-90°)@[0,1,0.3] = [1,0,0.3]
        self._check(_rz(np.pi / 2), [0, 0, 0.1], [0, 1, 0.3], [1, 0, 0.3])


# ---------------------------------------------------------------------------
# TestRotationInvariance: relpos is independent of absolute torso yaw
# ---------------------------------------------------------------------------


class TestRotationInvariance:
    """For a fixed egocentric offset, varying the torso yaw (and rotating the body
    accordingly) should always give the same body_relpos result.

    This is the key invariance property: the agent's egocentric observations
    should not change if the whole scene is globally rotated.
    """

    YAWS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -np.pi / 2]

    def _world_pos_for_ego(self, ego_offset, yaw):
        """Compute world position of body given egocentric offset and torso yaw."""
        # ego = Rz(-yaw) @ (world - torso), so world = torso + Rz(+yaw) @ ego
        Rz_pos = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0],
             [np.sin(yaw),  np.cos(yaw), 0],
             [0,            0,           1]]
        )
        return Rz_pos @ np.array(ego_offset)

    def _check_invariant(self, ego_offset):
        results = []
        for yaw in self.YAWS:
            body_world = self._world_pos_for_ego(ego_offset, yaw)
            env, data = _make_env(
                [_rz(yaw), _DUMMY_XMAT],
                [[0, 0, 0], body_world],
            )
            results.append(np.array(env.body_relpos(data, "body")))

        # All results should match the expected ego offset
        for i, result in enumerate(results):
            np.testing.assert_allclose(
                result,
                np.array(ego_offset, dtype=np.float32),
                atol=1e-5,
                err_msg=f"Invariance failed at yaw={np.degrees(self.YAWS[i]):.1f}°",
            )

    def test_invariant_forward(self):
        """Body directly in front: egocentric [1,0,0] at all yaw angles."""
        self._check_invariant([1.0, 0.0, 0.0])

    def test_invariant_left(self):
        """Body to the left: egocentric [0,1,0] at all yaw angles."""
        self._check_invariant([0.0, 1.0, 0.0])

    def test_invariant_diagonal(self):
        """Body diagonally ahead-left: egocentric [1,1,0] at all yaw angles."""
        self._check_invariant([1.0, 1.0, 0.0])

    def test_invariant_behind_right(self):
        """Body behind and to the right: egocentric [-1,-1,0] at all yaw angles."""
        self._check_invariant([-1.0, -1.0, 0.0])


# ---------------------------------------------------------------------------
# TestSubQuat: verify _sub_quat matches Julia's subQuat
# ---------------------------------------------------------------------------


def _quat_rz(theta):
    """Unit quaternion for Rz(theta): [cos(θ/2), 0, 0, sin(θ/2)] in [w,x,y,z]."""
    return jp.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])


def _quat_rx(theta):
    """Unit quaternion for Rx(theta): [cos(θ/2), sin(θ/2), 0, 0]."""
    return jp.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])


def _quat_ry(theta):
    """Unit quaternion for Ry(theta): [cos(θ/2), 0, sin(θ/2), 0]."""
    return jp.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])


_IDENTITY_QUAT = jp.array([1.0, 0.0, 0.0, 0.0])


class TestSubQuat:
    """_sub_quat should match Julia's subQuat: conj(q_current) * q_target → 3D vel.

    Convention: quaternions are [w, x, y, z] (MuJoCo / brax).
    Output is axis * angle with dt=1 (same as MuJoCo mju_subQuat).
    """

    def test_identity_same_rotation(self):
        """Relative rotation from q to itself is zero angular velocity."""
        q = _quat_rz(np.pi / 3)
        result = np.array(_sub_quat(q, q))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-5)

    def test_identity_current_rz90(self):
        """From identity to Rz(90°): angular velocity [0, 0, π/2]."""
        result = np.array(_sub_quat(_quat_rz(np.pi / 2), _IDENTITY_QUAT))
        np.testing.assert_allclose(result, [0.0, 0.0, np.pi / 2], atol=1e-5)

    def test_identity_current_rz_neg90(self):
        """From identity to Rz(-90°): angular velocity [0, 0, -π/2]."""
        result = np.array(_sub_quat(_quat_rz(-np.pi / 2), _IDENTITY_QUAT))
        np.testing.assert_allclose(result, [0.0, 0.0, -np.pi / 2], atol=1e-5)

    def test_identity_current_rx90(self):
        """From identity to Rx(90°): angular velocity [π/2, 0, 0]."""
        result = np.array(_sub_quat(_quat_rx(np.pi / 2), _IDENTITY_QUAT))
        np.testing.assert_allclose(result, [np.pi / 2, 0.0, 0.0], atol=1e-5)

    def test_identity_current_ry45(self):
        """From identity to Ry(45°): angular velocity [0, π/4, 0]."""
        result = np.array(_sub_quat(_quat_ry(np.pi / 4), _IDENTITY_QUAT))
        np.testing.assert_allclose(result, [0.0, np.pi / 4, 0.0], atol=1e-5)

    def test_nonidentity_current(self):
        """From Rz(30°) to Rz(90°): relative rotation is Rz(60°) → [0,0,π/3]."""
        result = np.array(_sub_quat(_quat_rz(np.pi / 2), _quat_rz(np.pi / 6)))
        np.testing.assert_allclose(result, [0.0, 0.0, np.pi / 3], atol=1e-5)

    def test_order_matters(self):
        """Swapping arguments gives the opposite sign (anti-commutativity)."""
        q1 = _quat_rz(np.pi / 4)
        q2 = _quat_rz(np.pi / 2)
        fwd = np.array(_sub_quat(q2, q1))  # from q1 to q2: +π/4
        rev = np.array(_sub_quat(q1, q2))  # from q2 to q1: -π/4
        np.testing.assert_allclose(fwd, -rev, atol=1e-5)

    def test_large_angle_near_pi(self):
        """Rotation close to π: angle should be near ±π, not wrap to near 0."""
        # Rz(170°): angle ≈ 0.97π
        q_target = _quat_rz(np.radians(170))
        result = np.array(_sub_quat(q_target, _IDENTITY_QUAT))
        angle = np.linalg.norm(result)
        np.testing.assert_allclose(angle, np.radians(170), atol=1e-4)

    def test_output_shape(self):
        """Output must be a 3-vector, not 4-vector (unlike relative_quat)."""
        result = np.array(_sub_quat(_quat_rz(1.0), _IDENTITY_QUAT))
        assert result.shape == (3,)
