"""Shared numerical helpers for VNL tasks.

The functions in this module are stateless and JAX-compatible.  Task-specific
target selection, weighting configuration, and metric names remain in the
individual environments.
"""

import brax.math
import jax.numpy as jp


def quaternion_angle(q1, q2, *, degrees: bool = False):
    """Returns the shortest rotation angle between unit quaternions.

    Quaternion components are reduced over the final axis, so leading batch
    dimensions are preserved.  Squaring the dot product makes the result
    invariant to the equivalent ``q`` and ``-q`` representations.
    """
    dot = jp.sum(q1 * q2, axis=-1)
    cosine = 2.0 * jp.square(dot) - 1.0
    angle = jp.arccos(jp.clip(cosine, -1.0, 1.0))
    return jp.rad2deg(angle) if degrees else angle


def gaussian_reward(error, *, weight, scale):
    """Returns ``weight * exp(-0.5 * (error / scale) ** 2)``."""
    return weight * jp.exp(-0.5 * jp.square(error / scale))


def squared_l2_norm(value, *, axis=-1):
    """Returns the squared L2 norm along an explicitly selected axis."""
    return jp.sum(jp.square(value), axis=axis)


def absolute_actuator_power(qvel, actuator_force, *, axis=-1):
    """Returns summed absolute actuator power along the selected axis."""
    return jp.sum(jp.abs(qvel * actuator_force), axis=axis)


def wrap_angle_to_pi(angle):
    """Wraps angles in radians to the interval ``[-pi, pi]``."""
    return jp.arctan2(jp.sin(angle), jp.cos(angle))


def world_vector_to_local(vector_world, frame_orientation_world):
    """Rotates a world-frame vector into a quaternion-defined local frame."""
    return brax.math.inv_rotate(vector_world, frame_orientation_world)


def world_point_to_local(point_world, frame_position_world, frame_orientation_world):
    """Transforms a world-frame point into a quaternion-defined local frame."""
    return world_vector_to_local(
        point_world - frame_position_world, frame_orientation_world
    )
