# TODOs: Rewards definition for the escape bowl


from mujoco_playground._src import reward

import numpy as np


def _upright_reward(physics, walker, deviation_angle=0):
    """Returns a reward proportional to how upright the torso is.

    Args:
      physics: an instance of `Physics`.
      walker: the focal walker.
      deviation_angle: A float, in degrees. The reward is 0 when the torso is
        exactly upside-down and 1 when the torso's z-axis is less than
        `deviation_angle` away from the global z-axis.
    """
    deviation = np.cos(np.deg2rad(deviation_angle))
    upright_torso = physics.bind(walker.root_body).xmat[-1]
    if hasattr(walker, "pelvis_body"):
        upright_pelvis = physics.bind(walker.pelvis_body).xmat[-1]
        upright_zz = np.stack([upright_torso, upright_pelvis])
    else:
        upright_zz = upright_torso
    upright = reward.tolerance(
        upright_zz,
        bounds=(deviation, float("inf")),
        sigmoid="linear",
        margin=1 + deviation,
        value_at_margin=0,
    )
    return np.min(upright)
