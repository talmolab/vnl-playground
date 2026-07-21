"""Maintain velocity task for virtual C. elegans.

The worm is initialized in a neutral standing pose facing forward (+x axis)
and must maintain a target forward velocity. No upright reward is used.

Termination occurs if:
- Torso becomes too tilted (fallen)
- Torso goes below ground level
- NaN detected in simulation data
"""

import collections
from typing import Any, Callable, Dict, Mapping, Optional, Union

import mujoco
import jax
import jax.numpy as jp
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_fns

from vnl_playground.tasks.celegans import base as celegans_base
from vnl_playground.tasks.celegans import consts
from vnl_playground.tasks.reward_registry import RewardRegistry

_registry = RewardRegistry()


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the MaintainVelocity environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        init_z=0.0,
        target_speed=0.01,  # cm/s
        episode_length=2000,
        action_repeat=1,
        reward_terms={
            "forward_velocity": {"weight": 1.0},
            "lateral_velocity": {"weight": 0.0},
            "angular_velocity_z": {"weight": 0.0},
        },
        termination_criteria={
            "fallen": {"min_torso_z": -1, "max_torso_angle": 60},
            "nan_termination": {},
        },
        **celegans_base.default_config(),
    )


class MaintainVelocity(celegans_base.CelegansEnv):
    """Maintain velocity environment.

    The worm must maintain a target forward velocity in the +x direction.
    Initialized in neutral standing pose facing forward.
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the MaintainVelocity environment.

        Args:
            rng: Random number generator key.
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        # ConfigDict annoyingly sorts dictionary by keys
        friction = [
            self.config.friction["tan_floor"],
            self.config.friction["tan_body"],
            self.config.friction["tor"],
            self.config.friction["roll_floor"],
            self.config.friction["roll_body"],
        ]
        solimp = [
            self.config.solimp["d0"],
            self.config.solimp["dwidth"],
            self.config.solimp["width"],
            self.config.solimp["midpoint"],
            self.config.solimp["power"],
        ]
        solref = [
            self.config.solref["timeconst"],
            self.config.solref["dampratio"],
        ]
        solreffriction = [
            self.config.solreffriction["timeconst"],
            self.config.solreffriction["dampratio"],
        ]

        if self.config.contact_geom.lower() == "mesh":
            contact_geom = mujoco.mjtGeom.mjGEOM_MESH
        elif self.config.contact_geom.lower() == "capsule":
            contact_geom = mujoco.mjtGeom.mjGEOM_CAPSULE
        else:
            contact_geom = mujoco.mjtGeom.mjGEOM_SPHERE

        # Initialize worm at origin facing forward (+x direction)
        # quat (1, 0, 0, 0) = identity = facing +x by default
        init_x, init_y, init_z = 0.0, 0.0, self._config.init_z
        init_quat = (1, 0, 0, 0)

        self.add_worm(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
            friction=friction,
            solimp=solimp,
            solref=solref,
            solreffriction=solreffriction,
            contact_geom=contact_geom,
            muscle_config=self._config.muscle_config,
            joint_config=self._config.joint_config,
        )

        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment state.

        Args:
            rng: Random number generator state.

        Returns:
            mjx_env.State: The initial environment state after reset.
        """
        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward by one timestep.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            mjx_env.State: The new environment state after stepping.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)

        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> collections.OrderedDict:
        """Get the current observation from the simulation data.

        Args:
            data: The simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with state and privileged_state keys.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        # touch_sensors = self._get_touch_sensors(data) TODO: Add touch sensors back once worm has them
        origin = self._get_origin(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                # touch_sensors
                origin,
            ]
        )

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(state=obs)

    @_registry.reward("forward_velocity")
    def _forward_velocity_reward(self, data, info, metrics, weight) -> float:
        """Reward for maintaining target forward velocity in +x direction.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted forward velocity reward.
        """
        del info

        forward_vel = self.subtree_linvel(data)[0]

        target_speed = self._config.target_speed

        reward_value = reward_fns.tolerance(
            forward_vel,
            bounds=(target_speed, target_speed),
            margin=abs(target_speed),
            sigmoid="linear",
            value_at_margin=0.0,
        )

        weighted_reward = reward_value * weight
        metrics["rewards/forward_velocity"] = metrics[
            "rewards/forward_velocity/per_step"
        ] = weighted_reward
        metrics["magnitudes/forward_velocity"] = metrics[
            "magnitudes/forward_velocity/per_step"
        ] = forward_vel

        return weighted_reward

    @_registry.reward("lateral_velocity")
    def _lateral_velocity_cost(self, data, info, metrics, weight) -> float:
        """Cost for lateral (y-direction) velocity to encourage straight-line motion.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (positive value).

        Returns:
            Negative weighted cost (penalty for lateral movement).
        """
        del info
        lateral_vel = self.subtree_linvel(data)[1]  # y-direction velocity
        cost = -weight * jp.square(lateral_vel)
        metrics["costs/lateral_velocity"] = metrics[
            "costs/lateral_velocity/per_step"
        ] = cost
        metrics["magnitudes/lateral_velocity"] = metrics[
            "magnitudes/lateral_velocity/per_step"
        ] = lateral_vel
        return cost

    @_registry.reward("angular_velocity_z")
    def _angular_velocity_z_cost(self, data, info, metrics, weight) -> float:
        """Cost for yaw rate (z-axis angular velocity) to prevent turning.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (positive value).

        Returns:
            Negative weighted cost (penalty for turning).
        """
        del info
        # Use gyro sensor for angular velocity

        gyro = self.gyro(data)

        yaw_rate = gyro[2]  # z-axis angular velocity
        cost = -weight * jp.square(yaw_rate)
        metrics["costs/angular_velocity_z"] = metrics[
            "costs/angular_velocity_z/per_step"
        ] = cost
        metrics["magnitudes/angular_velocity_z"] = metrics[
            "magnitudes/angular_velocity_z/per_step"
        ] = yaw_rate
        return cost

    @_registry.termination("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = 0.03,
        max_torso_angle: float = 60,
    ) -> bool:
        """Check if worm has fallen.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum z height threshold.
            max_torso_angle: Maximum angle from vertical in degrees.

        Returns:
            Boolean indicating if fallen.
        """
        del info

        torso_body = self.root_body(data)
        torso_z = torso_body.xpos[2]

        below_ground = torso_z < min_torso_z

        # xmat is 3x3 rotation matrix, [-1, -1] is element (2,2) = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_ground, too_tilted)

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Check for NaN values in simulation data.

        Args:
            data: Simulation data.
            info: State info (unused).

        Returns:
            Boolean indicating if NaN detected in qpos.
        """
        return jp.any(jp.isnan(data.qpos))

    def null_action(self) -> jp.ndarray:
        """Return zero action."""
        return jp.zeros(self.action_size)
