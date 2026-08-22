"""Maintain velocity task for virtual rodent.

The rodent is initialized in a neutral standing pose facing forward (+x axis)
and must maintain a target forward velocity. No upright reward is used.

Termination occurs if:
- Torso becomes too tilted (fallen)
- Torso goes below ground level
- NaN detected in simulation data
"""

import collections
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_fns

from vnl_playground.tasks.reward_registry import RewardRegistry
from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts

_registry = RewardRegistry()


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the MaintainVelocity environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="newton",
        mujoco_impl="jax",
        contacts_per_world=80,
        constraints_per_world=320,
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        init_z=0.0,
        target_speed=0.5,
        episode_length=2000,
        action_repeat=1,
        reward_terms={
            "forward_velocity": {"weight": 1.0},
            "lateral_velocity": {"weight": 0.0},
            "angular_velocity_z": {"weight": 0.0},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.03, "max_torso_angle": 60},
            "nan_termination": {},
        },
    )


class MaintainVelocity(rodent_base.RodentEnv):
    """Maintain velocity environment.

    The rodent must maintain a target forward velocity in the +x direction.
    Initialized in neutral standing pose facing forward.
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict[str, str | int | list[Any]] | None = None,
        num_worlds: int = 1,
    ) -> None:
        """Initialize the MaintainVelocity environment.

        Args:
            rng: Random number generator key.
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides, num_worlds)
        self._rng = rng

        # Initialize rodent at origin facing forward (+x direction)
        # quat (1, 0, 0, 0) = identity = facing +x by default
        init_x, init_y, init_z = 0.0, 0.0, self._config.init_z
        init_quat = (1, 0, 0, 0)

        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
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
            naconmax=self._config.contacts_per_world * self._num_worlds,
            njmax=self._config.constraints_per_world,
        )
        data = mjx.forward(self.mjx_model, data)

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
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
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

        body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        forward_vel = body.subtree_linvel[0]

        target_speed = self._config.target_speed

        reward_value = reward_fns.tolerance(
            forward_vel,
            bounds=(target_speed, target_speed),
            margin=target_speed,
            sigmoid="linear",
            value_at_margin=0.0,
        )

        weighted_reward = reward_value * weight
        metrics["rewards/forward_velocity"] = weighted_reward

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
        body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        lateral_vel = body.subtree_linvel[1]  # y-direction velocity
        cost = -weight * jp.square(lateral_vel)  # negative to penalize
        metrics["rewards/lateral_velocity"] = cost
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
        gyro = data.bind(
            self.mjx_model, self._spec.sensor(f"gyro{self._suffix}")
        ).sensordata
        yaw_rate = gyro[2]  # z-axis angular velocity
        cost = -weight * jp.square(yaw_rate)  # negative to penalize
        metrics["rewards/angular_velocity_z"] = cost
        return cost

    @_registry.termination("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = 0.03,
        max_torso_angle: float = 60,
    ) -> bool:
        """Check if rodent has fallen.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum z height threshold.
            max_torso_angle: Maximum angle from vertical in degrees.

        Returns:
            Boolean indicating if fallen.
        """
        del info

        torso_body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
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
