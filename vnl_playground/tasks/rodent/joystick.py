"""Joystick velocity tracking task for virtual rodent.

The rodent tracks joystick-style velocity commands: forward velocity and
angular velocity (yaw). Commands are periodically resampled during the episode.

Only forward movement and turning are supported since strafing and backwards
movement don't work well with the rodent morphology.

Modeled after locomotion joystick tasks (e.g., G1 humanoid) but simplified for
rodent-scale velocities and reward structure.
"""

import collections
from typing import Any, Callable, Dict, Mapping, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.task_registry import TaskRegistry

_registry = TaskRegistry()


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the Joystick environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        # Standard sim params (same as maintain_velocity)
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.02,
        sim_dt=0.002,
        solver="newton",
        mujoco_impl="jax",
        naconmax=90 * 1024,
        njmax=1200,
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        # Episode params
        episode_length=1000,
        action_repeat=1,
        # Command ranges (rodent-scale velocities)
        # Forward-only (no backwards) + turning
        lin_vel_x=(0.0, 0.5),  # m/s, forward only
        ang_vel_yaw=(-1.0, 1.0),  # rad/s
        # Command sampling
        command_config=config_dict.create(
            resample_interval=500,  # steps between resamples
            zero_prob=0.1,  # probability of zero command
        ),
        # Reward config
        reward_config=config_dict.create(
            tracking_sigma=0.25,  # exponential reward sharpness
        ),
        # Reward terms
        reward_terms={
            # Tracking rewards
            "tracking_lin_vel": {"weight": 1.0},
            "tracking_ang_vel": {"weight": 1.5},  # Higher weight - turning is harder
            # Energy rewards (lowered to allow more aggressive turning)
            "torques": {"weight": -0.00002},
            "action_rate": {"weight": -0.002},
            "energy": {"weight": -0.0002},
            "dof_acc": {"weight": -2e-8, "max_value": 1e5},
            # Other rewards
            "alive": {"weight": 0.1},
            "termination": {"weight": -100.0},
            "stand_still": {"weight": -0.5},
            "lateral_velocity": {"weight": -0.1},  # Penalize sideways drift
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.03, "max_torso_angle": 60},
            "nan_termination": {},
        },
    )


class Joystick(rodent_base.RodentEnv):
    """Joystick velocity tracking environment.

    The rodent must track commanded linear and angular velocities.
    Commands are periodically resampled during the episode.
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the Joystick environment.

        Args:
            rng: Random number generator key.
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        # Initialize rodent at origin facing forward (+x direction)
        init_x, init_y, init_z = 0.0, 0.0, 0.03
        init_quat = (1, 0, 0, 0)  # identity = facing +x

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
        rng, cmd_rng = jax.random.split(rng)

        info = {
            "rng": rng,
            "step": jp.array(0),
            "command": self._sample_command(cmd_rng),  # [vx, vyaw]
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "last_act": self.null_action(),  # for action_rate
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

        info = state.info.copy()

        # Update action history
        info["last_act"] = info["action"]
        info["prev_action"] = info["action"]
        info["action"] = action

        # Increment step counter
        new_step = info["step"] + 1
        info["step"] = new_step

        # Resample command at interval
        resample_interval = self._config.command_config.resample_interval
        should_resample = (new_step % resample_interval) == 0

        rng, cmd_rng = jax.random.split(info["rng"])
        info["rng"] = rng
        new_command = self._sample_command(cmd_rng)
        info["command"] = jp.where(should_resample, new_command, info["command"])

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

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample velocity command with probability of zero command.

        Args:
            rng: Random number generator key.

        Returns:
            Array of shape (2,) with [vx, vyaw] (forward velocity and yaw rate).
        """
        rng, zero_rng, vx_rng, vyaw_rng = jax.random.split(rng, 4)

        # Sample velocities from uniform ranges
        vx = jax.random.uniform(
            vx_rng,
            minval=self._config.lin_vel_x[0],
            maxval=self._config.lin_vel_x[1],
        )
        vyaw = jax.random.uniform(
            vyaw_rng,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        command = jp.array([vx, vyaw])

        # With zero_prob, set command to zeros
        is_zero = jax.random.uniform(zero_rng) < self._config.command_config.zero_prob
        command = jp.where(is_zero, jp.zeros(2), command)

        return command

    def _get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Get linear velocity in torso local frame.

        Args:
            data: Simulation data.

        Returns:
            Linear velocity [vx, vy, vz] in torso frame.
        """
        torso = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        world_vel = torso.subtree_linvel
        # Transform to local frame using rotation matrix
        local_vel = jp.dot(world_vel, torso.xmat.reshape(3, 3))
        return local_vel

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
                info["command"],  # Include command (2,): [vx, vyaw]
            ]
        )

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(state=obs)

    def null_action(self) -> jp.ndarray:
        """Return zero action."""
        return jp.zeros(self.action_size)

    # --- Reward Functions ---

    @_registry.reward("tracking_lin_vel")
    def _tracking_lin_vel_reward(self, data, info, metrics, weight) -> jax.Array:
        """Exponential reward for matching commanded forward velocity.

        Args:
            data: Simulation data.
            info: State info with command.
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted tracking reward.
        """
        command = info["command"]  # [vx, vyaw]
        local_vel = self._get_local_linvel(data)

        # Squared error in forward (x) velocity only
        lin_vel_error = jp.square(command[0] - local_vel[0])
        sigma = self._config.reward_config.tracking_sigma

        reward_value = jp.exp(-lin_vel_error / sigma)
        weighted_reward = reward_value * weight

        metrics["rewards/tracking_lin_vel"] = weighted_reward
        metrics["local_vel_x"] = local_vel[0]
        metrics["command_vx"] = command[0]

        return weighted_reward

    @_registry.reward("tracking_ang_vel")
    def _tracking_ang_vel_reward(self, data, info, metrics, weight) -> jax.Array:
        """Exponential reward for matching commanded yaw rate.

        Args:
            data: Simulation data.
            info: State info with command.
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted tracking reward.
        """
        command = info["command"]  # [vx, vyaw]
        # Use gyro sensor z-component for yaw rate
        gyro = data.bind(
            self.mjx_model, self._spec.sensor(f"gyro{self._suffix}")
        ).sensordata
        yaw_rate = gyro[2]

        ang_vel_error = jp.square(command[1] - yaw_rate)
        sigma = self._config.reward_config.tracking_sigma

        reward_value = jp.exp(-ang_vel_error / sigma)
        weighted_reward = reward_value * weight

        metrics["rewards/tracking_ang_vel"] = weighted_reward
        metrics["yaw_rate"] = yaw_rate
        metrics["command_vyaw"] = command[1]

        return weighted_reward

    @_registry.reward("torques")
    def _torques_cost(self, data, info, metrics, weight) -> jax.Array:
        """Cost for actuator forces.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (negative for penalty).

        Returns:
            Weighted torque cost.
        """
        del info
        torque_cost = jp.sum(jp.abs(data.qfrc_actuator))
        weighted_cost = weight * torque_cost

        metrics["rewards/torques"] = weighted_cost
        metrics["torque_sum"] = torque_cost

        return weighted_cost

    @_registry.reward("action_rate")
    def _action_rate_cost(self, data, info, metrics, weight) -> jax.Array:
        """Cost for action changes between timesteps.

        Args:
            data: Simulation data (unused).
            info: State info with action history.
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (negative for penalty).

        Returns:
            Weighted action rate cost.
        """
        del data
        action_diff = info["action"] - info["last_act"]
        action_rate = jp.sum(jp.square(action_diff))
        weighted_cost = weight * action_rate

        metrics["rewards/action_rate"] = weighted_cost
        metrics["action_rate"] = action_rate

        return weighted_cost

    @_registry.reward("energy")
    def _energy_cost(self, data, info, metrics, weight) -> jax.Array:
        """Cost for mechanical work (velocity * force).

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (negative for penalty).

        Returns:
            Weighted energy cost.
        """
        del info
        energy = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
        weighted_cost = weight * energy

        metrics["rewards/energy"] = weighted_cost
        metrics["energy"] = energy

        return weighted_cost

    @_registry.reward("dof_acc")
    def _dof_acc_cost(self, data, info, metrics, weight, max_value=None) -> jax.Array:
        """Cost for joint accelerations.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (negative for penalty).
            max_value: Optional cap on the acceleration sum before weighting.

        Returns:
            Weighted acceleration cost.
        """
        del info
        dof_acc = jp.sum(jp.square(data.qacc))
        if max_value is not None:
            dof_acc = jp.minimum(dof_acc, max_value)
        weighted_cost = weight * dof_acc

        metrics["rewards/dof_acc"] = weighted_cost
        metrics["dof_acc"] = dof_acc

        return weighted_cost

    @_registry.reward("alive")
    def _alive_reward(self, data, info, metrics, weight) -> jax.Array:
        """Constant reward for staying alive.

        Args:
            data: Simulation data (unused).
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Reward weight.

        Returns:
            Constant alive reward.
        """
        del data, info
        metrics["rewards/alive"] = weight
        return weight

    @_registry.reward("termination")
    def _termination_penalty(self, data, info, metrics, weight) -> jax.Array:
        """Large penalty on episode termination.

        Args:
            data: Simulation data.
            info: State info.
            metrics: Metrics dict (with termination flags).
            weight: Penalty weight (negative).

        Returns:
            Termination penalty (weight if terminated, 0 otherwise).
        """
        # Check if terminated (using already computed metrics)
        terminated = metrics.get("terminations/any", jp.array(0.0))
        penalty = jp.where(terminated > 0.5, weight, 0.0)
        metrics["rewards/termination"] = penalty
        return penalty

    @_registry.reward("stand_still")
    def _stand_still_penalty(self, data, info, metrics, weight) -> jax.Array:
        """Penalty for unwanted movement (moving when that axis command is zero).

        Decoupled: penalizes forward velocity only when vx command is ~0,
        and penalizes yaw rate only when vyaw command is ~0.

        Args:
            data: Simulation data.
            info: State info with command.
            metrics: Metrics dict for logging.
            weight: Penalty weight (negative).

        Returns:
            Penalty for unwanted movement.
        """
        command = info["command"]  # [vx, vyaw]
        local_vel = self._get_local_linvel(data)
        gyro = data.bind(
            self.mjx_model, self._spec.sensor(f"gyro{self._suffix}")
        ).sensordata
        yaw_rate = gyro[2]

        # Penalize forward movement only when forward command is ~0
        vx_cmd_zero = jp.abs(command[0]) < 0.05
        lin_penalty = jp.where(vx_cmd_zero, jp.square(local_vel[0]), 0.0)

        # Penalize turning only when turn command is ~0
        vyaw_cmd_zero = jp.abs(command[1]) < 0.1
        ang_penalty = jp.where(vyaw_cmd_zero, jp.square(yaw_rate), 0.0)

        penalty = weight * (lin_penalty + ang_penalty)
        metrics["rewards/stand_still"] = penalty

        return penalty

    @_registry.reward("lateral_velocity")
    def _lateral_velocity_penalty(self, data, info, metrics, weight) -> jax.Array:
        """Penalty for lateral (sideways) velocity.

        The rodent shouldn't strafe - any lateral velocity is undesired drift.
        This is especially important during turning where the rodent might
        otherwise slide sideways.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Penalty weight (negative).

        Returns:
            Weighted lateral velocity penalty.
        """
        del info
        local_vel = self._get_local_linvel(data)
        lateral_vel_sq = jp.square(local_vel[1])  # y-velocity in local frame

        penalty = weight * lateral_vel_sq
        metrics["rewards/lateral_velocity"] = penalty
        metrics["local_vel_y"] = local_vel[1]

        return penalty

    # --- Termination Functions ---

    @_registry.termination("fallen")
    def _fallen_termination(
        self, data: mjx.Data, info, min_torso_z: float, max_torso_angle: float
    ) -> jax.Array:
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

        # xmat is 3x3 rotation matrix, [2, 2] is cos(angle from vertical)
        upright_z = torso_body.xmat.reshape(3, 3)[2, 2]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_ground, too_tilted)

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> jax.Array:
        """Check for NaN values in simulation data.

        Args:
            data: Simulation data.
            info: State info (unused).

        Returns:
            Boolean indicating if NaN detected.
        """
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0
