"""Maintain velocity task for Fauna Robotics Sprout humanoid.

The Sprout is initialized in standing pose facing forward (+x axis)
and must maintain a target forward velocity. This is a simple proof-of-concept
locomotion task.

Termination occurs if:
- Torso becomes too tilted (fallen)
- Torso goes below a minimum height
- NaN detected in simulation data
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
from mujoco_playground._src import reward as reward_fns

from vnl_playground.tasks.sprout import base as sprout_base
from vnl_playground.tasks.sprout import consts


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the MaintainVelocity environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        walker_xml_path=consts.SPROUT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="newton",
        mujoco_impl="jax",
        naconmax=16 * 8192,
        njmax=512,
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        target_speed=0.3,
        episode_length=2000,
        action_repeat=1,
        reward_terms={
            "forward_velocity": {"weight": 1.0},
            "lateral_velocity": {"weight": 0.0},
            "angular_velocity_z": {"weight": 0.0},
            "upright": {"weight": 0.5},
            "energy_penalty": {"weight": 0.0},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.3, "max_torso_angle": 60},
            "nan_termination": {},
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


class MaintainVelocity(sprout_base.SproutEnv):
    """Maintain velocity environment for Sprout humanoid.

    The robot must maintain a target forward velocity in the +x direction.
    Initialized in standing pose facing forward.
    """

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

        # Initialize Sprout at spawn height facing forward (+x direction)
        init_x, init_y = 0.0, 0.0
        init_z = consts.SPAWN_HEIGHT
        init_quat = (1, 0, 0, 0)

        self.add_sprout(
            torque_actuators=self._config.torque_actuators,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
        )
        self._spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        self.compile()

        # Set initial standing pose via keyframe
        self._init_qpos = self._compute_init_qpos()

    def _compute_init_qpos(self) -> jp.ndarray:
        """Compute initial qpos with standing pose.

        Returns qpos array with freejoint (7) + joint positions (26).
        """
        qpos = np.zeros(self._mj_model.nq)
        # Freejoint: [x, y, z, qw, qx, qy, qz]
        qpos[2] = consts.SPAWN_HEIGHT  # z position
        qpos[3] = 1.0  # qw = 1 (identity quaternion)

        # Set standing pose joint angles
        for i, joint in enumerate(self._spec.joints):
            if joint.name == "root":
                continue
            joint_name = joint.name
            # Strip suffix if present
            base_name = joint_name.replace(self._suffix, "")
            if base_name in consts.STANDING_POSE:
                # Joint index in qpos: 7 (freejoint) + joint_index
                qpos_idx = self._mj_model.jnt_qposadr[
                    self._mj_model.joint(joint_name).id
                ]
                qpos[qpos_idx] = consts.STANDING_POSE[base_name]

        return jp.array(qpos)

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

        # Apply standing pose
        data = data.replace(qpos=self._init_qpos)

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
        obs = self._get_obs(data, info)

        info["prev_action"] = info["action"]
        info["action"] = action

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

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> collections.OrderedDict:
        """Get the current observation from the simulation data.

        Args:
            data: The simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with task_obs and proprioception keys.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        origin = self._get_origin(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                origin,
            ]
        )

        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _is_done(
        self, data: mjx.Data, info: Mapping[str, Any], metrics
    ) -> bool:
        """Check if episode should terminate."""
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            terminated = termination_fcn(self, data, info, **kwargs)
            any_terminated = jp.logical_or(any_terminated, terminated)
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        """Compute total reward."""
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    # ---- Reward functions ----

    def _named_reward(name: str):
        """Decorator to register reward functions."""

        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("forward_velocity")
    def _forward_velocity_reward(self, data, info, metrics, weight) -> float:
        """Reward for maintaining target forward velocity in +x direction."""
        del info

        body = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
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

    @_named_reward("lateral_velocity")
    def _lateral_velocity_cost(self, data, info, metrics, weight) -> float:
        """Penalty for lateral (y-direction) velocity."""
        del info
        body = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        lateral_vel = body.subtree_linvel[1]
        cost = -weight * jp.square(lateral_vel)
        metrics["rewards/lateral_velocity"] = cost
        return cost

    @_named_reward("angular_velocity_z")
    def _angular_velocity_z_cost(self, data, info, metrics, weight) -> float:
        """Penalty for yaw rate (z-axis angular velocity) to prevent turning."""
        del info
        angvel = data.bind(
            self.mjx_model,
            self._spec.sensor(f"torso_link_site_angvel{self._suffix}"),
        ).sensordata
        yaw_rate = angvel[2]
        cost = -weight * jp.square(yaw_rate)
        metrics["rewards/angular_velocity_z"] = cost
        return cost

    @_named_reward("upright")
    def _upright_reward(self, data, info, metrics, weight) -> float:
        """Reward for keeping torso upright (z-axis aligned with world up)."""
        del info
        torso_body = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        # xmat[2,2] = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        reward_value = reward_fns.tolerance(
            upright_z,
            bounds=(0.9, 1.0),
            margin=0.5,
            sigmoid="linear",
            value_at_margin=0.0,
        )
        weighted_reward = reward_value * weight
        metrics["rewards/upright"] = weighted_reward
        return weighted_reward

    @_named_reward("energy_penalty")
    def _energy_penalty(self, data, info, metrics, weight) -> float:
        """Penalty for high actuator forces (energy efficiency)."""
        del info
        ctrl = data.ctrl
        cost = -weight * jp.sum(jp.square(ctrl))
        metrics["rewards/energy_penalty"] = cost
        return cost

    # ---- Termination criteria ----

    def _named_termination_criterion(name: str):
        """Decorator to register termination functions."""

        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = 0.3,
        max_torso_angle: float = 60,
    ) -> bool:
        """Check if robot has fallen.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum z height threshold for torso.
            max_torso_angle: Maximum angle from vertical in degrees.

        Returns:
            Boolean indicating if fallen.
        """
        del info

        torso_body = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        torso_z = torso_body.xpos[2]

        below_threshold = torso_z < min_torso_z

        # xmat[-1, -1] is element (2,2) = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_threshold, too_tilted)

    @_named_termination_criterion("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Check for NaN values in simulation data."""
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    def null_action(self) -> jp.ndarray:
        """Return zero action."""
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["proprioception"])[0])

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(
            lambda x: jp.prod(jp.array(x.shape)), obs
        )
