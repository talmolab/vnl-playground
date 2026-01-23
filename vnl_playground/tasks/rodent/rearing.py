"""Rearing task for virtual rodent.

The rodent is initialized in a neutral standing pose and must raise its head
above a target height relative to its torso, emulating a rearing motion.
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

from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="cg",
        mujoco_impl="jax",
        naconmax=16 * 8192,
        njmax=512,
        iterations=10,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        # Rearing-specific config
        target_relative_height=0.05,  # Target skull-torso height difference
        episode_length=1000,
        action_repeat=1,
        # Reward terms (both dense and sparse available)
        reward_terms={
            "head_height_dense": {"weight": 1.0},  # Dense: progress toward target
            "head_height_sparse": {"weight": 1.0},  # Sparse: binary above threshold
            "upright": {"healthy_z_range": (0.02, 0.5), "weight": 0.5},
            "energy_cost": {"max_value": 50.0, "weight": 0.01},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.02, "max_torso_angle": 80},
            "nan_termination": {},
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


def _named_reward(name: str):
    """Decorator to register a reward function."""

    def decorator(reward_fcn: Callable):
        _REWARD_FCN_REGISTRY[name] = reward_fcn
        return reward_fcn

    return decorator


def _named_termination_criterion(name: str):
    """Decorator to register a termination criterion."""

    def decorator(termination_fcn: Callable):
        _TERMINATION_FCN_REGISTRY[name] = termination_fcn
        return termination_fcn

    return decorator


class Rearing(rodent_base.RodentEnv):
    """Rearing environment - rodent must raise head above target height."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        clips: Optional[Any] = None,  # Accepted for loader compatibility, unused
    ) -> None:
        del clips  # Rearing task does not use reference clips
        super().__init__(config, config_overrides)

        # Initialize rodent at origin, standing pose
        init_x, init_y, init_z = 0.0, 0.0, 0.03
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
        # Compute forward kinematics to get body positions
        data = mjx.forward(self.mjx_model, data)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
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

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> collections.OrderedDict:
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        # Include relative head height in observation
        relative_height = self._get_relative_head_height(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
                jp.array([relative_height]),  # Add head height info
            ]
        )

        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _get_relative_head_height(self, data: mjx.Data) -> float:
        """Get skull height relative to torso."""
        skull_body = data.bind(self.mjx_model, self._spec.body(f"skull{self._suffix}"))
        torso_body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        return skull_body.xpos[2] - torso_body.xpos[2]

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics) -> bool:
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
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def null_action(self) -> jp.ndarray:
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
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)


# --- Reward Functions ---


@_named_reward("head_height_dense")
def _head_height_dense_reward(env, data, info, metrics, weight) -> float:
    """Dense reward: smooth increase as head approaches target height."""
    relative_height = env._get_relative_head_height(data)
    target = env._config.target_relative_height
    metrics["relative_head_height"] = relative_height

    # Tolerance-based smooth reward
    reward_value = reward_fns.tolerance(
        relative_height,
        bounds=(target, float("inf")),
        margin=target,
        sigmoid="linear",
        value_at_margin=0.0,
    )
    weighted_reward = reward_value * weight
    metrics["rewards/head_height_dense"] = weighted_reward
    return weighted_reward


@_named_reward("head_height_sparse")
def _head_height_sparse_reward(env, data, info, metrics, weight) -> float:
    """Sparse reward: binary reward when head exceeds target height."""
    relative_height = env._get_relative_head_height(data)
    target = env._config.target_relative_height

    above_target = relative_height >= target
    weighted_reward = jp.astype(above_target, float) * weight
    metrics["rewards/head_height_sparse"] = weighted_reward
    return weighted_reward


@_named_reward("upright")
def _upright_reward(env, data, info, metrics, weight, healthy_z_range) -> float:
    """Reward for keeping torso in healthy z range (staying upright)."""
    torso_z = env._get_body_height(data)
    metrics["torso_z"] = torso_z
    min_z, max_z = healthy_z_range
    in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
    weighted_reward = jp.astype(in_range, float) * weight
    metrics["rewards/upright"] = weighted_reward
    return weighted_reward


@_named_reward("energy_cost")
def _energy_cost(env, data, info, metrics, weight, max_value) -> float:
    """Penalize energy consumption."""
    energy_use = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
    metrics["energy_use"] = energy_use
    cost = weight * jp.minimum(energy_use, max_value)
    metrics["rewards/energy_cost"] = -cost
    return -cost


# --- Termination Functions ---


@_named_termination_criterion("fallen")
def _fallen_termination(
    env, data: mjx.Data, info, min_torso_z: float, max_torso_angle: float
) -> bool:
    """Check if rodent has fallen."""
    del info
    torso_body = data.bind(env.mjx_model, env._spec.body(f"torso{env._suffix}"))
    torso_z = torso_body.xpos[2]
    below_ground = torso_z < min_torso_z

    # Check torso orientation - xmat[-1, -1] is the z-component of the z-axis
    upright_z = torso_body.xmat.reshape(3, 3)[2, 2]
    max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
    too_tilted = upright_z < max_cos_angle

    return jp.logical_or(below_ground, too_tilted)


@_named_termination_criterion("nan_termination")
def _nan_termination(env, data, info) -> bool:
    """Check for NaN values in simulation data."""
    del info
    flattened_vals, _ = flatten_util.ravel_pytree(data)
    num_nans = jp.sum(jp.isnan(flattened_vals))
    return num_nans > 0
