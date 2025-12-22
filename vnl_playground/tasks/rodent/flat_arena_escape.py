# Flat arena escape definition, based on the MuJoCo Playground rodent tasks.
#
# This is a "flat arena" variant of BowlEscape: no heightfield is generated/attached.
# The rodent starts at the origin on a flat floor and is rewarded for escaping beyond
# a configurable radius.

import collections
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from jax import flatten_util

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_fns

from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=512,
        render_width=64,
        render_height=64,
        enabled_geom_groups=[0, 1, 2],
        use_rasterizer=False,
        history=3,
    )


def default_config() -> config_dict.ConfigDict:
    """Default configuration for FlatArenaEscape."""
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
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
        vision=False,
        vision_config=default_vision_config(),
        torque_actuators=True,
        rescale_factor=0.9,
        target_speed=0.75,
        episode_length=1500,
        action_repeat=1,
        # Escape task parameters
        arena_hsize=2.0,  # escape radius threshold (meters)
        spawn_pos=(0.0, 0.0, 0.05),  # initial root position on the flat floor
        min_torso_z=0.03,  # terminate if torso drops below this (meters)
        reward_terms={
            "escape": {"weight": 1.0},
            "upright": {"weight": 1.0},
            "speed": {"weight": 1.0},
        },
        termination_criteria={
            "fallen": {},
            "nan_termination": {},
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


class FlatArenaEscape(rodent_base.RodentEnv):
    """Flat arena escape environment."""

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[
            Dict[str, Union[str, int, float, list[Any], dict]]
        ] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._rng = rng

        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )
        self._vision = self._config.vision

        # Spawn rodent on the flat arena (no heightfield interpolation needed).
        sx, sy, sz = self._config.spawn_pos
        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[float(sx), float(sy), float(sz)],
        )

        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def reset(
        self, rng: jax.Array, qpos0: Optional[jp.ndarray] = None
    ) -> mjx_env.State:
        if self._vision:
            raise NotImplementedError("Vision is not implemented for FlatArenaEscape.")

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
        if qpos0 is not None:
            data = data.replace(qpos=qpos0)
            data = mjx.forward(self.mjx_model, data)

        metrics: Dict[str, Any] = {}

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

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Tuple[jp.ndarray, jp.ndarray]:
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
        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

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

    # Rewards
    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("escape")
    def _escape_reward(self, data, info, metrics, weight) -> float:
        del info
        terrain_size = float(self._config.arena_hsize)
        torso_xpos = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xpos
        dist_xy = jp.linalg.norm(torso_xpos[:2])
        escape_reward = reward_fns.tolerance(
            dist_xy,
            bounds=(terrain_size, float("inf")),
            margin=terrain_size,
            value_at_margin=0,
            sigmoid="linear",
        )
        reward = escape_reward * weight
        metrics["rewards/escape"] = reward
        return reward

    @_named_reward("upright")
    def _upright_reward(self, data, info, metrics, weight) -> float:
        del info
        deviation_angle = 0
        deviation = np.cos(np.deg2rad(deviation_angle))
        upright_torso = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xmat[
            -1, -1
        ]
        upright_head = data.bind(self.mjx_model, self._spec.body("skull-rodent")).xmat[
            -1, -1
        ]
        upright = reward_fns.tolerance(
            jp.stack([upright_torso, upright_head]),
            bounds=(deviation, np.inf),
            sigmoid="linear",
            margin=1 + deviation,
            value_at_margin=0,
        )
        reward = jp.min(upright) * weight
        metrics["rewards/upright"] = reward
        return reward

    @_named_reward("speed")
    def _speed_reward(self, data, info, metrics, weight) -> float:
        del info
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        vel = jp.linalg.norm(body.subtree_linvel)
        target_speed = self._config.target_speed
        reward = (
            reward_fns.tolerance(
                vel,
                bounds=(target_speed, target_speed),
                margin=target_speed,
                sigmoid="linear",
                value_at_margin=0.0,
            )
            * weight
        )
        metrics["rewards/speed"] = reward
        return reward

    # Termination
    def _named_termination_criterion(name: str):
        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("fallen")
    def _torso_too_low(self, data: mjx.Data, info) -> bool:
        del info
        torso_pos = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xpos
        z = torso_pos[2]
        return z <= float(self._config.min_torso_z)

    @_named_termination_criterion("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return int(jp.sum(flatten_util.ravel_pytree(obs_size["proprioception"])[0]))

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return int(self.observation_size - self.proprioceptive_obs_size)

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return int(jp.sum(flatten_util.ravel_pytree(obs)[0]))

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)
