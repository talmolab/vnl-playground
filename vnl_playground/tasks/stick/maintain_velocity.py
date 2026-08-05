"""Maintain velocity task for stick bug.

The stick bug is initialized in a neutral pose and must maintain a target
forward velocity. The stick insect model is very small (millimeter scale),
so thresholds and target speeds are scaled accordingly.

Termination occurs if:
- Body becomes too tilted (fallen)
- Body goes below ground level
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
from vnl_playground.tasks.stick import base as stick_base
from vnl_playground.tasks.stick import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.STICK_XML_PATH,
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
        torque_actuators=False,
        rescale_factor=0.9,
        init_z=0.005,
        target_speed=0.01,  # Stick bug is mm-scale, so speed is small
        episode_length=2000,
        action_repeat=1,
        reward_terms={
            "forward_velocity": {"weight": 1.0},
            "lateral_velocity": {"weight": 0.0},
            "angular_velocity_z": {"weight": 0.0},
        },
        termination_criteria={
            "fallen": {"min_body_z": 0.001, "max_body_angle": 60},
            "nan_termination": {},
        },
    )


_registry = RewardRegistry()


class MaintainVelocity(stick_base.StickBugEnv):
    """Maintain velocity environment for stick bug.

    The stick bug must maintain a target forward velocity in the +x direction.
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict[str, str | int | list[Any]] | None = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._rng = rng

        init_x, init_y, init_z = 0.0, 0.0, self._config.init_z
        init_quat = (1, 0, 0, 0)

        self.add_stick(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
        )
        self._spec.worldbody.add_light(pos=[0, 0, 0.1], dir=[0, 0, -1])
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
        origin = self._get_origin(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
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
        del info
        body = data.bind(
            self.mjx_model,
            self._spec.body(f"reference_base{self._suffix}"),
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

    @_registry.reward("lateral_velocity")
    def _lateral_velocity_cost(self, data, info, metrics, weight) -> float:
        del info
        body = data.bind(
            self.mjx_model,
            self._spec.body(f"reference_base{self._suffix}"),
        )
        lateral_vel = body.subtree_linvel[1]
        cost = -weight * jp.square(lateral_vel)
        metrics["rewards/lateral_velocity"] = cost
        return cost

    @_registry.reward("angular_velocity_z")
    def _angular_velocity_z_cost(self, data, info, metrics, weight) -> float:
        del info
        gyro = data.bind(
            self.mjx_model,
            self._spec.sensor(f"gyro{self._suffix}"),
        ).sensordata
        yaw_rate = gyro[2]
        cost = -weight * jp.square(yaw_rate)
        metrics["rewards/angular_velocity_z"] = cost
        return cost

    @_registry.termination("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_body_z: float = 0.001,
        max_body_angle: float = 60,
    ) -> bool:
        del info
        root_body = data.bind(
            self.mjx_model,
            self._spec.body(f"reference_base{self._suffix}"),
        )
        body_z = root_body.xpos[2]
        below_ground = body_z < min_body_z

        upright_z = root_body.xmat[-1, -1]
        max_cos_angle = np.cos(np.deg2rad(max_body_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_ground, too_tilted)

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        return jp.any(jp.isnan(data.qpos))

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)
