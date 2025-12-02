"""Head tracking task (Olveczky Lab real-time mocap tracking task)"""

import collections
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Union

import brax.math
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import base as rodent_base
from . import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        mujoco_impl="jax",
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="newton",
        iterations=8,
        ls_iterations=8,
        nconmax=256,
        njmax=128,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        head_z_target=0.125,
        qvel_init="zeros",
        reward_terms={
            # Head tracking (to mimic real-time mocap tracking head)
            "hold_head_z": {"weight": 1.0, "time_threshold": 0.2},
            "head_z_dist": {"exp_scale": 0.03, "weight": 0.05},
            # Costs / regularizers
            "torso_z_range": {"healthy_z_range": (0.03, 1.0), "weight": 0.05},
            "control_cost": {"weight": 0.01},
            "control_diff_cost": {"weight": 0.01},
            "energy_cost": {"max_value": 50.0, "weight": 0.01},
        },
        termination_criteria={
            "fallen": {"healthy_z_range": (0.0325, 0.5)},  # Meters
            "upside_down": {},  # when root quat[2] < 0
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


class HeadTrackRear(rodent_base.RodentEnv):
    """Head tracking task to reinforce rearing behavior.

    Designed to mirror our real-time mocap tracking setup, this environment is defined as follows:
    - The episode begins with the rodent in a default standing pose.
    - A target z-height for the height is given by the config (for this basic version)
    - The rodent is rewarded only when it keeps its head at or above that height for 1 second.
    - Each 1 second rear gets a reward of 1.0, so it will rear as many times are possible
    within the episode (which will be n seconds, default to 10)
    - The episode terminates early if the rodent falls over (and gets a small reward for staying upright)
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
    ) -> None:
        """
        Initialize the rodent imitation environment.
        Args:
            config (config_dict.ConfigDict, optional): Configuration dictionary for the environment.
                Defaults to `default_config()`.
            config_overrides (optional):
                Dictionary of configuration overrides.
        """
        super().__init__(config, config_overrides)
        self.add_rodent(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
        )
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """
        Resets the environment state: draws a new reference clip and initializes the rodent's pose to match.
        Args:
            rng (jax.Array): JAX random number generator stare.
        Returns:
            mjx_env.State: The initial state of the environment after reset.
        """

        start_rng, clip_rng = jax.random.split(rng)

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel=jp.zeros(self.mjx_model.nv))

        data = mjx.forward(self.mjx_model, data)

        info: dict[str, Any] = {}
        info["action"] = self.null_action()
        info["above_z_time"] = 0.0

        zero = 0.0
        metrics = {
            "nans": zero,
            "reward": zero,
        }

        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        """Step the environment forward.
        Args:
            state (mjx_env.State): Current environment state.
            action (jax.Array): Action to apply.
        Returns:
            mjx_env.State: The new state of the environment.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info

        info["action"] = action

        # Check if the head is above the target z-height
        # If yes, increment info["above_z_time"] by self._config.ctrl_dt
        head_xpos = self._get_head_xpos(data)
        above_target = head_xpos[2] >= self._config.head_z_target
        info["above_z_time"] = jp.where(
            above_target, info["above_z_time"] + self._config.ctrl_dt, 0.0
        )

        obs = self._get_obs(data, info)
        terminated = self._is_done(data, info, state.metrics)
        done = terminated
        reward = self._get_reward(data, info, state.metrics)

        # Handle nans during sim by resetting env
        # reward = jp.nan_to_num(reward)
        # flattened_obs, _ = jax.flatten_util.ravel_pytree(obs)
        # flattened_vals, _ = jax.flatten_util.ravel_pytree(data)
        # num_nans = jp.sum(jp.isnan(flattened_vals)) + jp.sum(jp.isnan(flattened_obs))
        # nan = jp.where(num_nans > 0, 1.0, 0.0)
        # done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            # nans=nan,
            reward=reward,
        )
        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    # the proprioceptive obs in main branch because it matches track-mjx obs
    def _get_proprioception(self, data: mjx.Data, flatten: bool = True) -> jp.ndarray:
        """Get proprioception data from the environment."""
        qpos = data.qpos[7:]  # skip the root joint
        qvel = data.qvel[6:]  # skip the root joint velocity
        actuator_ctrl = data.qfrc_actuator
        _, body_height, _ = data.bind(
            self.mjx_model, self._spec.body(f"torso{self._suffix}")
        ).xpos
        world_zaxis = data.bind(
            self.mjx_model, self._spec.body(f"torso{self._suffix}")
        ).xmat.flatten()[6:]
        appendages_pos = self._get_appendages_pos(data)
        proprioception = collections.OrderedDict(
            joint_angles=qpos,
            joint_ang_vels=qvel,
            actuator_ctrl=actuator_ctrl,
            body_height=jp.array([body_height]),
            world_zaxis=world_zaxis,
            appendages_pos=appendages_pos,
        )
        if flatten:
            proprioception, _ = jax.flatten_util.ravel_pytree(proprioception)
        return proprioception

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        return collections.OrderedDict(
            proprioception=self._get_proprioception(data, flatten=False),
        )

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics) -> bool:
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            terminated = termination_fcn(self, data, info, **kwargs)
            any_terminated = jp.logical_or(any_terminated, terminated)
            # Also log terminations as floats so averaging -> hazard rate
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    # Rewards
    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("hold_head_z")
    def _hold_head_z_reward(self, data, info, metrics, weight, time_threshold) -> float:
        in_reward_window = jp.logical_and(
            info["above_z_time"] >= time_threshold,
            info["above_z_time"] < time_threshold + self._config.ctrl_dt,
        )
        reward = jp.where(in_reward_window, weight, 0.0)
        metrics["rewards/hold_head_z"] = reward
        return reward

    @_named_reward("head_z_dist")
    def _head_z_dist_reward(self, data, info, metrics, weight, exp_scale) -> float:
        head_z = self._get_head_xpos(data)[2]
        head_z_dist = jp.abs(head_z - (self._config.head_z_target*1.1)) # set target to 10% above threshold 
        reward = weight * jp.exp(-((head_z_dist / exp_scale) ** 2) / 2)
        metrics["rewards/head_z_dist"] = reward
        return reward

    @_named_reward("torso_z_range")
    def _torso_z_range_reward(
        self, data, info, metrics, weight, healthy_z_range
    ) -> float:
        metrics["torso_z"] = torso_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        metrics["in_range"] = in_range.astype(float)
        reward = weight * in_range
        metrics["rewards/torso_z_range"] = reward
        return reward

    @_named_reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_sqr"] = ctrl_sqr = jp.sum(jp.square(info["action"]))
        cost = weight * ctrl_sqr
        metrics["costs/control_cost"] = cost
        return -cost

    @_named_reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = jp.sum(jp.square(info["action"]))
        cost = weight * ctrl_diff_sqr
        metrics["costs/control_diff_cost"] = cost
        return -cost

    @_named_reward("energy_cost")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        energy_use = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["costs/energy_cost"] = cost
        return -cost

    @_named_reward("jerk_cost")
    def _jerk_cost(self, data, info, weight, metrics, window_len) -> float:
        raise NotImplementedError("jerk_cost is not implemented")

    # Termination
    def _named_termination_criterion(name: str):
        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("fallen")
    def _fallen(self, data, info, healthy_z_range) -> bool:
        torso_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        fall = jp.logical_or(torso_z < min_z, torso_z > max_z)
        return fall

    @_named_termination_criterion("upside_down")
    def _upside_down(self, data, info) -> bool:
        root_quat = data.qpos[3:7]
        up_dot = root_quat[2]
        upside_down = jp.where(up_dot < 0.0, 1.0, 0.0)
        return upside_down

    def _get_head_xpos(self, data: mjx.Data) -> jp.ndarray:
        return data.bind(
            self.mjx_model, self._spec.body(f"{consts.HEAD}{self._suffix}")
        ).xpos
