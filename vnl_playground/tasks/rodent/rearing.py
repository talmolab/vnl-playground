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
from vnl_playground.tasks.reward_registry import RewardRegistry

_registry = RewardRegistry()


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
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
        # Rearing-specific config
        target_relative_height=0.05,  # Target skull-torso height difference
        reset_height_threshold=0.01,  # Must return below this to earn another reward
        rearing_hold_duration=1.0,  # Seconds to hold rearing pose for success
        episode_length=1000,  # 10 seconds at ctrl_dt=0.01
        action_repeat=1,
        # Reward terms: shaping rewards are low, success bonus dominates
        reward_terms={
            "head_height_dense": {"weight": 0.0},  # Shaping: progress toward target
            "head_height_sparse": {"weight": 0.0},  # Shaping: binary above threshold
            "upright": {"healthy_z_range": (0.02, 0.5), "weight": 0.2},
            "energy_cost": {"max_value": 50.0, "weight": 0.01},
            "rearing_success": {"weight": 500.0},  # Large bonus for completing task
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.02, "max_torso_angle": 80},
            "nan_termination": {},
        },
    )


class Rearing(rodent_base.RodentEnv):
    """Rearing environment - rodent must raise head above target height."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
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
            "rearing_steps": jp.array(0),  # Track consecutive steps in rearing pose
            "rearing_success": jp.array(False),  # Whether rearing was just completed
            "can_earn_rearing": jp.array(True),  # Must reset before earning again
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

        info = state.info.copy()
        obs = self._get_obs(data, info)

        info["prev_action"] = info["action"]
        info["action"] = action

        # Update rearing step counter with reset requirement
        relative_height = self._get_relative_head_height(data)
        is_rearing = relative_height >= self._config.target_relative_height
        is_reset_position = relative_height <= self._config.reset_height_threshold

        # Increment counter only while rearing
        new_rearing_steps = jp.where(
            is_rearing,
            info["rearing_steps"] + 1,
            jp.array(0),
        )

        required_steps = int(self._config.rearing_hold_duration / self._config.ctrl_dt)
        reached_threshold = new_rearing_steps >= required_steps

        # Only award if we've reset since last success
        can_earn = info["can_earn_rearing"]
        rearing_success = jp.logical_and(reached_threshold, can_earn)
        info["rearing_success"] = rearing_success

        # State machine for can_earn_rearing:
        # - After success: must reset before earning again
        # - At reset position: become eligible again
        info["can_earn_rearing"] = jp.where(
            rearing_success,
            jp.array(False),  # Just succeeded, need to go back down
            jp.where(
                is_reset_position,
                jp.array(True),  # Returned to low position, can earn again
                info["can_earn_rearing"],  # Otherwise keep current state
            ),
        )

        # Reset step counter when threshold reached
        info["rearing_steps"] = jp.where(
            reached_threshold,
            jp.array(0),
            new_rearing_steps,
        )

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

        # Include relative head height in observation
        relative_height = self._get_relative_head_height(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                jp.array([relative_height]),  # Add head height info
            ]
        )

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

        # Privileged observations include rearing progress info
        required_steps = int(self._config.rearing_hold_duration / self._config.ctrl_dt)
        privileged_task_obs = jp.concatenate(
            [
                task_obs,
                jp.array(
                    [info["rearing_steps"] / required_steps]
                ),  # Normalized progress
                jp.array([info["can_earn_rearing"]], dtype=float),  # Eligibility
            ]
        )
        privileged_obs = collections.OrderedDict(
            task_obs=privileged_task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

        return collections.OrderedDict(
            state=obs,
            privileged_state=privileged_obs,
        )

    def _get_relative_head_height(self, data: mjx.Data) -> float:
        """Get skull height relative to torso."""
        skull_body = data.bind(self.mjx_model, self._spec.body(f"skull{self._suffix}"))
        torso_body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        return skull_body.xpos[2] - torso_body.xpos[2]

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    # --- Reward Functions ---

    @_registry.reward("head_height_dense")
    def _head_height_dense_reward(self, data, info, metrics, weight) -> float:
        """Dense reward: smooth increase as head approaches target height."""
        relative_height = self._get_relative_head_height(data)
        target = self._config.target_relative_height
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

    @_registry.reward("head_height_sparse")
    def _head_height_sparse_reward(self, data, info, metrics, weight) -> float:
        """Sparse reward: binary reward when head exceeds target height."""
        relative_height = self._get_relative_head_height(data)
        target = self._config.target_relative_height

        above_target = relative_height >= target
        weighted_reward = jp.astype(above_target, float) * weight
        metrics["rewards/head_height_sparse"] = weighted_reward
        return weighted_reward

    @_registry.reward("upright")
    def _upright_reward(self, data, info, metrics, weight, healthy_z_range) -> float:
        """Reward for keeping torso in healthy z range (staying upright)."""
        torso_z = self._get_body_height(data)
        metrics["torso_z"] = torso_z
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        weighted_reward = jp.astype(in_range, float) * weight
        metrics["rewards/upright"] = weighted_reward
        return weighted_reward

    @_registry.reward("energy_cost")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        """Penalize energy consumption."""
        energy_use = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = -cost
        return -cost

    @_registry.reward("rearing_success")
    def _rearing_success_reward(self, data, info, metrics, weight) -> float:
        """Large reward when rearing pose is held for the required duration."""
        rearing_success = info["rearing_success"]
        weighted_reward = jp.astype(rearing_success, float) * weight
        metrics["rewards/rearing_success"] = weighted_reward
        metrics["rearing_steps"] = info["rearing_steps"]
        metrics["can_earn_rearing"] = jp.astype(info["can_earn_rearing"], float)
        return weighted_reward

    # --- Termination Functions ---

    @_registry.termination("fallen")
    def _fallen_termination(
        self, data: mjx.Data, info, min_torso_z: float, max_torso_angle: float
    ) -> bool:
        """Check if rodent has fallen."""
        del info
        torso_body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        torso_z = torso_body.xpos[2]
        below_ground = torso_z < min_torso_z

        # Check torso orientation - xmat[-1, -1] is the z-component of the z-axis
        upright_z = torso_body.xmat.reshape(3, 3)[2, 2]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_ground, too_tilted)

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Check for NaN values in simulation data."""
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0
