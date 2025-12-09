from typing import Any, Dict, Optional, Union, Tuple, Callable, Mapping

import collections

from etils import epath
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as rw
from mujoco_playground._src.dm_control_suite import common

from vnl_mjx.tasks.rodent import base as rodent_base
from vnl_mjx.tasks.rodent import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        iterations=5,
        ls_iterations=5,
        mujoco_impl="jax",
        solver = 'newton',
        nconmax=256,
        njmax=128,
        noslip_iterations=0, #added to avoid error in train.py
        torque_actuators=True,
        rescale_factor=0.9,
        episode_length=1000,
        action_repeat=1,
        action_scale=1,
        energy_termination_threshold=np.inf,
        target_speed=0.5,
        reward_terms = {
            "progress": {"weight": 0.5},   # COM progress in +x
            "speed": {"weight": 0.3},   # Gaussian around target_speed
            "upright": {"weight": 0.2},  # improved tilt-based
            "control_cost": {"weight": 1e-3},
            "control_diff_cost": {"weight": 1e-3},
            "energy_cost": {"max_value": 50.0, "weight": 5e-4},
        },
        termination_criteria={
            "nan_termination": {},
            "fallen": {"healthy_z_range": (0.0325, 0.5)},  # Meters
        }
    )

_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}

class FlatWalk(rodent_base.RodentEnv):
    """Flat walk environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self.add_rodent(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
        )
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Sample a random initial configuration with some probability.

        data = mjx_env.init(self.mjx_model)
        zero = 0.0
        metrics = {
            "nans": zero,
            "reward": zero,
        }

        task_obs, proprioceptive_obs = self._get_obs(data)
        obs = jp.concatenate([task_obs, proprioceptive_obs])
        # TODO: currently, this denotes the task specific inputs
        task_obs_size = task_obs.shape[0]
        proprioceptive_obs_size = proprioceptive_obs.shape[0]
        info = {
            # need to use this name for compatibility with track-mjx training scripts
            "reference_obs_size": task_obs_size,  # TODO: change name to task obs size
            "proprioceptive_obs_size": proprioceptive_obs_size,
        }

        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()

        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        cur_x = torso.xpos[0]
        info["prev_x"] = cur_x  # jp.array(cur_x) is also fine

        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        # Unpack the state.

        # Apply the action to the model.
        data = mjx_env.step(self.mjx_model, state.data, action)
        
        info = state.info

        info["prev_action"] = state.info["action"]
        info["action"] = action

        # Get the new observation.
        task_obs, proprioceptive_obs = self._get_obs(data)
        obs = jp.concatenate([task_obs, proprioceptive_obs])

        # Compute the reward.
        reward = self._get_reward(data, info, state.metrics)
        #reward = rewards["speed * upright"]

        termination = self._is_done(data, info, state.metrics)

        done = jp.astype(termination, float)

        state.metrics.update(
            # nans=nan,
            reward=reward,
        )

        state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
        )

        return state
    
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
    

    def _get_obs(self, data: mjx.Data) -> Tuple[jp.ndarray, jp.ndarray]:
        """Get the current observation from the simulation data.

        Args:
            data (mjx.Data): The simulation data.

        Returns:
            jp.ndarray: The concatenated position and velocity observations.
        """
        proprioception = self._get_proprioception(data)
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        task_obs = jp.concatenate(
            [
                data.bind(self.mjx_model, self._spec.body("torso-rodent")).xpos,
                proprioception,
                kinematic_sensors,
                touch_sensors,
            ]
        )

        #proprioceptive_obs = jp.concatenate(
        #    [
        #        # align with the most recent checkpoint
        #        data.qpos[7:],
        #        data.qvel[6:],
        #        data.qfrc_actuator,
        #        self._get_appendages_pos(data),
        #        self._get_kinematic_sensors(data),
        #    ]
        #)
        return task_obs, proprioception

    #def _get_reward(
    #    self,
    #    data: mjx.Data,
    #) -> Dict[str, jax.Array]:
    #    speed_reward = self._get_speed_reward(data)
    #    upright_reward = self._upright_reward(data, deviation_angle=10)
    #    return {
    #        "speed_reward": speed_reward,
    #        "upright_reward": upright_reward,
    #        "speed * upright": speed_reward * upright_reward,
    #    }

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)
    
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

    #@_named_reward("speed")
    #def _speed_reward(
    #    self, data: mjx.Data, info, metrics, weight) -> jp.ndarray:
    #    body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
    #    vel = jp.linalg.norm(body.subtree_linvel)
    #    target_speed = self._config.target_speed
    #    reward_value = rw.tolerance(
    #        vel, bounds=(target_speed, target_speed), margin=target_speed
    #    )
    #    metrics["rewards/speed"] = reward_value*weight
    #    return reward_value*weight
    
    @_named_reward("progress")
    def _progress_reward(self, data: mjx.Data, info, metrics, weight) -> jp.ndarray:
        """Reward based on forward COM progress along world +x.

        Only counts positive forward displacement (no reward for moving backward).
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        cur_x = torso.xpos[0]

        # Get previous x; fall back to current x if not set (e.g. first step)
        prev_x = info.get("prev_x", cur_x)

        # Forward progress only
        delta_x = jp.maximum(cur_x - prev_x, 0.0)

        # Update prev_x for the next step
        info["prev_x"] = cur_x

        # Scale and log
        reward = delta_x * weight
        metrics["rewards/progress"] = reward
        return reward


    @_named_reward("speed")
    def _speed_reward(
        self, data: mjx.Data, info, metrics, weight
    ) -> jp.ndarray:
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        vel_world = body.subtree_linvel  # (3,)

        # xmat is a flattened 3×3 rotation matrix, row-major
        R = body.xmat.reshape(3, 3)
        # Assume local +x axis is "nose forward"
        forward_dir = R[:, 0]
        #forward_dir = jp.array([1.0, 0.0, 0.0])

        forward_speed = jp.dot(vel_world, forward_dir)
        forward_speed = jp.maximum(forward_speed, 0.0)

        target_speed = self._config.target_speed
        speed_error = (forward_speed - target_speed) / target_speed
        reward_value = jp.exp(-0.5 * (speed_error ** 2) / 0.25)

        #target_speed = self._config.target_speed
        #reward_value = rw.tolerance(
        #    forward_speed,
        #    bounds=(target_speed, target_speed),
        #    margin=target_speed,
        #)

        metrics["rewards/forward_speed"] = reward_value * weight
        return reward_value * weight

    @_named_reward("upright")
    def _upright_reward(self, data, info, metrics, weight, deviation_angle=30):
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        R = torso.xmat.reshape(3, 3)
        # world z-axis:
        z_world = jp.array([0., 0., 1.])
        # torso's local z-axis in world frame:
        z_torso = R[:, 2]
        cos_angle = jp.dot(z_torso, z_world)  # 1 = upright, -1 = upside-down

        # IMPORTANT: use NumPy here so these are *Python floats*, not JAX tracers
        deviation = float(np.cos(np.deg2rad(deviation_angle)))  # e.g. cos(30°)
        margin = float(1.0 - deviation)                         # > 0 scalar

        upright = rw.tolerance(
            cos_angle,
            bounds=(deviation, 1.0),      # lower/upper as Python floats
            sigmoid="quadratic",
            margin=margin,                # Python float
            # value_at_margin default is fine, or set explicitly if you want
            # value_at_margin=0.0,
        )
        
        reward = upright * weight
        metrics["rewards/upright"] = reward
        return reward
    
    @_named_reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_sqr"] = ctrl_sqr = jp.sum(jp.square(info["action"]))
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = cost
        return -cost

    @_named_reward("control_diff_cost")
    def _control_diff_cost(
        self, data, info, metrics, weight
    ) -> float:
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = jp.sum(
            jp.square(info["action"] - info["prev_action"])
        )
        cost = weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = cost
        return -cost

    @_named_reward("energy_cost")
    def _energy_cost(
        self, data, info, metrics, weight, max_value
    ) -> float:
        energy_use = jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.qfrc_actuator[6:]))
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = cost
        return -cost

    def _get_termination(
        self,
        data: mjx.Data,
    ) -> jax.Array:
        """
        Returns 1 if the rodent falls under the ground, 0 otherwise.

        Args:
            data (mjx.Data): _description_

        Returns:
            jax.Array: _description_
        """
        z = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xpos[-1]
        fall_under_ground = jp.where(z < 0.03, 1.0, 0.0)
        return fall_under_ground
    
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

    @_named_termination_criterion("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        # Handle nans during sim by resetting env
        flattened_vals, _ = jax.flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0
