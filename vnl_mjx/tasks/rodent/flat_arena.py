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
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

from vnl_mjx.tasks.rodent import base as rodent_base
from vnl_mjx.tasks.rodent import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        iterations=4,
        ls_iterations=4,
        mujoco_impl="jax",
        solver = 'cg',
        noslip_iterations=0, #added to avoid error in train.py
        torque_actuators=True,
        rescale_factor=0.9,
        episode_length=1000,
        action_repeat=1,
        action_scale=1,
        energy_termination_threshold=np.inf,
        target_speed=0.5,
        termination_criteria={
            "nan_termination": {},
            "fallen": {"healthy_z_range": (0.0325, 0.5)},  # Meters
        }
    )

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

        task_obs, proprioceptive_obs = self._get_obs(data)
        obs = jp.concatenate([task_obs, proprioceptive_obs])
        reward, done = jp.zeros(2)
        metrics = {}
        # TODO: currently, this denotes the task specific inputs
        task_obs_size = task_obs.shape[0]
        proprioceptive_obs_size = proprioceptive_obs.shape[0]
        info = {
            # need to use this name for compatibility with track-mjx training scripts
            "reference_obs_size": task_obs_size,  # TODO: change name to task obs size
            "proprioceptive_obs_size": proprioceptive_obs_size,
        }

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

        term_criteria = self._is_done(data, info, state.metrics)

        # Get the new observation.
        task_obs, proprioceptive_obs = self._get_obs(data)
        obs = jp.concatenate([task_obs, proprioceptive_obs])

        # Compute the reward.
        rewards = self._get_reward(data)
        reward = rewards["speed * upright"]
        termination = self._get_termination(data)

        done = jp.logical_or(termination, term_criteria)
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

        proprioceptive_obs = jp.concatenate(
            [
                # align with the most recent checkpoint
                data.qpos[7:],
                data.qvel[6:],
                data.qfrc_actuator,
                self._get_appendages_pos(data),
                self._get_kinematic_sensors(data),
            ]
        )
        return task_obs, proprioceptive_obs

    def _get_reward(
        self,
        data: mjx.Data,
    ) -> Dict[str, jax.Array]:
        speed_reward = self._get_speed_reward(data)
        upright_reward = self._upright_reward(data, deviation_angle=10)
        return {
            "speed_reward": speed_reward,
            "upright_reward": upright_reward,
            "speed * upright": speed_reward * upright_reward,
        }

    def _get_speed_reward(
        self,
        data: mjx.Data,
    ) -> jp.ndarray:
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        vel = jp.linalg.norm(body.subtree_linvel)
        target_speed = self._config.target_speed
        reward_value = reward.tolerance(
            vel, bounds=(target_speed, target_speed), margin=target_speed
        )
        return reward_value

    def _upright_reward(self, data: mjx.Data, deviation_angle=0):
        """Returns a reward proportional to how upright the torso is.

        Args:
        physics: an instance of `Physics`.
        walker: the focal walker.
        deviation_angle: A float, in degrees. The reward is 0 when the torso is
            exactly upside-down and 1 when the torso's z-axis is less than
            `deviation_angle` away from the global z-axis.
        """
        deviation = np.cos(np.deg2rad(deviation_angle))
        # xmat is the 3x3 rotation matrix of the current frame
        upright_torso = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xmat[
            -1, -1
        ]
        upright = reward.tolerance(
            upright_torso,
            bounds=(deviation, np.inf),
            sigmoid="linear",
            margin=1 + deviation,
            value_at_margin=0,
        )
        return np.min(upright)

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
        fall_under_ground = jp.where(z < 0.0, 1.0, 0.0)
        return fall_under_ground
    
    # Termination
    def _named_termination_criterion(name: str):
        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("root_too_far")
    def _root_too_far(self, data, info, max_distance) -> bool:
        target = self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        distance = jp.linalg.norm(target.root_position - root_pos)
        return distance > max_distance

    @_named_termination_criterion("root_too_rotated")
    def _root_too_rotated(self, data, info, max_degrees) -> bool:
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        ang_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        return ang_dist > jp.deg2rad(max_degrees)

    @_named_termination_criterion("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        pose_error = jp.linalg.norm(target.joints - joints)
        return pose_error > max_l2_error
    
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
