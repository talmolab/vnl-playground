from typing import Any, Dict, Optional, Union, Tuple, Mapping, Callable, List
import jax
import jax.flatten_util
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
import brax.math
from vnl_mjx.tasks.rodent.reference_clips import ReferenceClips

from mujoco_playground._src import mjx_env
#from track_mjx.io.load import ReferenceClip

from . import base as rodent_base
from . import consts

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path = consts.RODENT_XML_PATH,
        arena_xml_path = consts.ARENA_XML_PATH,
        sim_dt = 0.002,
        ctrl_dt = 0.01,
        solver = "cg",
        iterations = 4,
        ls_iterations = 4,
        noslip_iterations = 0,
        torque_actuators = False,
        rescale_factor = 0.9,

        mocap_hz = 50,
        clip_length = 250,
        reference_data_path = "../registered_snips.h5",
        clip_set = "all",
        reference_length = 5,
        qvel_init = "zeros",
        reward_terms = {
            "root_pos":      {"exp_scale":  1.0, "weight": 1.0},
            "root_quat":     {"exp_scale": 30.0, "weight": 1.0},
            "joints":        {"exp_scale":  1.0, "weight": 1.0},
            "joints_vel":    {"exp_scale":  1.0, "weight": 1.0},
            "bodies_pos":    {"exp_scale":  1.0, "weight": 1.0},
            "end_eff":       {"exp_scale":  1.0, "weight": 1.0},
        },
        termination_criteria = {
            "root_too_far":     {"max_distance":  0.05}, #Meters
            "root_too_rotated": {"max_degrees":  30.0},  #Degrees
            "pose_error":       {"max_l2_error":  0.5},  #Joint-space L2 distance
        },
    )

_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}

class Imitation(rodent_base.RodentEnv):
    """Multi-clip imitation environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self.add_rodent(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
        )
        self.compile()
        self.reference_clips = ReferenceClips(self._config.reference_data_path,
                                              self._config.clip_length)
        if self._config.clip_set != "all":
            raise NotImplementedError("'all' is the only implemented set of clips.")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        start_rng, clip_rng = jax.random.split(rng)
        info = {
            "reference_clip": jax.random.choice(clip_rng, self._num_clips()),
            "start_frame": jax.random.choice(start_rng, 44),
        }
        reference = self.reference_clips[info["reference_clip"], info["start_frame"]]

        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos = reference.qpos)
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel = jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel = reference.qvel)
        data = mjx.forward(self.mjx_model, data)

        info = self._get_info(data, info)
        obs = self._get_obs(data, info)
        rewards = self._get_rewards(data, info)
        reward = jp.sum(jax.flatten_util.ravel_pytree(rewards)[0])
        done = self._is_done(data, info)
        info["reward_terms"] = rewards
        info["current_frame"] = self._get_cur_frame(data, info)
        metrics = {}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = self._get_info(data, state.info)
        obs = self._get_obs(data, info)
        done = jp.logical_or(self._is_done(data, info), info["truncated"])
        rewards = self._get_rewards(data, info)
        info["reward_terms"] = rewards
        total_reward = jp.sum(jax.flatten_util.ravel_pytree(rewards)[0])
        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=total_reward,
            done=done,
        )

        return state

    def _get_info(self, data: mjx.Data, prev_info: dict[str, Any]):
        truncated = self._get_cur_frame(data, prev_info) >= self._clip_length() - self._config.reference_length
        info = prev_info | dict(truncated=truncated, current_frame = self._get_cur_frame(data, prev_info))
        return info

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "proprioception": self._get_proprioception(data, flatten=False),
            "imitation_target": self._get_imitation_target(data, info),
        }

    def _get_rewards(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, float]:
        rewards = dict()
        for name, kwargs in self._config.reward_terms.items():
            rewards[name] = _REWARD_FCN_REGISTRY[name](self, data, info, **kwargs)
        return rewards

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any]) -> bool:
        termination_reasons = dict()
        for name, kwargs in self._config.termination_criteria.items():
            termination_reasons[name] = _TERMINATION_FCN_REGISTRY[name](self, data, info, **kwargs)
        return jp.any(jax.flatten_util.ravel_pytree(termination_reasons)[0])
    
    def _load_reference_clips(self):
        raise NotImplementedError()

    def _num_clips(self):
        return self.reference_clips.qpos.shape[0]
    
    def _clip_length(self):
        return self.reference_clips.qpos.shape[1]
    
    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        return jp.floor(data.time * self._config.mocap_hz + info["start_frame"]).astype(int)
    
    def _get_current_target(self, data: mjx.Data, info: Mapping[str, Any]) -> ReferenceClips:
        current_frame = self._get_cur_frame(data, info)
        current_clip = info["reference_clip"]
        return self.reference_clips[current_clip, current_frame]

    def _get_imitation_reference(self, data: mjx.Data, info: Mapping[str, Any]) -> ReferenceClips:
        current_frame = self._get_cur_frame(data, info)
        target_slice = slice(current_frame+1, current_frame + self._config.reference_length + 1)
        current_clip = info["reference_clip"]
        return self.reference_clips[current_clip, target_slice]

    def _get_imitation_target(self,
                              data: mjx.Data,
                              info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        reference = self._get_imitation_reference(data, info)

        root_pos  = self.root_body(data).xpos
        root_quat = self.root_body(data).xquat

        root_targets  = jax.vmap(lambda ref_pos: brax.math.rotate(ref_pos - root_pos, root_quat))(reference.root_position)

        quat_targets = jax.vmap(lambda ref_quat: brax.math.relative_quat(ref_quat, root_quat))(reference.root_quaternion)
        
        _assert_all_are_prefix(reference.joint_names, self.get_joint_names(), "reference joints", "model joints")
        joint_targets = reference.joints - self._get_joint_angles(data)

        bodies_pos = self._get_bodies_pos(data, flatten=False)
        body_rel_pos = jp.array([reference.body_xpos(name) - bodies_pos[name] for name in bodies_pos])
        to_egocentric = jax.vmap(lambda diff_vec: brax.math.rotate(diff_vec, root_quat))
        body_targets = jax.vmap(to_egocentric)(body_rel_pos)
        return {
            "root": root_targets,
            "quat": quat_targets,
            "joint": joint_targets,
            "body": body_targets,
        }

    # Rewards
    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn
        return decorator
    
    @_named_reward("root_pos")
    def _root_pos_reward(self, data, info, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        distance = jp.linalg.norm(target.root_position - root_pos)
        reward = weight * jp.exp(-(distance/exp_scale)**2)
        return reward
    
    @_named_reward("root_quat")
    def _root_quat_reward(self, data, info, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_dist = 2.0*jp.dot(root_quat, target.root_quaternion)**2 - 1.0
        ang_dist = 0.5*jp.arccos(jp.minimum(1.0, quat_dist))
        exp_scale = jp.deg2rad(exp_scale)
        #brax.math.quat_diff(root_quat, target.quaternion) #TODO: doesn't exist, figure out easiest replacement
        reward = weight * jp.exp(-(ang_dist/exp_scale)**2)
        return reward
    
    @_named_reward("joints")
    def _joints_reward(self, data, info, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(target.joints - joints)
        reward = weight * jp.exp(-(distance/exp_scale)**2)
        return reward
    
    @_named_reward("joints_vel")
    def _joint_vels_reward(self, data, info, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joint_vels  = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(target.joints_velocity - joint_vels)
        reward = weight * jp.exp(-(distance/exp_scale)**2)
        return reward
    
    @_named_reward("bodies_pos")
    def _body_pos_reward(self, data, info, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        body_pos  = self._get_bodies_pos(data, flatten=False)
        dists = jp.array([bp - target.body_xpos(k) for k, bp in body_pos.items()])
        distance_sqr = jp.sum(dists**2)
        reward = weight * jp.exp(-distance_sqr/(exp_scale**2))
        return reward
    
    @_named_reward("end_eff")
    def _end_eff_reward(self, data, info, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        body_pos  = self._get_bodies_pos(data, flatten=False)
        dists = jp.array([body_pos[ef] - target.body_xpos(ef) for ef in consts.END_EFFECTORS])
        distance_sqr = jp.sum(dists**2)
        reward = weight * jp.exp(-distance_sqr/(exp_scale**2))
        return reward
    
    @_named_reward("torso_z_range")
    def _torso_z_range_reward(self, data, info, weight, healthy_z_range) -> float:
        torso_z = self.root_body(data).xpos[2]
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        return weight * in_range.astype(float)

    @_named_reward("control_cost")
    def _control_cost(self, data, info, weight) -> float:
        return weight * jp.sum(jp.square(info["action"]))
    
    @_named_reward("control_diff_cost")
    def _control_diff_cost(self, data, info, weight) -> float:
        return weight * jp.sum(jp.square(info["action"] - info["prev_action"]))
    
    @_named_reward("energy_cost")
    def _energy_cost(self, data, info, weight, max_value) -> float:
        return weight * jp.minimum(jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator)), max_value)
    
    @_named_reward("jerk_cost")
    def _jerk_cost(self, data, info, weight, window_len) -> float:
        raise NotImplementedError("jerk_cost is not implemented")
    
    # Termination
    def _named_termination_criterion(name: str):
        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn
        return decorator
    
    @_named_termination_criterion("root_too_far")
    def _root_too_far(self, data, info, max_distance) -> bool:
        target = self._get_current_target(data, info)
        root_pos  = self.root_body(data).xpos
        distance = jp.linalg.norm(target.root_position - root_pos)
        return distance > max_distance
    
    @_named_termination_criterion("root_too_rotated")
    def _root_too_rotated(self, data, info, max_degrees) -> bool:
        target = self._get_current_target(data, info)
        root_quat  = self.root_body(data).xquat
        quat_dist = 2.0*jp.dot(root_quat, target.root_quaternion)**2 - 1.0
        ang_dist = 0.5*jp.arccos(jp.minimum(1.0, quat_dist))
        return ang_dist > jp.deg2rad(max_degrees)

    @_named_termination_criterion("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        joints  = self._get_joint_angles(data)
        pose_error = jp.linalg.norm(target.joints - joints)
        return pose_error > max_l2_error

def _assert_all_are_prefix(a, b, a_name="a", b_name="b"):
    if isinstance(a, map):
        a = list(a)
    if isinstance(b, map):
        b = list(b)
    if len(a) != len(b):
        raise AssertionError(f"{a_name} has length {len(a)}, but {b_name} has length {len(b)}.")
    for a_el, b_el in zip(a, b):
        if not b_el.startswith(a_el):
            raise AssertionError(f"Comparing {a_name} and {b_name}. Expected {a_el} to match {b_el}.")
