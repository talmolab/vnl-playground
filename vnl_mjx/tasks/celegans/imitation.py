import collections
from typing import Any, Dict, Optional, Union, Tuple, Mapping, Callable, List
import jax
import jax.flatten_util
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import brax.math
from vnl_mjx.tasks.celegans.reference_clips import ReferenceClips

from mujoco_playground._src import mjx_env

from . import base as worm_base
from . import consts

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path = consts.CELEGANS_XML_PATH,
        arena_xml_path = consts.ARENA_XML_PATH,
        mujoco_impl = "mjx",
        sim_dt  = 0.002,
        ctrl_dt = 0.02,
        solver = "cg",
        iterations = 5,
        ls_iterations = 5,
        noslip_iterations = 0,
        torque_actuators = False,
        rescale_factor = 1.0,

        mocap_hz = 20,
        clip_length = 250,
        reference_data_path = "../registered_snips.h5",
        clip_set = "all",
        reference_length = 5,
        start_frame_range = [0, 44],
        qvel_init = "zeros",
        with_ghost = False,
        reward_terms = {
            # Imitation rewards
            "root_pos":      {"exp_scale":  0.05,  "weight": 1.0}, #Meters
            "root_quat":     {"exp_scale": 0.0625,   "weight": 1.0}, #Degrees
            "joints":        {"exp_scale":  16,   "weight": 1.0}, #Joint-space L2 distance
            "joints_vel":    {"exp_scale":  0.02,   "weight": 1.0}, #Joint velocity-space L2 distance
            "bodies_pos":    {"exp_scale":  0.125,  "weight": 1.0}, #Distance in concatenated euclidean space
            "end_eff":       {"exp_scale":  0.002, "weight": 1.0}, #Distance in concatenated euclidean space
            "upright":      {"healthy_z_range": (0.0, 1.0), "weight": 0.0},
            
        },
        # Costs / regularizers
        cost_terms = {
            "control":       {"weight": 0.02},
            "control_diff":  {"weight": 0.02},
            "energy":        {"max_value":  50.0, "weight": 0.0},
        },
        termination_criteria = {
            "fall":             {"healthy_z_range": (0.0, 1.0)},
            "root_too_far":     {"max_distance": 0.01},  #Meters
            "root_too_rotated": {"max_degrees":  60.0}, #Degrees
            "pose_error":       {"max_l2_error": 4.5},  #Joint-space L2 distance
        },
    )

_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}
_COST_FCN_REGISTRY: dict[str, Callable] = {}

class Imitation(worm_base.CelegansEnv):
    """Multi-clip imitation environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self.add_worm(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
        )
        if self._config.with_ghost:
            self.add_ghost_worm(rescale_factor=self._config.rescale_factor)

        self.compile()
        self.reference_clips = ReferenceClips(self._config.reference_data_path,
                                              self._config.clip_length)
        if self._config.clip_set != "all":
            raise NotImplementedError("'all' is the only implemented set of clips.")
        self._steps_per_curr_frame = 1/(self._config.mocap_hz * self._config.ctrl_dt)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        start_rng, clip_rng = jax.random.split(rng)
        clip_idx = jax.random.choice(clip_rng, self._num_clips())
        start_frame = jax.random.randint(start_rng, (), *self._config.start_frame_range)
        data = self._reset_data(clip_idx, start_frame)
        info : dict[str, Any] = {
            "start_frame": start_frame,
            "reference_clip": clip_idx,
            "prev_ctrl": jp.zeros((self.action_size,))
        }

        last_valid_frame = self._clip_length() - self._config.reference_length - 1
        info["truncated"] = jp.astype(self._get_cur_frame(data, info) > last_valid_frame, float)
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()


        obs = self._get_obs(data, info)
        # Used to initialize our intention network
        info["reference_obs_size"] = jax.flatten_util.ravel_pytree(obs["imitation_target"])[0].shape[-1] #TODO: use getter functions
        info["proprioceptive_obs_size"] = jax.flatten_util.ravel_pytree(obs["proprioception"])[0].shape[-1] #TODO: use getter functions

        obs = jax.flatten_util.ravel_pytree(obs)[0] #TODO: Use wrapper instead.
        
        rewards, dists = self._get_rewards(data, info)
        reward = jp.sum(jax.flatten_util.ravel_pytree(rewards)[0])
        costs, magnitudes = self._get_costs(data, info)
        cost = jp.sum(jax.flatten_util.ravel_pytree(costs)[0])
        total_reward = reward - cost

        termination_conditions = self._get_termination_conditions(data, info)
        done = jp.any(jax.flatten_util.ravel_pytree(termination_conditions)[0])

        metrics = {
            **rewards,
            **costs,
            **{n: jp.astype(t, float) for n, t in termination_conditions.items()},
            **dists,
            **magnitudes,
            "current_frame": jp.astype(self._get_cur_frame(data, info), float),
        }
        return mjx_env.State(data, obs, total_reward, jp.astype(done, float), metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        last_valid_frame = self._clip_length() - self._config.reference_length - 1
        info["truncated"] = jp.astype(self._get_cur_frame(data, info) > last_valid_frame, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        obs = jax.flatten_util.ravel_pytree(obs)[0]

        termination_conditions = self._get_termination_conditions(data, info)
        terminated = jp.any(jax.flatten_util.ravel_pytree(termination_conditions)[0])
        done = jp.logical_or(terminated, info["truncated"])
        
        rewards, dists = self._get_rewards(data, info)
        costs, magnitudes = self._get_costs(data, info)
        
        total_reward = jp.sum(jax.flatten_util.ravel_pytree(rewards)[0]) - jp.sum(jax.flatten_util.ravel_pytree(costs)[0])
        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=total_reward,
            done=jp.astype(done, float),
        )

        metrics = {
            **rewards,
            **costs,
            **{n: jp.astype(t, float) for n, t in termination_conditions.items()},
            **dists,
            **magnitudes,
            "current_frame": jp.astype(self._get_cur_frame(data, info), float),
        }
        state.metrics.update(metrics)

        return state

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        return collections.OrderedDict(
            proprioception=self._get_proprioception(data, flatten=False),
            imitation_target=self._get_imitation_target(data, info),
        )

    def _get_rewards(self, data: mjx.Data, info: Mapping[str, Any]) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        rewards, dists = dict(), dict()
        for name, kwargs in self._config.reward_terms.items():
            r, d = _REWARD_FCN_REGISTRY[name](self, data, info, **kwargs)
            rewards[name] = r
            dists[f"{name}_dist"] = d
        return rewards, dists

    def _get_termination_conditions(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, bool]:
        termination_reasons = dict()
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            termination_reasons[name] = termination_fcn(self, data, info, **kwargs)
        return termination_reasons

    def _get_costs(self, data: mjx.Data, info: Mapping[str, Any]) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        costs, magnitudes = dict(), dict()
        for name, kwargs in self._config.cost_terms.items():
            c, m = _COST_FCN_REGISTRY[name](self, data, info, **kwargs)
            costs[f"{name}_cost"] = c
            magnitudes[name] = m
        return costs, magnitudes
    
    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        data = mjx.make_data(self.mjx_model)#, impl=self._config.mujoco_impl)
        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)
        data = data.replace(qpos = reference.qpos)
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel = jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "noise":
            raise NotImplementedError("qvel_init='noise' is not yet implemented.")
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel = reference.qvel)
        data = mjx.forward(self.mjx_model, data)
        return data
    
    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    def _num_clips(self):
        return self.reference_clips.qpos.shape[0]
    
    def _clip_length(self):
        return self.reference_clips.qpos.shape[1]
    
    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        return jp.floor(data.time * self._config.mocap_hz + info["start_frame"]).astype(int)
    
    def _get_current_target(self, data: mjx.Data, info: Mapping[str, Any]) -> ReferenceClips:
        return self.reference_clips.at(clip = info["reference_clip"],
                                       frame = self._get_cur_frame(data, info))
    def _get_reference_clip(self, data: mjx.Data, info: Mapping[str, Any]) -> ReferenceClips:
        return self.reference_clips.slice(clip = info["reference_clip"],
                                          start_frame = 0,
                                          length = self._config.clip_length)
    
    def _get_imitation_reference(self, data: mjx.Data, info: Mapping[str, Any]) -> ReferenceClips:
        return self.reference_clips.slice(clip = info["reference_clip"],
                                          start_frame = self._get_cur_frame(data, info)+1,
                                          length = self._config.reference_length)

    def _get_imitation_target(self,
                              data: mjx.Data,
                              info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        reference = self._get_imitation_reference(data, info)

        root_pos  = self.root_body(data).xpos
        root_quat = self.root_body(data).xquat
        root_targets = jax.vmap(lambda ref_pos: brax.math.rotate(ref_pos - root_pos, root_quat))(reference.root_position)
        quat_targets = jax.vmap(lambda ref_quat: brax.math.relative_quat(ref_quat, root_quat))(reference.root_quaternion)
        
        _assert_all_are_prefix(reference.joint_names, self.get_joint_names(), "reference joints", "model joints")
        joint_targets = reference.joints - self._get_joint_angles(data)

        bodies_pos = self._get_bodies_pos(data, flatten=False)
        body_rel_pos = jp.array([reference.body_xpos(name) - bodies_pos[name] for name in bodies_pos])
        to_egocentric = jax.vmap(lambda diff_vec: brax.math.rotate(diff_vec, root_quat))
        body_targets = jax.vmap(to_egocentric)(body_rel_pos)

        return collections.OrderedDict(
            root = root_targets,
            quat = quat_targets,
            joint = joint_targets,
            body = body_targets,
        )

    # Rewards
    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn
        return decorator
    
    @_named_reward("root_pos")
    def _root_pos_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        target = self._get_current_target(data, info)
        root_pos = self._get_root_pos(data)
        distance = jp.linalg.norm(target.root_position - root_pos)
        reward = weight * jp.exp(-((distance/exp_scale)**2)/2)
        return reward, distance
    
    @_named_reward("root_quat")
    def _root_quat_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        target = self._get_current_target(data, info)
        root_quat = self._get_root_quat(data)
        quat_dist = 2.0*jp.dot(root_quat, target.root_quaternion)**2 - 1.0
        ang_dist = 0.5*jp.arccos(jp.minimum(1.0, quat_dist))
        exp_scale = jp.deg2rad(exp_scale)
        reward = weight * jp.exp(-((ang_dist/exp_scale)**2)/2)
        return reward, ang_dist
    
    @_named_reward("joints")
    def _joints_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(target.joints - joints)
        reward = weight * jp.exp(-((distance/exp_scale)**2)/2)
        return reward, distance
    
    @_named_reward("joints_vel")
    def _joint_vels_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        target = self._get_current_target(data, info)
        joint_vels  = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(target.joints_velocity - joint_vels)
        reward = weight * jp.exp(-((distance/exp_scale)**2)/2)
        return reward, distance
    
    @_named_reward("bodies_pos")
    def _body_pos_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        target = self._get_current_target(data, info)
        body_pos  = self._get_bodies_pos(data, flatten=False)
        dists = jp.array([bp - target.body_xpos(k) for k, bp in body_pos.items()])
        distance_sqr = jp.sum(dists**2)
        reward = weight * jp.exp(-(distance_sqr/(exp_scale)**2)/2)
        return reward, distance_sqr
    
    @_named_reward("end_eff")
    def _end_eff_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        target = self._get_current_target(data, info)
        body_pos  = self._get_bodies_pos(data, flatten=False)
        dists = jp.array([body_pos[ef] - target.body_xpos(ef) for ef in consts.END_EFFECTORS])
        distance_sqr = jp.sum(dists**2)
        reward = weight * jp.exp(-(distance_sqr/(exp_scale)**2)/2)
        return reward, distance_sqr
    
    @_named_reward("upright")
    def _upright_reward(self, data, info, weight, healthy_z_range) -> Tuple[float, float]:
        torso_z = self._get_body_height(data)#root_body(data).xpos[2]
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        return weight * in_range.astype(float), torso_z

    # Costs
    def _named_cost(name: str):
        def decorator(cost_fcn: Callable):
            _COST_FCN_REGISTRY[name] = cost_fcn
            return cost_fcn
        return decorator

    @_named_cost("control")
    def _control_cost(self, data, info, weight) -> Tuple[float, float]:
        ctrl_magnitude = jp.sum(jp.square(info["action"]))
        return weight * ctrl_magnitude, ctrl_magnitude
    
    @_named_cost("control_diff")
    def _control_diff_cost(self, data, info, weight) -> Tuple[float, float]:
        ctrl_diff = jp.sum(jp.square(info["action"] - info["prev_action"]))
        return weight * ctrl_diff, ctrl_diff
    
    @_named_cost("energy")
    def _energy_cost(self, data, info, weight, max_value) -> Tuple[float, float]:
        energy = jp.minimum(jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator)), max_value)
        return weight * energy, energy
    
    @_named_cost("jerk")
    def _jerk_cost(self, data, info, weight, window_len) -> Tuple[float, float]:
        raise NotImplementedError("jerk_cost is not implemented")
    
    # Termination
    def _named_termination_criterion(name: str):
        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn
        return decorator

    @_named_termination_criterion("fall")
    def _fall(self, data, info, healthy_z_range) -> float:
        torso_z = self._get_body_height(data)#root_body(data).xpos[2]
        min_z, max_z = healthy_z_range
        return jp.logical_or(torso_z < min_z, torso_z > max_z)

    @_named_termination_criterion("root_too_far")
    def _root_too_far(self, data, info, max_distance) -> bool:
        target = self._get_current_target(data, info)
        root_pos  = self._get_root_pos(data)
        distance = jp.linalg.norm(target.root_position - root_pos)
        return distance > max_distance
    
    @_named_termination_criterion("root_too_rotated")
    def _root_too_rotated(self, data, info, max_degrees) -> bool:
        target = self._get_current_target(data, info)
        root_quat = self._get_root_quat(data)
        quat_dist = 2.0*jp.dot(root_quat, target.root_quaternion)**2 - 1.0
        ang_dist = 0.5*jp.arccos(jp.minimum(1.0, quat_dist))
        return ang_dist > jp.deg2rad(max_degrees)

    @_named_termination_criterion("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
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

class Imitation2D(Imitation):
    """Imitation environment for 2D C. Elegans."""

    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
    
    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        data = mjx.make_data(self.mjx_model)#, impl=self._config.mujoco_impl)
        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)

        qpos2d = jnp.concat([reference.qpos[..., :3], reference.qpos[7:]], axis=-1)
        data = data.replace(qpos=qpos2d)
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel = jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "noise":
            raise NotImplementedError("qvel_init='noise' is not yet implemented.")
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel = jp.concat([reference.qvel[..., :3], reference.qvel[..., 6:]], axis=-1))
        data = mjx.forward(self.mjx_model, data)
        return data