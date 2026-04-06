"""Version 2 of modular imitation task for the rodent.

"""

import collections
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, override

import brax.math
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import get_sensor_data

from . import base as modular_base
from . import consts
from vnl_playground.tasks.reference_clips import ReferenceClips


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.MODULAR_RODENT_WALKER_XML_PATH,
        arena_xml_path=consts.MODULAR_RODENT_ARENA_XML_PATH,
        rescale_factor=0.9,
        torque_actuators=True,
        mujoco_impl="warp",
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="newton",
        iterations=25,
        ls_iterations=25,
        naconmax=128*512,
        njmax=1024,
        noslip_iterations=0,
        tolerance=1e-8,
        cone="elliptic",
        impratio=10.0,
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=50,
        clip_length=250,
        clip_set="all",
        start_frame_range=[0,0],
        qvel_init="zeros",
        keep_clips_idx=None,
        include_vel=True,
        # Termination
        max_target_distance=0.05, # meters
        min_torso_z=0.03,         # meters
        # Reward scales
        reward_terms={
            "limb_pos_exp_scale": 0.02,  # sigma for distance-based errors (meters)
            "joint_exp_scale": 0.2,      # sigma for angle-based errors (radians)
            "tendon_exp_scale": 0.5,     # sigma for angle-based errors for tendons (radians)
            "root_pos_scale": 0.05,      # sigma for root position reward (meters)
            "root_pos_weight": 5.0,      # weight for root position reward
            "root_orientation_weight": 5.0,      # weight for root angle reward
        },
        energy_cost = -0.04,
        force_cost = 0.0,
    )


class ModularImitation_v4(modular_base.ModularRodentEnv):
    """Modular multi-clip imitation environment for the rodent.

    This is an alternate version with egocentric error signals and simplified
    observations.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
        clips: Optional[ReferenceClips] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self.compile()

        if clips is not None:
            self.reference_clips = clips
        else:
            self.reference_clips = ReferenceClips(
                self._config.reference_data_path,
                self._config.clip_length,
                self._config.keep_clips_idx,
            )

        max_n_clips = self.reference_clips.qpos.shape[0]
        if self._config.clip_set == "all":
            self._clip_set = max_n_clips
        elif isinstance(self._config.clip_set, (list, tuple, jp.ndarray, np.ndarray)):
            self._clip_set = jp.array(self._config.clip_set)
        elif self._config.clip_set in self.reference_clips.clip_names:
            (self._clip_set,) = jp.where(
                self._config.clip_set == self.reference_clips.clip_names
            )
        else:
            raise ValueError(
                "config.clip_set must be 'all', a list of clip indices"
                f" or a behavior name. Got {self._config.clip_set}."
            )

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        start_rng, clip_rng = jax.random.split(rng)
        if clip_idx is None:
            clip_idx = jax.random.choice(clip_rng, self._clip_set)
        if start_frame is None:
            start_frame = jax.random.randint(
                start_rng, (), *self._config.start_frame_range
            )

        data = self._reset_data(clip_idx, start_frame)
        info: dict[str, Any] = {
            "start_frame": start_frame,
            "reference_clip": clip_idx,
        }
        last_valid_frame = self._clip_length() - 2  # -2 for interpolation headroom
        info["truncated"] = jp.astype(self._get_cur_frame(data, info) > last_valid_frame, float)
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()

        metrics = {"current_frame": self._get_cur_frame(data, info)}
        obs, reward, done = self._process(data, info, metrics)
        return mjx_env.State(data, obs, reward, done.astype(float), metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: dict[str, jax.Array],
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        ctrl = self._action_to_ctrl(action)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, n_steps)

        info = state.info
        last_valid_frame = self._clip_length() - 2
        info["truncated"] = jp.astype(self._get_cur_frame(data, info) > last_valid_frame, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        metrics = state.metrics
        obs, reward, terminated = self._process(data, info, metrics)
        reward = jax.tree.map(jp.nan_to_num, reward)
        done = jp.logical_or(terminated, info["truncated"])
        metrics["current_frame"] = self._get_cur_frame(data, info)
        return state.replace(data=data, obs=obs, info=info, reward=reward, done=done.astype(float))

    def _process(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: dict
    ) -> tuple:
        """Compute obs/reward/terminated from current state; updates metrics in-place.

        Computes proprioception and targets once and passes them to all sub-methods,
        avoiding redundant work.

        Returns:
            (obs, reward, terminated)
        """
        prop = self._get_modular_proprioception(data)
        cur = self._interpolated_target(data, info, time_ahead=0.0)
        fut = self._interpolated_target(data, info, time_ahead=0.1)
        obs = self._get_obs(prop, cur, fut)
        reward = self._get_reward(data, prop, cur, metrics)
        terminated = self._is_done(data, info, cur, metrics)
        self._add_extra_metrics(prop, cur, metrics)
        return obs, reward, terminated

    @override
    def _get_modular_proprioception(
        self, data: mjx.Data
    ) -> dict[str, dict[str, jp.ndarray]]:
        """Hierarchical proprioception dict organized by body module.
        Simplified compared to base.
        """
        m = self.mj_model
        torso_xmat = data.bind(self.mjx_model, self._spec.body("torso")).xmat
        torso_xpos = data.bind(self.mjx_model, self._spec.body("torso")).xpos

        def joint(name: str) -> jp.ndarray:
            jnt = data.bind(self.mjx_model, self._spec.joint(name))
            if self._config.include_vel:
                return jp.array([jnt.qpos, jnt.qvel])
            else:
                return jp.array([jnt.qpos])
        
        def body_world_pos(name: str) -> jp.array:
            return data.bind(self.mjx_model, self._spec.body(name)).xpos
        def body_pos(name: str) -> jp.array:
            bdy = data.bind(self.mjx_model, self._spec.body(name))
            pos = torso_xmat.T @ (bdy.xpos - torso_xpos)
            if self._config.include_vel:
                return jp.concat((pos, torso_xmat.T @ bdy.cvel[3:6]))
            else:
                return pos
        def world_xaxis(name: str) -> jp.ndarray:
            return data.bind(self.mjx_model, self._spec.body(name)).xmat[:, 0]
        def world_zaxis(name: str) -> jp.ndarray:
            return data.bind(self.mjx_model, self._spec.body(name)).xmat[:, 2]
        def xaxis(name: str) -> jp.ndarray:
            return torso_xmat.T @ world_xaxis(name)
        def zaxis(name: str) -> jp.ndarray:
            return torso_xmat.T @ world_zaxis(name)
        def height(body_name: str) -> jp.ndarray:
            return body_world_pos(body_name)[2:3]

        return {
            "hand_L": {
                "palm_contact": get_sensor_data(m, data, "hand_L/palm_contact"),
                "hand_pos": body_pos("hand_L"),
                "wrist_angle": joint("wrist_L"),
                "finger_angle": joint("finger_L"),
                "xaxis": xaxis("hand_L"),
                "finger_zz": world_zaxis("finger_L")[2:3],
            },
            "arm_L": {
                "elbow_pos": body_pos("lower_arm_L"),
                "elbow_angle": joint("elbow_L"),
                "shoulder_sup": joint("shoulder_sup_L"),
                "shoulder": joint("shoulder_L"),
                "scapula_extend": joint("scapula_L_extend"),
                "scapula_abduct": joint("scapula_L_abduct"),
                "scapula_supinate": joint("scapula_L_supinate"),
                "elbow_height": body_world_pos("lower_arm_L")[2:3],
                "shoulder_height": body_world_pos("upper_arm_L")[2:3],
            },
            "hand_R": {
                "palm_contact": get_sensor_data(m, data, "hand_R/palm_contact"),
                "hand_pos": body_pos("hand_R"),
                "wrist_angle": joint("wrist_R"),
                "finger_angle": joint("finger_R"),
                "xaxis": xaxis("hand_R"),
                "finger_zz": world_zaxis("finger_R")[2:3],
            },
            "arm_R": {
                "elbow_pos": body_pos("lower_arm_R"),
                "elbow_angle": joint("elbow_R"),
                "shoulder_sup": joint("shoulder_sup_R"),
                "shoulder": joint("shoulder_R"),
                "scapula_extend": joint("scapula_R_extend"),
                "scapula_abduct": joint("scapula_R_abduct"),
                "scapula_supinate": joint("scapula_R_supinate"),
                "elbow_height": body_world_pos("lower_arm_R")[2:3],
                "shoulder_height": body_world_pos("upper_arm_R")[2:3],
            },
            "foot_L": {
                "sole_contact": get_sensor_data(m, data, "foot_L/sole_contact"),
                "heel_contact": get_sensor_data(m, data, "foot_L/heel_contact"),
                "foot_pos": body_pos("foot_L"),
                "knee_angle": joint("knee_L"),
                "ankle_angle": joint("ankle_L"),
                "toe_angle": joint("toe_L"),
                "xaxis": xaxis("foot_L"),
                "toe_zz": world_zaxis("toe_L")[2:3],
            },
            "leg_L": {
                "knee_pos": body_pos("lower_leg_L"),
                "hip_supinate": joint("hip_L_supinate"),
                "hip_abduct": joint("hip_L_abduct"),
                "hip_extend": joint("hip_L_extend"),
                "knee_angle": joint("knee_L"),
                "ankle_angle": joint("ankle_L"),
                "pelvis_zaxis": zaxis("pelvis"),
                "hip_height": jp.expand_dims(self.site_z(data, "hip_L"), axis=0),
                #"hip_accelerometer": get_sensor_data(m, data, "leg_L/hip_accelerometer"),
                #"hip_gyro": get_sensor_data(m, data, "leg_L/hip_gyro"),
            },
            "foot_R": {
                "sole_contact": get_sensor_data(m, data, "foot_R/sole_contact"),
                "heel_contact": get_sensor_data(m, data, "foot_R/heel_contact"),
                "foot_pos": body_pos("foot_R"),
                "knee_angle": joint("knee_R"),
                "ankle_angle": joint("ankle_R"),
                "toe_angle": joint("toe_R"),
                "xaxis": xaxis("foot_R"),
                "toe_zz": world_zaxis("toe_R")[2:3],
            },
            "leg_R": {
                "knee_pos": body_pos("lower_leg_R"),
                "hip_supinate": joint("hip_R_supinate"),
                "hip_abduct": joint("hip_R_abduct"),
                "hip_extend": joint("hip_R_extend"),
                "knee_angle": joint("knee_R"),
                "ankle_angle": joint("ankle_R"),
                "pelvis_zaxis": zaxis("pelvis"),
                "hip_height": jp.expand_dims(self.site_z(data, "hip_R"), axis=0),
            },
            "torso": {
                "accelerometer": get_sensor_data(m, data, "torso/accelerometer"),
                "gyro": get_sensor_data(m, data, "torso/gyro"),
                #"zaxis": yaw_mat @ body_zaxis_world(torso_id),
                "lumbar_bend":   0.658 * joint("vertebra_2_bend")   + 0.342 * joint("vertebra_5_bend"),
                "lumbar_twist":  0.571 * joint("vertebra_3_twist")  + 0.429 * joint("vertebra_6_twist"),
                "lumbar_extend": 0.605 * joint("vertebra_1_extend") + 0.395 * joint("vertebra_4_extend"),
                "linvel": torso_xmat.T @ get_sensor_data(m, data, "torso/linvel"), #TODO: check
                "height_above_ground": torso_xpos[2:3],
                "pelvis_zaxis": zaxis("pelvis"),
                "pelvis_xaxis": xaxis("pelvis"),
            },
            "head": {
                "accelerometer": get_sensor_data(m, data, "head/accelerometer"),
                "pos": body_pos("skull"),
                "xaxis": xaxis("skull"),
                "zaxis": zaxis("skull"),
                "atlas": joint("atlas"),
                "atlant_extend": joint("vertebra_atlant_extend"),
                "mandible": joint("mandible"),
                #"cervical_extend": tendon("cervical_extend"),
                #"cervical_bend": tendon("cervical_bend"),
                #"cervical_twist": tendon("cervical_twist"),
            },
        }

    def _get_obs(
        self, prop: dict, cur: dict, fut: dict
    ) -> collections.OrderedDict:
        """Hierarchical observation dict organized by body module.

        Each module contains {proprioception, current_target, future_target}.
        Root is included with only current_target and future_target (no
        proprioception, matching Julia's dummy root proprioception).
        """
        obs = collections.OrderedDict()
        for module in consts.MODULES:
            obs[module] = collections.OrderedDict(
                proprioception=prop[module],
                current_target=cur[module],
                future_target=fut[module],
            )
        obs["root"] = collections.OrderedDict(
            current_target=cur["root"],
            future_target=fut["root"],
        )
        return obs

    def _get_reward(self, data, prop: dict, cur: dict, metrics: dict) -> float:
        """Compute per-module rewards, log to metrics, return scalar total."""
        target = cur
        pos_scale = self._config.reward_terms.limb_pos_exp_scale
        joint_scale = self._config.reward_terms.joint_exp_scale
        tendon_scale = self._config.reward_terms.tendon_exp_scale
        root_pos_scale = self._config.reward_terms.root_pos_scale
        root_pos_weight = self._config.reward_terms.root_pos_weight
        root_orientation_weight = self._config.reward_terms.root_orientation_weight

        rewards = {
            "hand_L": {
                "finger_level": prop["hand_L"]["finger_zz"][0],
                "wrist_joint": _reward_shape(
                    prop["hand_L"]["wrist_angle"][0:1],
                    target["hand_L"]["wrist_angle"],
                    joint_scale,
                ),
                "orientation": jp.dot(
                    prop["hand_L"]["xaxis"], target["hand_L"]["xaxis"]
                ),
                "hand_pos": _reward_shape(
                    prop["hand_L"]["hand_pos"][0:3],
                    target["hand_L"]["pos"],
                    pos_scale,
                ),
            },
            "arm_L": {
                "elbow_joint": _reward_shape(
                    prop["arm_L"]["elbow_angle"][0:1],
                    target["arm_L"]["elbow_angle"],
                    joint_scale,
                ),
                "elbow_pos": _reward_shape(
                    prop["arm_L"]["elbow_pos"][0:3],
                    target["arm_L"]["elbow_pos"],
                    pos_scale,
                ),
                "shoulder_height": _reward_shape(
                    prop["arm_L"]["shoulder_height"],
                    target["arm_L"]["shoulder_height"],
                    pos_scale,
                ),
            },
            "hand_R": {
                "finger_level": prop["hand_R"]["finger_zz"][0],
                "wrist_joint": _reward_shape(
                    prop["hand_R"]["wrist_angle"][0:1],
                    target["hand_R"]["wrist_angle"],
                    joint_scale,
                ),
                "orientation": jp.dot(
                    prop["hand_R"]["xaxis"], target["hand_R"]["xaxis"]
                ),
                "hand_pos": _reward_shape(
                    prop["hand_R"]["hand_pos"][0:3],
                    target["hand_R"]["pos"],
                    pos_scale,
                ),
            },
            "arm_R": {
                "elbow_joint": _reward_shape(
                    prop["arm_R"]["elbow_angle"][0:1],
                    target["arm_R"]["elbow_angle"],
                    joint_scale,
                ),
                "elbow_pos": _reward_shape(
                    prop["arm_R"]["elbow_pos"][0:3],
                    target["arm_R"]["elbow_pos"],
                    pos_scale,
                ),
                "shoulder_height": _reward_shape(
                    prop["arm_R"]["shoulder_height"],
                    target["arm_R"]["shoulder_height"],
                    pos_scale,
                ),
            },
            "foot_L": {
                "toe_level": prop["foot_L"]["toe_zz"][0],
                "ankle_joint": _reward_shape(
                    prop["foot_L"]["ankle_angle"][0:1],
                    target["foot_L"]["ankle_angle"],
                    joint_scale,
                ),
                "orientation": jp.dot(
                    prop["foot_L"]["xaxis"], target["foot_L"]["xaxis"]
                ),
                "foot_pos": _reward_shape(
                    prop["foot_L"]["foot_pos"][0:3],
                    target["foot_L"]["pos"],
                    pos_scale,
                ),
            },
            "leg_L": {
                "knee_joint": _reward_shape(
                    prop["leg_L"]["knee_angle"][0:1],
                    target["leg_L"]["knee_angle"],
                    joint_scale,
                ),
                "knee_pos": _reward_shape(
                    prop["leg_L"]["knee_pos"][0:3],
                    target["leg_L"]["knee_pos"],
                    pos_scale,
                ),
                #"orientation": jp.dot(
                #    prop["foot_L"]["xaxis"], target["foot_L"]["xaxis"]
                #),
                "hip_height": _reward_shape(
                    prop["leg_L"]["hip_height"],
                    target["leg_L"]["hip_height"],
                    pos_scale,
                ),
            },
            "foot_R": {
                "toe_level": prop["foot_R"]["toe_zz"][0],
                "ankle_joint": _reward_shape(
                    prop["foot_R"]["ankle_angle"][0:1],
                    target["foot_R"]["ankle_angle"],
                    joint_scale,
                ),
                "orientation": jp.dot(
                    prop["foot_R"]["xaxis"], target["foot_R"]["xaxis"]
                ),
                "foot_pos": _reward_shape(
                    prop["foot_R"]["foot_pos"][0:3],
                    target["foot_R"]["pos"],
                    pos_scale,
                ),
            },
            "leg_R": {
                "knee_joint": _reward_shape(
                    prop["leg_R"]["knee_angle"][0:1],
                    target["leg_R"]["knee_angle"],
                    joint_scale,
                ),
                "knee_pos": _reward_shape(
                    prop["leg_R"]["knee_pos"][0:3],
                    target["leg_R"]["knee_pos"],
                    pos_scale,
                ),
                #"orientation": jp.dot(
                #    prop["foot_R"]["xaxis"], target["foot_R"]["xaxis"]
                #),
                "hip_height": _reward_shape(
                    prop["leg_R"]["hip_height"],
                    target["leg_R"]["hip_height"],
                    pos_scale,
                ),
            },
            "torso": {
                "pelvis_x": jp.dot(
                    prop["torso"]["pelvis_xaxis"], target["torso"]["pelvis_xaxis"]
                ),
                "pelvis_z": jp.dot(
                    prop["torso"]["pelvis_zaxis"], target["torso"]["pelvis_zaxis"]
                ),
                "height_above_ground": _reward_shape(
                    prop["torso"]["height_above_ground"],
                    target["torso"]["height_above_ground"],
                    pos_scale,
                ),
                "lumbar_bend": _reward_shape(
                    prop["torso"]["lumbar_bend"][0:1],
                    target["torso"]["lumbar_bend"],
                    tendon_scale,
                ),
                "lumbar_twist": _reward_shape(
                    prop["torso"]["lumbar_twist"][0:1],
                    target["torso"]["lumbar_twist"],
                    tendon_scale,
                ),
                "lumbar_extend": _reward_shape(
                    prop["torso"]["lumbar_extend"][0:1],
                    target["torso"]["lumbar_extend"],
                    tendon_scale,    
                ),
            },
            "head": {
                "orientation_x": jp.dot(
                    prop["head"]["xaxis"], target["head"]["xaxis"]
                ),
                "orientation_z": jp.dot(
                    prop["head"]["zaxis"], target["head"]["zaxis"]
                ),
                "head_pos": _reward_shape(
                    prop["head"]["pos"][0:3],
                    target["head"]["pos"],
                    pos_scale,
                ),
                "mandible": _reward_shape(
                    prop["head"]["mandible"][0:1],
                    target["head"]["mandible"],
                    joint_scale,
                ),
            },
            "root": {
                "distance": root_pos_weight * jp.exp(-(jp.linalg.norm(target["root"]["pos"]) / root_pos_scale) ** 2),
                "orientation_x": root_orientation_weight * target["root"]["xaxis"][0] #Dot-product
            },
        }
        
        for module, actuator_ids in self._ctrl_indices.items():
            actuator_ids = jp.array(actuator_ids)

            if self._config.force_cost != 0.0:
                module_sum_force = jp.sum(jp.abs(data.actuator_force[actuator_ids]))
                rewards[module]["force_cost"] = self._config.force_cost * module_sum_force
            
            if self._config.energy_cost != 0.0:
                module_energy_cost = jp.sum(jp.abs(
                    data.actuator_velocity[actuator_ids] * data.actuator_force[actuator_ids]
                ))
                rewards[module]["energy_cost"] = self._config.energy_cost * module_energy_cost

        metrics["rewards"] = rewards
        per_module_rewards = {k: sum(v.values()) for k, v in rewards.items()}
        
        return per_module_rewards

    def _add_extra_metrics(self, prop: dict, cur: dict, metrics: dict) -> None:
        '''Add a selection of extra metrics useful for tuning reward terms.'''
        target = cur
        metrics["hand_L"] = {
            "finger_level": prop["hand_L"]["finger_zz"][0],
            "wrist_angle_err": prop["hand_L"]["wrist_angle"][0:1] - target["hand_L"]["wrist_angle"],
            "pos_err": jp.linalg.norm(prop["hand_L"]["hand_pos"][0:3] - target["hand_L"]["pos"]),
        }
        metrics["arm_L"] = {
            "elbow_joint_err": prop["arm_L"]["elbow_angle"][0:1] - target["arm_L"]["elbow_angle"],
            "elbow_height_err": prop["arm_L"]["elbow_height"] - target["arm_L"]["elbow_height"],
        }
        metrics["foot_L"] = {
            "pos_err": jp.linalg.norm(prop["foot_L"]["foot_pos"][0:3] - target["foot_L"]["pos"]),
        }
        metrics["leg_L"] = {
            "orientation_match": jp.dot(prop["foot_L"]["xaxis"], target["foot_L"]["xaxis"]),
        }
        metrics["root"] = {
            # target["root"]["pos"/"rot"] are egocentric (from _build_target)
            "pos_err": jp.linalg.norm(target["root"]["pos"][0:3]),
            "xaxis_dot": target["root"]["xaxis"][0],
        }

    def _is_done(
        self, data: mjx.Data, info: Mapping[str, Any], cur: dict, metrics: dict
    ) -> bool:
        # 1. Torso too low (matching Julia's height_above_ground/10 < min_torso_z)
        torso_z = self.root_pos(data)[2]
        too_low = torso_z < self._config.min_torso_z
        metrics["terminations/torso_too_low"] = too_low

        # 2. Root too far from target (cur["root"]["pos"] is already egocentric)
        root_dist = jp.linalg.norm(cur["root"]["pos"])
        too_far = root_dist > self._config.max_target_distance
        metrics["terminations/root_too_far"] = too_far

        # 3. NaN check
        nan_terminated = jp.any(jp.isnan(data.qpos))
        metrics["terminations/nan_termination"] = jp.astype(nan_terminated, float)

        any_terminated = jp.logical_or(too_low, jp.logical_or(too_far, nan_terminated))
        metrics["terminations/any"] = any_terminated
        return any_terminated

    # -------------------------------------------------------------------------
    # Target access helpers
    # -------------------------------------------------------------------------

    def _interpolated_target(
        self, data: mjx.Data, info: Mapping[str, Any], time_ahead: float = 0.0
    ) -> dict:
        """Interpolated target at current time + time_ahead, in the current agent's frame.

        Linearly interpolates raw reference data between consecutive frames, then
        transforms to the current agent's yaw-aligned torso frame. Both current
        and future targets are always expressed relative to the agent's pose at
        observation time.
        """
        precise_frame = (
            (data.time + time_ahead) * self._config.mocap_hz + info["start_frame"]
        )
        int_frame = jp.clip(
            jp.floor(precise_frame).astype(int), 0, self._clip_length() - 2
        )
        frac = precise_frame - jp.floor(precise_frame)
        clip = info["reference_clip"]

        def interp(arr: jp.ndarray) -> jp.ndarray:
            return (1.0 - frac) * arr[clip, int_frame] + frac * arr[clip, int_frame + 1]

        def body_ref_xpos(name: str) -> jp.ndarray:
            return interp(self.reference_clips.body_xpos(name))

        def body_ref_xquat(name: str) -> jp.ndarray:
            return interp(self.reference_clips.body_xquat(name))

        def joint_ref(name: str) -> jp.ndarray:
            adr = self.reference_clips._qpos_names[name]
            return interp(self.reference_clips.qpos[:, :, adr:adr+1])

        # --- transform to current agent's torso frame ---
        torso_xpos = data.bind(self.mjx_model, self._spec.body("torso")).xpos
        torso_xmat = data.bind(self.mjx_model, self._spec.body("torso")).xmat

        def to_local(world_pos: jp.ndarray) -> jp.ndarray:
            """World-frame position → torso-frame position"""
            return torso_xmat.T @ (world_pos - torso_xpos)

        def _world_xaxis(name: str) -> jp.ndarray:
            """Body x-axis in world frame, from xquat [w, x, y, z]."""
            q = body_ref_xquat(name)
            w, x, y, z = q[0], q[1], q[2], q[3]
            return jp.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])

        def _world_zaxis(name: str) -> jp.ndarray:
            """Body z-axis in world frame, from xquat [w, x, y, z]."""
            q = body_ref_xquat(name)
            w, x, y, z = q[0], q[1], q[2], q[3]
            return jp.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])

        def xaxis_wrt_torso(name: str) -> jp.ndarray:
            return torso_xmat.T @ _world_xaxis(name)

        def zaxis_wrt_torso(name: str) -> jp.ndarray:
            return torso_xmat.T @ _world_zaxis(name)

        def height(body_name: str) -> jp.ndarray:
            """Absolute z-height of a body origin, shape (1,) (frame-invariant)."""
            return body_ref_xpos(body_name)[2:3]

        return {
            "hand_L": {
                "pos":          to_local(body_ref_xpos("hand_L")),
                "wrist_angle":  joint_ref("wrist_L"),
                "finger_angle": joint_ref("finger_L"),
                "xaxis":        xaxis_wrt_torso("hand_L"),
            },
            "arm_L": {
                "elbow_angle":     joint_ref("elbow_L"),
                "elbow_height":    height("lower_arm_L"),
                "shoulder_height": height("upper_arm_L"),
                "elbow_pos":       to_local(body_ref_xpos("lower_arm_L")),
            },
            "hand_R": {
                "pos":          to_local(body_ref_xpos("hand_R")),
                "wrist_angle":  joint_ref("wrist_R"),
                "finger_angle": joint_ref("finger_R"),
                "xaxis":        xaxis_wrt_torso("hand_R"),
            },
            "arm_R": {
                "elbow_angle":     joint_ref("elbow_R"),
                "elbow_height":    height("lower_arm_R"),
                "shoulder_height": height("upper_arm_R"),
                "elbow_pos":       to_local(body_ref_xpos("lower_arm_R")),
            },
            "foot_L": {
                "pos":         to_local(body_ref_xpos("foot_L")),
                "toe_angle":   joint_ref("toe_L"),
                "ankle_angle": joint_ref("ankle_L"),
                "xaxis":       xaxis_wrt_torso("foot_L"),
            },
            "leg_L": {
                "knee_pos":   to_local(body_ref_xpos("lower_leg_L")),
                "knee_angle": joint_ref("knee_L"),
                "hip_height": height("upper_leg_L"),
            },
            "foot_R": {
                "pos":         to_local(body_ref_xpos("foot_R")),
                "toe_angle":   joint_ref("toe_R"),
                "ankle_angle": joint_ref("ankle_R"),
                "xaxis":       xaxis_wrt_torso("foot_R"),
            },
            "leg_R": {
                "knee_pos":   to_local(body_ref_xpos("lower_leg_R")),
                "knee_angle": joint_ref("knee_R"),
                "hip_height": height("upper_leg_R"),
            },
            "torso": {
                "pelvis_xaxis":        xaxis_wrt_torso("pelvis"),
                "pelvis_zaxis":        zaxis_wrt_torso("pelvis"),
                "lumbar_bend":         0.658 * joint_ref("vertebra_2_bend") + 0.342 * joint_ref("vertebra_5_bend"),
                "lumbar_twist":        0.571 * joint_ref("vertebra_3_twist") + 0.429 * joint_ref("vertebra_6_twist"),
                "lumbar_extend":       0.605 * joint_ref("vertebra_1_extend") + 0.395 * joint_ref("vertebra_4_extend"),
                "height_above_ground": height("torso"),
            },
            "head": {
                "xaxis":          xaxis_wrt_torso("skull"),
                "zaxis":          zaxis_wrt_torso("skull"),
                "pos": to_local(body_ref_xpos("skull")),
                "mandible":       joint_ref("mandible"),
            },
            "root": {
                "pos": to_local(body_ref_xpos("torso")),
                "xaxis": xaxis_wrt_torso("torso"),
            },
        }

    # -------------------------------------------------------------------------
    # Reset helpers
    # -------------------------------------------------------------------------

    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            njmax=self._config.njmax,
            naconmax=self._config.naconmax,
            #nccdmax=self._config.naccdmax, #Available in mujoco-warp, but not yet in MJX?
        )
        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)
        data = data.replace(qpos=reference.qpos)
        if self._config.qvel_init == "zeros":
            data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel=reference.qvel)
        data = mjx.forward(self.mjx_model, data)
        return data

    def _clip_length(self) -> int:
        return self.reference_clips.qpos.shape[1]

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        time_in_frames = data.time * self._config.mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        add_labels: bool = False,
        termination_extra_frames: int = 0,
        render_ghost: bool = True,
    ) -> Sequence[np.ndarray]:
        """Render trajectory with optional ghost overlay of imitation target."""
        from vnl_playground.tasks.utils import _recolour_tree, dm_scale_spec

        if render_ghost:
            spec = self._spec.copy()
            ghost_spec = mujoco.MjSpec.from_file(str(consts.MODULAR_RODENT_WALKER_XML_PATH))
            ghost_rescale = self._config.rescale_factor
            if ghost_rescale != 1.0:
                ghost_spec = dm_scale_spec(ghost_spec, ghost_rescale)
            # Recolor walker bodies to white/transparent
            for body in ghost_spec.worldbody.bodies:
                _recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])
            spawn_site = spec.worldbody.add_frame(pos=(0, 0, 0.0), quat=(1, 0, 0, 0))
            spawn_body = spawn_site.attach_body(
                ghost_spec.body("walker"), "", suffix="-ghost"
            )
            spawn_body.add_freejoint()
            mj_model = spec.compile()
        else:
            mj_model = self.mj_model

        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        mj_data = mujoco.MjData(mj_model)

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        if camera is None:
            camera = -1

        rendered_frames = []
        for i, state in enumerate(trajectory):
            frame = self._get_cur_frame(state.data, state.info)
            clip = state.info["reference_clip"]
            ref = self.reference_clips.at(clip=clip, frame=frame)

            if render_ghost:
                mj_data.qpos = jp.concatenate((state.data.qpos, ref.qpos))
                mj_data.qvel = jp.concatenate((state.data.qvel, ref.qvel))
            else:
                mj_data.qpos = state.data.qpos
                mj_data.qvel = state.data.qvel
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
            if modify_scene_fns is not None:
                modify_scene_fns[i](renderer.scene)
            rendered_frame = renderer.render()
            if add_labels:
                import cv2

                clip_label = (
                    self.reference_clips.clip_names[clip]
                    if self.reference_clips.clip_names is not None
                    else str(clip)
                )
                cv2.putText(
                    rendered_frame,
                    f"Clip {clip} ({clip_label})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            rendered_frames.append(rendered_frame)
            if state.done:
                if add_labels:
                    import cv2

                    reason = "<Unknown>"
                    if state.info["truncated"]:
                        reason = "truncated"
                    for name in [
                        "torso_too_low",
                        "root_too_far",
                        "nan_termination",
                    ]:
                        if state.metrics.get(f"terminations/{name}", 0) > 0:
                            reason = name
                    cv2.putText(
                        rendered_frame,
                        reason,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                for t in range(termination_extra_frames):
                    rel_t = t / termination_extra_frames
                    fade_factor = 1 / (1 + np.exp(10 * (rel_t - 0.5)))
                    faded_frame = (rendered_frame * fade_factor).astype(np.uint8)
                    rendered_frames.append(faded_frame)
        return rendered_frames

    # -------------------------------------------------------------------------
    # Observation size properties
    # -------------------------------------------------------------------------

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.array(x.shape), obs)


# ---------------------------------------------------------------------------
# Module-level reward helpers
# ---------------------------------------------------------------------------


def _reward_shape(a: jp.ndarray, b: jp.ndarray, scale: float) -> jp.ndarray:
    """exp(-(norm(a-b)/scale)^2) — matches Julia's reward_shape."""
    return jp.exp(-(jp.linalg.norm(a - b) / scale) ** 2)

