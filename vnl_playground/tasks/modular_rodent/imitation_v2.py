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
        # Termination
        max_target_distance=0.05, # meters
        min_torso_z=0.03,         # meters
        # Reward scales
        reward_terms={
            "limb_pos_exp_scale": 0.015, # sigma for distance-based errors (meters)
            "joint_exp_scale": 0.1,      # sigma for angle-based errors (radians)
            "root_pos_scale": 0.05,      # sigma for root position reward (meters)
            "root_pos_weight": 5.0,      # weight for root position reward
            "root_ang_scale": 0.5,       # sigma for root angle reward (radians)
            "root_ang_weight": 5.0,      # weight for root angle reward
        },
    )


class ModularImitation_v2(modular_base.ModularRodentEnv):
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

        m = self._mj_model

        # Body IDs for manual axis/position computations (replaces sensor reads for
        # framexaxis, framezaxis, framepos sensors — avoids MJX sensor bugs).
        self._body_ids = {
            name: m.body(name).id
            for name in ["torso", "pelvis", "skull", "hand_L", "hand_R",
                         "finger_L", "finger_R", "foot_L", "foot_R",
                         "toe_L", "toe_R", "lower_arm_L", "lower_arm_R",
                         "lower_leg_L", "lower_leg_R"]
        }

        # Site IDs for knee positions.
        # MJX has a bug: framepos with objtype="site" ignores refbody and returns
        # world-frame position. We fix this by computing manually from site_xpos.
        self._site_ids = {
            name: m.site(name).id
            for name in ["knee_L", "knee_R"]
        }

        # Knee site local positions (in upper_leg_L/R frame) for reference target.
        # Used to reconstruct reference knee site world position from body_xpos + xquat.
        self._knee_L_local = jp.array(m.site("knee_L").pos)
        self._knee_R_local = jp.array(m.site("knee_R").pos)

        # jointpos sensors → qpos address (used for proprioception and reference target values).
        # Only include true jointpos sensors (not tendonpos).
        # Symmetric: L and R entries are listed in parallel.
        _jointpos_sensors = [
            "hand_L/wrist_angle",    "hand_R/wrist_angle",
            "hand_L/finger_angle",   "hand_R/finger_angle",
            "arm_L/elbow_angle",     "arm_R/elbow_angle",
            "arm_L/shoulder_sup",    "arm_R/shoulder_sup",
            "arm_L/shoulder",        "arm_R/shoulder",
            "arm_L/scapula_extend",  "arm_R/scapula_extend",
            "arm_L/scapula_abduct",  "arm_R/scapula_abduct",
            "arm_L/scapula_supinate","arm_R/scapula_supinate",
            "foot_L/knee_angle",     "foot_R/knee_angle",
            "foot_L/ankle_angle",    "foot_R/ankle_angle",
            "foot_L/toe_angle",      "foot_R/toe_angle",
            "leg_L/hip_supinate",    "leg_R/hip_supinate",
            "leg_L/hip_abduct",      "leg_R/hip_abduct",
            "leg_L/hip_extend",      "leg_R/hip_extend",
            "leg_L/knee_angle",      "leg_R/knee_angle",
            "leg_L/ankle_angle",     "leg_R/ankle_angle",
            "head/atlas", "head/atlant_extend", "head/mandible",
            "head/cervical_extend", "head/cervical_bend", "head/cervical_twist",
        ]
        self._sensor_qpos_addr = {
            name: int(m.jnt_qposadr[m.sensor_objid[m.sensor(name).id]])
            for name in _jointpos_sensors
        }

        # Lumbar sensors are tendonpos (NOT jointpos) — linear weighted sums of vertebra joints.
        # Precompute (qpos_addr, coefficient) pairs for computing target tendon lengths from qpos.
        _lumbar_joints = {
            "lumbar_extend": [("vertebra_1_extend", 0.604983465832), ("vertebra_4_extend", 0.395016534168)],
            "lumbar_bend":   [("vertebra_2_bend",   0.658212326553), ("vertebra_5_bend",   0.341787673447)],
            "lumbar_twist":  [("vertebra_3_twist",  0.570669983621), ("vertebra_6_twist",  0.429330016379)],
        }
        self._tendon_qpos = {
            tendon: [(int(m.jnt_qposadr[m.joint(jname).id]), coef)
                     for jname, coef in joints]
            for tendon, joints in _lumbar_joints.items()
        }

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
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jp.astype(truncated, float)
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()

        metrics = {"current_frame": jp.astype(self._get_cur_frame(data, info), float)}
        obs, reward, done = self._process(data, info, metrics)
        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

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
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jp.astype(truncated, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        metrics = state.metrics
        obs, reward, terminated = self._process(data, info, metrics)
        reward = jax.tree.map(jp.nan_to_num, reward)
        done = jp.logical_or(terminated, info["truncated"])
        metrics["current_frame"] = jp.astype(self._get_cur_frame(data, info), float)
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
        reward = self._get_reward(prop, cur, metrics)
        terminated = self._is_done(data, info, cur, metrics)
        self._add_extra_metrics(prop, cur, metrics)
        return obs, reward, terminated

    @override
    def _get_modular_proprioception(
        self, data: mjx.Data
    ) -> dict[str, dict[str, jp.ndarray]]:
        """Hierarchical proprioception dict organized by body module.
        Simplified compared to base.

        Position and axis quantities are computed directly from data.xpos/xmat/site_xpos
        to avoid MJX sensor bugs (e.g. framepos+objtype=site ignores refbody).
        Contact, accelerometer, gyro, and linvel sensors are kept as sensor reads.
        """
        m = self._mj_model
        yaw_mat = self.torso_yawmat(data)

        # Precompute torso rotation inverse for sensors with refbody=torso.
        # data.xmat[id].reshape(3,3): column k = body's k-th axis in world frame.
        # .T gives world→body transform (framexaxis/framezaxis/framepos with refbody=torso).
        bid = self._body_ids
        sid = self._site_ids
        torso_id = bid["torso"]
        torso_rotmat_inv = data.xmat[torso_id].reshape(3, 3).T

        def body_xaxis_torso(body_id: int) -> jp.ndarray:
            """Body x-axis in torso body frame. Matches framexaxis with refbody=torso."""
            return torso_rotmat_inv @ data.xmat[body_id].reshape(3, 3)[:, 0]

        def body_zaxis_torso(body_id: int) -> jp.ndarray:
            """Body z-axis in torso body frame. Matches framezaxis with refbody=torso."""
            return torso_rotmat_inv @ data.xmat[body_id].reshape(3, 3)[:, 2]

        def body_zaxis_world(body_id: int) -> jp.ndarray:
            """Body z-axis in world frame. Matches framezaxis with no refbody."""
            return data.xmat[body_id].reshape(3, 3)[:, 2]

        def body_pos_torso(body_id: int) -> jp.ndarray:
            """Body origin in torso body frame. Matches framepos objtype=xbody."""
            return torso_rotmat_inv @ (data.xpos[body_id] - data.xpos[torso_id])

        def site_pos_torso(site_id: int) -> jp.ndarray:
            """Site position in torso body frame. Fixes framepos objtype=site MJX bug."""
            return torso_rotmat_inv @ (data.site_xpos[site_id] - data.xpos[torso_id])

        def joint(sensor_name: str) -> jp.ndarray:
            """Joint angle from qpos, shape (1,) matching jointpos sensor output."""
            addr = self._sensor_qpos_addr[sensor_name]
            return data.qpos[addr:addr + 1]

        def lumbar_tendon(tendon_name: str) -> jp.ndarray:
            """Tendon length as weighted sum of vertebra joint qpos values."""
            return jp.array([sum(coef * data.qpos[addr]
                                 for addr, coef in self._tendon_qpos[tendon_name])])

        return {
            "hand_L": {
                "palm_contact": get_sensor_data(m, data, "hand_L/palm_contact"),
                "egocentric_hand_pos": self.body_relpos(data, "hand_L"),
                "wrist_angle": joint("hand_L/wrist_angle"),
                "finger_angle": joint("hand_L/finger_angle"),
                "xaxis": body_xaxis_torso(bid["hand_L"]),
                "finger_zz": body_zaxis_world(bid["finger_L"])[2:3],
            },
            "arm_L": {
                "egocentric_elbow_pos": body_pos_torso(bid["lower_arm_L"]),
                "elbow_angle": joint("arm_L/elbow_angle"),
                "shoulder_sup": joint("arm_L/shoulder_sup"),
                "shoulder": joint("arm_L/shoulder"),
                "scapula_extend": joint("arm_L/scapula_extend"),
                "scapula_abduct": joint("arm_L/scapula_abduct"),
                "scapula_supinate": joint("arm_L/scapula_supinate"),
                "elbow_height": jp.expand_dims(self.site_z(data, "elbow_L"), axis=0),
                "shoulder_height": jp.expand_dims(self.site_z(data, "shoulder_L"), axis=0),
                "hand_linvel": yaw_mat @ get_sensor_data(m, data, "arm_L/hand_linvel"),
            },
            "hand_R": {
                "palm_contact": get_sensor_data(m, data, "hand_R/palm_contact"),
                "egocentric_hand_pos": self.body_relpos(data, "hand_R"),
                "wrist_angle": joint("hand_R/wrist_angle"),
                "finger_angle": joint("hand_R/finger_angle"),
                "xaxis": body_xaxis_torso(bid["hand_R"]),
                "finger_zz": body_zaxis_world(bid["finger_R"])[2:3],
            },
            "arm_R": {
                "egocentric_elbow_pos": body_pos_torso(bid["lower_arm_R"]),
                "elbow_angle": joint("arm_R/elbow_angle"),
                "shoulder_sup": joint("arm_R/shoulder_sup"),
                "shoulder": joint("arm_R/shoulder"),
                "scapula_extend": joint("arm_R/scapula_extend"),
                "scapula_abduct": joint("arm_R/scapula_abduct"),
                "scapula_supinate": joint("arm_R/scapula_supinate"),
                "elbow_height": jp.expand_dims(self.site_z(data, "elbow_R"), axis=0),
                "shoulder_height": jp.expand_dims(self.site_z(data, "shoulder_R"), axis=0),
                "hand_linvel": yaw_mat @ get_sensor_data(m, data, "arm_R/hand_linvel"),
            },
            "foot_L": {
                "sole_contact": get_sensor_data(m, data, "foot_L/sole_contact"),
                "heel_contact": get_sensor_data(m, data, "foot_L/heel_contact"),
                "egocentric_foot_pos": self.body_relpos(data, "foot_L"),
                "knee_angle": joint("foot_L/knee_angle"),
                "ankle_angle": joint("foot_L/ankle_angle"),
                "toe_angle": joint("foot_L/toe_angle"),
                "xaxis": body_xaxis_torso(bid["foot_L"]),
                "toe_zz": body_zaxis_world(bid["toe_L"])[2:3],
            },
            "leg_L": {
                "egocentric_knee_pos": site_pos_torso(sid["knee_L"]),
                "hip_supinate": joint("leg_L/hip_supinate"),
                "hip_abduct": joint("leg_L/hip_abduct"),
                "hip_extend": joint("leg_L/hip_extend"),
                "knee_angle": joint("leg_L/knee_angle"),
                "ankle_angle": joint("leg_L/ankle_angle"),
                "pelvis_zaxis": yaw_mat @ body_zaxis_world(bid["pelvis"]),
                "hip_height": jp.expand_dims(self.site_z(data, "hip_L"), axis=0),
                #"hip_accelerometer": get_sensor_data(m, data, "leg_L/hip_accelerometer"),
                #"hip_gyro": get_sensor_data(m, data, "leg_L/hip_gyro"),
            },
            "foot_R": {
                "sole_contact": get_sensor_data(m, data, "foot_R/sole_contact"),
                "heel_contact": get_sensor_data(m, data, "foot_R/heel_contact"),
                "egocentric_foot_pos": self.body_relpos(data, "foot_R"),
                "knee_angle": joint("foot_R/knee_angle"),
                "ankle_angle": joint("foot_R/ankle_angle"),
                "toe_angle": joint("foot_R/toe_angle"),
                "xaxis": body_xaxis_torso(bid["foot_R"]),
                "toe_zz": body_zaxis_world(bid["toe_R"])[2:3],
            },
            "leg_R": {
                "egocentric_knee_pos": site_pos_torso(sid["knee_R"]),
                "hip_supinate": joint("leg_R/hip_supinate"),
                "hip_abduct": joint("leg_R/hip_abduct"),
                "hip_extend": joint("leg_R/hip_extend"),
                "knee_angle": joint("leg_R/knee_angle"),
                "ankle_angle": joint("leg_R/ankle_angle"),
                "pelvis_zaxis": yaw_mat @ body_zaxis_world(bid["pelvis"]),
                "hip_height": jp.expand_dims(self.site_z(data, "hip_R"), axis=0),
                #"hip_accelerometer": get_sensor_data(m, data, "leg_R/hip_accelerometer"),
                #"hip_gyro": get_sensor_data(m, data, "leg_R/hip_gyro"),
            },
            "torso": {
                "accelerometer": get_sensor_data(m, data, "torso/accelerometer"),
                "gyro": get_sensor_data(m, data, "torso/gyro"),
                "zaxis": yaw_mat @ body_zaxis_world(torso_id),
                "lumbar_extend": lumbar_tendon("lumbar_extend"),
                "lumbar_bend": lumbar_tendon("lumbar_bend"),
                "lumbar_twist": lumbar_tendon("lumbar_twist"),
                "linvel": get_sensor_data(m, data, "torso/linvel"),
                "height_above_ground": jp.expand_dims(
                    self.body_relpos(data, "torso")[2], axis=0
                ),
                "pelvis_zaxis": yaw_mat @ body_zaxis_world(bid["pelvis"]),
            },
            "head": {
                "accelerometer": get_sensor_data(m, data, "head/accelerometer"),
                "egocentric_pos": self.body_relpos(data, "skull"),
                "xaxis": body_xaxis_torso(bid["skull"]),
                "zaxis": body_zaxis_torso(bid["skull"]),
                "atlas": joint("head/atlas"),
                "atlant_extend": joint("head/atlant_extend"),
                "mandible": joint("head/mandible"),
                "cervical_extend": joint("head/cervical_extend"),
                "cervical_bend": joint("head/cervical_bend"),
                "cervical_twist": joint("head/cervical_twist"),
                "linvel": get_sensor_data(m, data, "head/linvel"),
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

    def _get_reward(self, prop: dict, cur: dict, metrics: dict) -> float:
        """Compute per-module rewards, log to metrics, return scalar total."""
        target = cur
        pos_scale = self._config.reward_terms.limb_pos_exp_scale
        joint_scale = self._config.reward_terms.joint_exp_scale

        # target["root"]["pos"] and ["rot"] are already egocentric (from _build_target)
        root_dist = jp.linalg.norm(target["root"]["pos"])
        root_ang = jp.linalg.norm(target["root"]["rot"])  # _sub_quat gives axis*angle; ‖·‖ = angle

        rewards = {
            "hand_L": {
                "finger_level": prop["hand_L"]["finger_zz"][0],
                "wrist_joint": _reward_shape(
                    prop["hand_L"]["wrist_angle"],
                    target["hand_L"]["wrist_angle"],
                    joint_scale,
                ),
                "orientation": jp.dot(
                    prop["hand_L"]["xaxis"], target["hand_L"]["xaxis"]
                ),
                "hand_pos": _reward_shape(
                    prop["hand_L"]["egocentric_hand_pos"],
                    target["hand_L"]["pos"],
                    pos_scale,
                ),
            },
            "arm_L": {
                "elbow_joint": _reward_shape(
                    prop["arm_L"]["elbow_angle"],
                    target["arm_L"]["elbow_angle"],
                    joint_scale,
                ),
                "elbow_height": _reward_shape(
                    prop["arm_L"]["elbow_height"],
                    target["arm_L"]["elbow_height"],
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
                    prop["hand_R"]["wrist_angle"],
                    target["hand_R"]["wrist_angle"],
                    joint_scale,
                ),
                "orientation": jp.dot(
                    prop["hand_R"]["xaxis"], target["hand_R"]["xaxis"]
                ),
                "hand_pos": _reward_shape(
                    prop["hand_R"]["egocentric_hand_pos"],
                    target["hand_R"]["pos"],
                    pos_scale,
                ),
            },
            "arm_R": {
                "elbow_joint": _reward_shape(
                    prop["arm_R"]["elbow_angle"],
                    target["arm_R"]["elbow_angle"],
                    joint_scale,
                ),
                "elbow_height": _reward_shape(
                    prop["arm_R"]["elbow_height"],
                    target["arm_R"]["elbow_height"],
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
                    prop["foot_L"]["ankle_angle"],
                    target["foot_L"]["ankle_angle"],
                    joint_scale,
                ),
                #"orientation": jp.dot(
                #    prop["foot_L"]["xaxis"], target["foot_L"]["xaxis"]
                #),
                "foot_pos": _reward_shape(
                    prop["foot_L"]["egocentric_foot_pos"],
                    target["foot_L"]["pos"],
                    pos_scale,
                ),
            },
            "leg_L": {
                "knee_joint": _reward_shape(
                    prop["leg_L"]["knee_angle"],
                    target["leg_L"]["knee_angle"],
                    joint_scale,
                ),
                "knee_pos": _reward_shape(
                    prop["leg_L"]["egocentric_knee_pos"],
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
                    prop["foot_R"]["ankle_angle"],
                    target["foot_R"]["ankle_angle"],
                    joint_scale,
                ),
                #"orientation": jp.dot(
                #    prop["foot_R"]["xaxis"], target["foot_R"]["xaxis"]
                #),
                "foot_pos": _reward_shape(
                    prop["foot_R"]["egocentric_foot_pos"],
                    target["foot_R"]["pos"],
                    pos_scale,
                ),
            },
            "leg_R": {
                "knee_joint": _reward_shape(
                    prop["leg_R"]["knee_angle"],
                    target["leg_R"]["knee_angle"],
                    joint_scale,
                ),
                "knee_pos": _reward_shape(
                    prop["leg_R"]["egocentric_knee_pos"],
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
                "orientation_z": _cosine_dist(
                    prop["torso"]["zaxis"], target["torso"]["zaxis"]
                ),
                "height_above_ground": _reward_shape(
                    prop["torso"]["height_above_ground"],
                    target["torso"]["height_above_ground"],
                    pos_scale,
                ),
                "lumbar_bend": _reward_shape(
                    prop["torso"]["lumbar_bend"],
                    target["torso"]["lumbar_bend"],
                    joint_scale,
                ),
                "lumbar_twist": _reward_shape(
                    prop["torso"]["lumbar_twist"],
                    target["torso"]["lumbar_twist"],
                    joint_scale,
                ),
                "lumbar_extend": _reward_shape(
                    prop["torso"]["lumbar_extend"],
                    target["torso"]["lumbar_extend"],
                    joint_scale,
                ),
            },
            "head": {
                "orientation_x": _cosine_dist(
                    prop["head"]["xaxis"], target["head"]["xaxis"]
                ),
                "orientation_z": _cosine_dist(
                    prop["head"]["zaxis"], target["head"]["zaxis"]
                ),
                "head_pos": _reward_shape(
                    prop["head"]["egocentric_pos"],
                    target["head"]["egocentric_pos"],
                    pos_scale,
                ),
                "mandible": _reward_shape(
                    prop["head"]["mandible"],
                    target["head"]["mandible"],
                    joint_scale,
                ),
            },
            "root": {
                "distance": self._config.reward_terms.root_pos_weight
                * jp.exp(
                    -(root_dist / self._config.reward_terms.root_pos_scale) ** 2
                ),
                "angle": self._config.reward_terms.root_ang_weight
                * jp.exp(
                    -(root_ang / self._config.reward_terms.root_ang_scale) ** 2
                ),
            },
        }

        metrics["rewards"] = rewards
        per_module_rewards = {k: sum(v.values()) for k, v in rewards.items()}
        
        return per_module_rewards

    def _add_extra_metrics(self, prop: dict, cur: dict, metrics: dict) -> None:
        '''Add a selection of extra metrics useful for tuning reward terms.'''
        target = cur
        metrics["hand_L"] = {
            "finger_level": prop["hand_L"]["finger_zz"][0],
            "wrist_angle_err": prop["hand_L"]["wrist_angle"] - target["hand_L"]["wrist_angle"],
            "pos_err": jp.linalg.norm(prop["hand_L"]["egocentric_hand_pos"] - target["hand_L"]["pos"]),
        }
        metrics["arm_L"] = {
            "elbow_joint_err": prop["arm_L"]["elbow_angle"] - target["arm_L"]["elbow_angle"],
            "elbow_height_err": prop["arm_L"]["elbow_height"] - target["arm_L"]["elbow_height"],
        }
        metrics["foot_L"] = {
            "pos_err": jp.linalg.norm(prop["foot_L"]["egocentric_foot_pos"] - target["foot_L"]["pos"]),
        }
        metrics["leg_L"] = {
            "orientation_match": jp.dot(prop["foot_L"]["xaxis"], target["foot_L"]["xaxis"]),
        }
        metrics["root"] = {
            # target["root"]["pos"/"rot"] are egocentric (from _build_target)
            "pos_err": jp.linalg.norm(target["root"]["pos"]),
            "ang_err": jp.linalg.norm(target["root"]["rot"]),
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
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        nan_terminated = jp.sum(jp.isnan(flattened_vals)) > 0
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

        def body_xpos(name: str) -> jp.ndarray:
            return interp(self.reference_clips.body_xpos(name))

        def body_xquat(name: str) -> jp.ndarray:
            return interp(self.reference_clips.body_xquat(name))

        ref_qpos = interp(self.reference_clips.qpos)

        # --- transform to current agent's torso frame ---
        yaw_mat = self.torso_yawmat(data)
        ref_torso_xpos = body_xpos("torso")
        ref_torso_xy = jp.array([ref_torso_xpos[0], ref_torso_xpos[1], 0.0])

        # Full inverse rotation of agent's torso: world → torso body frame.
        # Used for sensors with refbody=torso (framexaxis/zaxis/pos with refname="torso").
        torso_id = self._mj_model.body("torso").id
        torso_rotmat_inv = data.xmat[torso_id].reshape(3, 3).T

        def to_local(world_pos: jp.ndarray) -> jp.ndarray:
            """World-frame position → current agent's yaw-aligned frame (z preserved)."""
            return yaw_mat @ (world_pos - ref_torso_xy)

        def _world_xaxis(name: str) -> jp.ndarray:
            """Body x-axis in world frame, from xquat [w, x, y, z]."""
            q = body_xquat(name)
            w, x, y, z = q[0], q[1], q[2], q[3]
            return jp.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])

        def _world_zaxis(name: str) -> jp.ndarray:
            """Body z-axis in world frame, from xquat [w, x, y, z]."""
            q = body_xquat(name)
            w, x, y, z = q[0], q[1], q[2], q[3]
            return jp.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])

        def xaxis_yaw(name: str) -> jp.ndarray:
            """Body x-axis in yaw-aligned frame. Matches framexaxis with no refbody."""
            return yaw_mat @ _world_xaxis(name)

        def zaxis_yaw(name: str) -> jp.ndarray:
            """Body z-axis in yaw-aligned frame. Matches framezaxis with no refbody."""
            return yaw_mat @ _world_zaxis(name)

        def xaxis_torso(name: str) -> jp.ndarray:
            """Body x-axis in torso body frame. Matches framexaxis with refbody=torso."""
            return torso_rotmat_inv @ _world_xaxis(name)

        def zaxis_torso(name: str) -> jp.ndarray:
            """Body z-axis in torso body frame. Matches framezaxis with refbody=torso."""
            return torso_rotmat_inv @ _world_zaxis(name)

        def joint(sensor_name: str) -> jp.ndarray:
            """Single joint angle from ref_qpos, shape (1,) matching sensor output."""
            addr = self._sensor_qpos_addr[sensor_name]
            return ref_qpos[addr:addr + 1]

        def tendon(tendon_name: str) -> jp.ndarray:
            """Tendon length from ref_qpos as weighted sum of vertebra joints.
            Matches tendonpos sensor output for fixed tendons (lumbar_*)."""
            return jp.array([sum(coef * ref_qpos[addr]
                                 for addr, coef in self._tendon_qpos[tendon_name])])

        def height(body_name: str) -> jp.ndarray:
            """Absolute z-height of a body origin, shape (1,) (frame-invariant)."""
            return body_xpos(body_name)[2:3]

        def ref_site_xpos(site_local: jp.ndarray, parent_body: str) -> jp.ndarray:
            """Reference world position of a site from parent body xpos + xquat.

            Reconstructs the site world position from HDF5 body data, matching
            what data.site_xpos would give if the model is at the reference pose.
            """
            q = body_xquat(parent_body)  # [w, x, y, z] MuJoCo convention
            w, x, y, z = q[0], q[1], q[2], q[3]
            R = jp.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
            ])
            return body_xpos(parent_body) + R @ site_local

        return {
            "hand_L": {
                "pos":          to_local(body_xpos("hand_L")),
                "wrist_angle":  joint("hand_L/wrist_angle"),
                "finger_angle": joint("hand_L/finger_angle"),
                "xaxis":        xaxis_torso("hand_L"),        # sensor: refbody=torso
            },
            "arm_L": {
                "elbow_angle":     joint("arm_L/elbow_angle"),
                "elbow_height":    height("lower_arm_L"),   # body origin ≈ elbow joint
                "shoulder_height": height("upper_arm_L"),   # body origin ≈ shoulder joint
                "elbow_pos_torso": torso_rotmat_inv @ (body_xpos("lower_arm_L") - ref_torso_xpos),
            },
            "hand_R": {
                "pos":          to_local(body_xpos("hand_R")),
                "wrist_angle":  joint("hand_R/wrist_angle"),
                "finger_angle": joint("hand_R/finger_angle"),
                "xaxis":        xaxis_torso("hand_R"),        # sensor: refbody=torso
            },
            "arm_R": {
                "elbow_angle":     joint("arm_R/elbow_angle"),
                "elbow_height":    height("lower_arm_R"),
                "shoulder_height": height("upper_arm_R"),
                "elbow_pos_torso": torso_rotmat_inv @ (body_xpos("lower_arm_R") - ref_torso_xpos),
            },
            "foot_L": {
                "pos":         to_local(body_xpos("foot_L")),
                "toe_angle":   joint("foot_L/toe_angle"),
                "ankle_angle": joint("foot_L/ankle_angle"),
                "xaxis":       xaxis_torso("foot_L"),         # sensor: refbody=torso
            },
            "leg_L": {
                # Reconstruct reference knee site position from upper_leg_L body + rotation.
                # Avoids HDF5 body mismatch; matches site_pos_torso(knee_L) in proprioception.
                "knee_pos":   torso_rotmat_inv @ (ref_site_xpos(self._knee_L_local, "upper_leg_L") - ref_torso_xpos),
                "knee_angle": joint("leg_L/knee_angle"),
                "hip_height": height("upper_leg_L"),               # body origin = hip
            },
            "foot_R": {
                "pos":         to_local(body_xpos("foot_R")),
                "toe_angle":   joint("foot_R/toe_angle"),
                "ankle_angle": joint("foot_R/ankle_angle"),
                "xaxis":       xaxis_torso("foot_R"),         # sensor: refbody=torso
            },
            "leg_R": {
                # Reconstruct reference knee site position from upper_leg_R body + rotation.
                "knee_pos":   torso_rotmat_inv @ (ref_site_xpos(self._knee_R_local, "upper_leg_R") - ref_torso_xpos),
                "knee_angle": joint("leg_R/knee_angle"),
                "hip_height": height("upper_leg_R"),
            },
            "torso": {
                "zaxis":               zaxis_yaw("torso"),   # sensor: no refbody → world → yaw
                "pelvis_zaxis":        zaxis_yaw("pelvis"),  # sensor: no refbody → world → yaw
                "lumbar_bend":         tendon("lumbar_bend"),    # tendonpos sensor
                "lumbar_twist":        tendon("lumbar_twist"),
                "lumbar_extend":       tendon("lumbar_extend"),
                "height_above_ground": height("torso"),
            },
            "head": {
                "xaxis":          xaxis_torso("skull"),       # sensor: refbody=torso
                "zaxis":          zaxis_torso("skull"),       # sensor: refbody=torso
                "egocentric_pos": to_local(body_xpos("skull")),
                "mandible":       joint("head/mandible"),
            },
            "root": {
                "pos": yaw_mat @ (ref_torso_xpos - self.root_pos(data)),
                "rot": _sub_quat(body_xquat("torso"), self.root_xquat(data)),
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


def _cosine_dist(a: jp.ndarray, b: jp.ndarray) -> jp.ndarray:
    """Cosine similarity — matches Julia's cosine_dist."""
    return jp.dot(a, b) / (jp.linalg.norm(a) * jp.linalg.norm(b))


def _sub_quat(q_target: jp.ndarray, q_current: jp.ndarray) -> jp.ndarray:
    """Relative rotation from q_current to q_target as a 3D angular velocity.

    Matches Julia's subQuat(qa=q_target, qb=q_current):
        q_diff = conj(q_current) * q_target
        return quat2Vel(q_diff, dt=1)

    Both quaternions use MuJoCo convention [w, x, y, z].
    The result is a 3-vector: axis * angle (radians), dt=1.
    """
    q_diff = brax.math.quat_mul(brax.math.quat_inv(q_current), q_target)
    w, xyz = q_diff[0], q_diff[1:]
    sin_a_2 = jp.linalg.norm(xyz)
    axis = xyz / (sin_a_2 + 1e-10)
    speed = 2.0 * jp.arctan2(sin_a_2, w)
    speed = jp.where(speed > jp.pi, speed - 2.0 * jp.pi, speed)
    return axis * speed


