"""Modular imitation task for the rodent.

Corresponds to modular_imitation.jl in the Julia implementation.
Follows the same structure as tasks/rodent/imitation.py.
"""

import collections
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import brax.math
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

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
        solver="cg",
        iterations=5,
        ls_iterations=5,
        naconmax=128*512,
        njmax=256,
        noslip_iterations=0,
        tolerance=1e-6,
        cone="pyramidal",
        impratio=1.0,
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=50,
        clip_length=250,
        clip_set="all",
        start_frame_range=[0, 44],
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


class ModularImitation(modular_base.ModularRodentEnv):
    """Modular multi-clip imitation environment for the rodent.

    Observations, proprioception, and rewards are organized hierarchically by
    body module (hand_L, arm_L, ..., torso, head), matching the Julia
    ModularImitationEnv. Reference targets are precomputed at init time.
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

        self._target_sensors = self._precompute_targets()

    def _precompute_targets(self) -> dict:
        """Precompute target sensor values for all reference clips and frames.

        Corresponds to Julia's precompute_target(). Processes one clip at a time
        (Python loop) with jax.vmap over frames within each clip to avoid OOM.

        Returns a nested dict with shape {module: {field: (n_clips, n_frames, dim)}}.
        """
        base_data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            njmax=self._config.njmax,
            naconmax=self._config.naconmax,
            #nccdmax=self._config.nccdmax # Available in mujoco-warp, but not yet in MJX?
        )

        def compute_for_frame(qpos, qvel):
            data = base_data.replace(qpos=qpos, qvel=qvel)
            data = mjx.forward(self.mjx_model, data)
            return self._get_target_from_pose(data)

        compute_for_clip = jax.jit(jax.vmap(compute_for_frame))

        n_clips = self.reference_clips.qpos.shape[0]
        results = [
            compute_for_clip(
                self.reference_clips.qpos[i],
                self.reference_clips.qvel[i],
            )
            for i in range(n_clips)
        ]
        return jax.tree_util.tree_map(lambda *xs: jp.stack(xs), *results)

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

        metrics = {
            "current_frame": jp.astype(self._get_cur_frame(data, info), float),
        }
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)
        self._add_extra_metrics(data, info, metrics)

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

        obs = self._get_obs(data, info)
        terminated = self._is_done(data, info, state.metrics)
        done = jp.logical_or(terminated, info["truncated"])
        reward = self._get_reward(data, info, state.metrics)
        reward = jax.tree.map(jp.nan_to_num, reward)
        self._add_extra_metrics(data, info, state.metrics)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        state.metrics["current_frame"] = jp.astype(
            self._get_cur_frame(data, info), float
        )
        return state

    def _get_obs(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> collections.OrderedDict:
        """Hierarchical observation dict organized by body module.

        Each module contains {proprioception, current_target, future_target}.
        Root is included with only current_target and future_target (no
        proprioception, matching Julia's dummy root proprioception).
        """
        prop = self._get_modular_proprioception(data)
        cur = self._interpolated_target(data, info, time_ahead=0.0)
        fut = self._interpolated_target(data, info, time_ahead=0.1)

        # Transform root to egocentric (matching Julia's state())
        yaw_mat = self.torso_yawmat(data)
        root_cur = self.root_pos(data)
        root_quat_cur = self.root_xquat(data)
        cur = _with_egocentric_root(cur, yaw_mat, root_cur, root_quat_cur)
        fut = _with_egocentric_root(fut, yaw_mat, root_cur, root_quat_cur)

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

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: dict
    ) -> float:
        """Compute per-module rewards, log to metrics, return scalar total.

        Reward structure matches Julia's compute_rewards() exactly.
        """
        prop = self._get_modular_proprioception(data)
        target = self._interpolated_target(data, info, time_ahead=0.0)
        pos_scale = self._config.reward_terms.limb_pos_exp_scale
        joint_scale = self._config.reward_terms.joint_exp_scale

        root_cur = self.root_pos(data)
        root_quat_cur = self.root_xquat(data)
        root_dist = jp.linalg.norm(target["root"]["pos"] - root_cur)
        root_ang = 2.0 * jp.arccos(
            jp.minimum(
                1.0,
                jp.abs(jp.dot(target["root"]["rot"], root_quat_cur)),
            )
        )

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
                "orientation": jp.dot(
                    prop["foot_L"]["xaxis"], target["foot_L"]["xaxis"]
                ),
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
                "orientation": jp.dot(
                    prop["foot_R"]["xaxis"], target["foot_R"]["xaxis"]
                ),
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

        per_module_rewards = {k: sum(v.values()) for k, v in rewards.items()}
        for module, module_sum in per_module_rewards.items():
            metrics[f"rewards/{module}"] = module_sum

        return per_module_rewards

    def _add_extra_metrics(self, data: mjx.Data, info: Mapping[str, Any], metrics: dict) -> None:
        '''Add a selection of extra metrics useful for tuning reward terms.'''
        prop = self._get_modular_proprioception(data)
        target = self._interpolated_target(data, info, time_ahead=0.0)
        root_quat_cur = self.root_xquat(data)
        root_ang = 2.0 * jp.arccos(
            jp.minimum(
                1.0,
                jp.abs(jp.dot(target["root"]["rot"], root_quat_cur)),
            )
        )
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
            "pos_err": jp.linalg.norm(target["root"]["pos"] - self.root_pos(data)),
            "ang_err": root_ang,
        }

    def _is_done(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: dict
    ) -> bool:
        # 1. Torso too low (matching Julia's height_above_ground/10 < min_torso_z)
        torso_z = self.root_pos(data)[2]
        too_low = torso_z < self._config.min_torso_z
        metrics["terminations/torso_too_low"] = too_low

        # 2. Root too far from target (uses raw global positions, not egocentric)
        target = self._interpolated_target(data, info, time_ahead=0.0)
        root_dist = jp.linalg.norm(target["root"]["pos"] - self.root_pos(data))
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
        """Linearly interpolated target at current time + time_ahead seconds.

        Corresponds to Julia's interpolated_target(env, env.target_timepoint[] + time_ahead).
        """
        precise_frame = (
            (data.time + time_ahead) * self._config.mocap_hz + info["start_frame"]
        )
        int_frame = jp.clip(
            jp.floor(precise_frame).astype(int), 0, self._clip_length() - 2
        )
        frac = precise_frame - jp.floor(precise_frame)
        clip = info["reference_clip"]
        return jax.tree_util.tree_map(
            lambda arr: (1.0 - frac) * arr[clip, int_frame]
            + frac * arr[clip, int_frame + 1],
            self._target_sensors,
        )

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


def _with_egocentric_root(
    target: dict,
    yaw_mat: jp.ndarray,
    root_cur: jp.ndarray,
    root_quat_cur: jp.ndarray,
) -> dict:
    """Transform target root pos/rot to egocentric frame.

    Matches Julia's state() which transforms current_target.root to egocentric
    coordinates relative to the agent's current pose.
    """
    egocentric_pos = yaw_mat @ (target["root"]["pos"] - root_cur)
    egocentric_rot = _sub_quat(target["root"]["rot"], root_quat_cur)
    return {
        **target,
        "root": {"pos": egocentric_pos, "rot": egocentric_rot},
    }
