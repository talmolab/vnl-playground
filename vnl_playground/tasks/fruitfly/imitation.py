"""Fruitfly multi-clip imitation environment."""

import collections
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import brax.math
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from jax import flatten_util

from .. import utils
from . import base as fruitfly_base
from . import consts
from vnl_playground.tasks.reference_clips import ReferenceClips
from vnl_playground.tasks.reward_registry import RewardRegistry

_registry = RewardRegistry()


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.FRUITFLY_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        mujoco_impl="warp",  # Use warp backend for faster testing
        naconmax=1024 * 10,
        sim_dt=0.0002,  # 5000 Hz physics
        ctrl_dt=0.002,  # 500 Hz control
        solver="newton",
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=False,  # Keep XML actuators as-is
        rescale_factor=1.0,
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=500,
        clip_length=600,
        clip_set="all",
        reference_length=5,
        start_frame_range=[0, 50],  # random_init_range from config
        qvel_init="zeros",
        keep_clips_idx=None,
        # Reward terms configuration.
        # For imitation rewards, the formula is: weight * exp(-((error / exp_scale)^2) / 2)
        # exp_scale acts as a tolerance parameter: larger values = more lenient rewards,
        # smaller values = sharper penalty for deviations from reference.
        reward_terms={
            # Imitation rewards
            "root_pos": {"exp_scale": 400.0, "weight": 1.0},  # Root position tolerance
            "root_quat": {
                "exp_scale": 4.0,
                "weight": 1.0,
            },  # Root orientation tolerance (degrees)
            "joints": {
                "exp_scale": 0.25,
                "weight": 1.0,
            },  # Joint angle tolerance (radians)
            "end_eff": {"exp_scale": 100.0, "weight": 1.0},  # End effector tolerance
            # Costs / regularizers
            "thorax_z_range": {"healthy_z_range": (-0.03, 0.1), "weight": 1.0},
            "control_cost": {"weight": 0.02},
            "control_diff_cost": {"weight": 0.02},
            "energy_cost": {"max_value": 50.0, "weight": 0.005},
        },
        termination_criteria={
            "root_too_far": {"max_distance": 0.5},
            "root_too_rotated": {"max_degrees": 15},
            "pose_error": {"max_l2_error": 20},
            "nan_termination": {},
        },
    )


class Imitation(fruitfly_base.FruitflyEnv):
    """Multi-clip imitation environment for fruitfly."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
        clips: Optional[ReferenceClips] = None,
    ) -> None:
        """
        Initialize the fruitfly imitation environment.

        Args:
            config: Configuration dictionary for the environment.
            config_overrides: Dictionary of configuration overrides.
            clips: Pre-loaded ReferenceClips object.
        """
        super().__init__(config, config_overrides)
        # self.add_fly(
        #     rescale_factor=self._config.rescale_factor,
        #     torque_actuators=self._config.torque_actuators,
        #     pos=(0, 0, 0),
        # )
        self._spec = mujoco.MjSpec.from_file(str(self._config.walker_xml_path))
        self._suffix = ""
        self.compile()

        if clips is not None:
            self.reference_clips = clips
        else:
            self.reference_clips = ReferenceClips(
                str(self._config.reference_data_path),
                self._config.clip_length,
                self._config.keep_clips_idx,
            )

        max_n_clips = self.reference_clips.joints.shape[0]
        if self._config.clip_set == "all":
            self._clip_set = max_n_clips
        elif isinstance(self._config.clip_set, (list, tuple, jp.ndarray, np.ndarray)):
            self._clip_set = jp.array(self._config.clip_set)
        else:
            raise ValueError(
                "config.clip_set must be 'all' or a list of clip indices. "
                f"Got {self._config.clip_set}."
            )

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        """
        Resets the environment state.

        Args:
            rng: JAX random number generator state.
            clip_idx: If provided, uses this clip index instead of sampling.
            start_frame: If provided, uses this start frame instead of sampling.

        Returns:
            mjx_env.State: The initial state of the environment after reset.
        """
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
        last_valid_frame = self._clip_length() - self._config.reference_length - 1
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

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        """Step the environment forward.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            mjx_env.State: The new state of the environment.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        last_valid_frame = self._clip_length() - self._config.reference_length - 1
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jp.astype(truncated, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        terminated = self._is_done(data, info, state.metrics)
        done = jp.logical_or(terminated, info["truncated"])
        reward = self._get_reward(data, info, state.metrics)

        # Handle nans during sim
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        current_frame = self._get_cur_frame(data, info)
        state.metrics["current_frame"] = jp.astype(current_frame, float)
        return state

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = collections.OrderedDict(
            task_obs=self._get_imitation_target(data, info),
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(state=obs)

    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        data = mjx.make_data(
            self.mj_model, impl=self._config.mujoco_impl, naconmax=self._config.naconmax
        )
        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)

        data = data.replace(qpos=reference.qpos)
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel=reference.qvel)

        data = mjx.forward(self.mjx_model, data)
        return data

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    def _clip_length(self):
        return self.reference_clips.joints.shape[1]

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        time_in_frames = data.time * self._config.mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the reference data at the current frame."""
        return self.reference_clips.at(
            clip=info["reference_clip"], frame=self._get_cur_frame(data, info)
        )

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the reference slice for observation."""
        return self.reference_clips.slice(
            clip=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.reference_length,
        )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Get the imitation target in egocentric coordinates."""
        reference = self._get_imitation_reference(data, info)

        root_pos = self.root_body(data).xpos
        root_quat = self.root_body(data).xquat
        root_targets = jax.vmap(
            lambda ref_pos: brax.math.rotate(ref_pos - root_pos, root_quat)
        )(reference.root_position)
        quat_targets = jax.vmap(
            lambda ref_quat: brax.math.relative_quat(ref_quat, root_quat)
        )(reference.root_quaternion)

        joint_targets = reference.joints - self._get_joint_angles(data)

        bodies_pos = self._get_bodies_pos(data, flatten=False)
        body_rel_pos = jp.array(
            [reference.body_xpos(name) - bodies_pos[name] for name in bodies_pos]
        )
        to_egocentric = jax.vmap(lambda diff_vec: brax.math.rotate(diff_vec, root_quat))
        body_targets = jax.vmap(to_egocentric)(body_rel_pos)

        return collections.OrderedDict(
            root=root_targets,
            quat=quat_targets,
            joint=joint_targets,
            body=body_targets,
        )

    # Rewards
    @_registry.reward("root_pos")
    def _root_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        distance = jp.linalg.norm(target.root_position - root_pos)
        metrics["root_pos_distance"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/root_pos"] = reward
        return reward

    @_registry.reward("root_quat")
    def _root_quat_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        rot_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        ang_dist_degrees = jp.rad2deg(rot_dist)
        metrics["root_angular_error"] = ang_dist_degrees
        reward = weight * jp.exp(-((ang_dist_degrees / exp_scale) ** 2) / 2)
        metrics["rewards/root_quat"] = reward
        return reward

    @_registry.reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(target.joints - joints)
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_registry.reward("joints_vel")
    def _joint_vels_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joint_vels = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(target.joints_velocity - joint_vels)
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    def _get_bodies_dist(self, data, info, metrics, bodies=consts.BODIES) -> float:
        target = self._get_current_target(data, info)
        body_pos = self._get_bodies_pos(data, flatten=False)
        total_dist_sqr = 0.0
        for body_name in bodies:
            dist_sqr = jp.sum((body_pos[body_name] - target.body_xpos(body_name)) ** 2)
            metrics["body_errors/" + body_name] = jp.sqrt(dist_sqr)
            total_dist_sqr += dist_sqr
        return jp.sqrt(total_dist_sqr)

    @_registry.reward("bodies_pos")
    def _body_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        total_dist = self._get_bodies_dist(data, info, metrics, consts.BODIES)
        metrics["body_errors/total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_registry.reward("end_eff")
    def _end_eff_reward(self, data, info, metrics, weight, exp_scale) -> float:
        total_dist = self._get_bodies_dist(data, info, metrics, consts.END_EFFECTORS)
        metrics["body_errors/end_eff_total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/end_eff"] = reward
        return reward

    @_registry.reward("thorax_z_range")
    def _thorax_z_range_reward(
        self, data, info, metrics, weight, healthy_z_range
    ) -> float:
        metrics["thorax_z"] = thorax_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(thorax_z >= min_z, thorax_z <= max_z)
        metrics["in_range"] = in_range.astype(float)
        reward = weight * in_range
        metrics["rewards/thorax_z_range"] = reward
        return reward

    @_registry.reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_sqr"] = ctrl_sqr = jp.sum(jp.square(info["action"]))
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = -cost
        return -cost

    @_registry.reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = jp.sum(
            jp.square(info["action"] - info["prev_action"])
        )
        cost = weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = -cost
        return -cost

    @_registry.reward("energy_cost")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        energy_use = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = -cost
        return -cost

    # Termination
    @_registry.termination("root_too_far")
    def _root_too_far(self, data, info, max_distance) -> bool:
        target = self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        distance = jp.linalg.norm(target.root_position - root_pos)
        return distance > max_distance

    @_registry.termination("root_too_rotated")
    def _root_too_rotated(self, data, info, max_degrees) -> bool:
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        ang_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        return ang_dist > jp.deg2rad(max_degrees)

    @_registry.termination("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        pose_error = jp.linalg.norm(target.joints - joints)
        return pose_error > max_l2_error

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        add_labels=False,
        termination_extra_frames=0,
    ) -> Sequence[np.ndarray]:
        """
        Renders a sequence of states (trajectory) with ghost fly.

        Args:
            trajectory: Sequence of environment states to render.
            height: Height of the rendered frames in pixels.
            width: Width of the rendered frames in pixels.
            camera: Camera name or index to use for rendering.
            scene_option: Additional scene rendering options.
            modify_scene_fns: Functions to modify the scene before rendering.
            add_labels: Whether to overlay labels on frames.
            termination_extra_frames: Number of extra frames on termination.

        Returns:
            Sequence[np.ndarray]: List of rendered frames.
        """
        # Create a new spec with a ghost fly
        spec = self._spec.copy()
        ghost_fly = mujoco.MjSpec.from_file(self._walker_xml_path)
        ghost_rescale = self._config.rescale_factor
        if ghost_rescale != 1.0:
            ghost_fly = utils.scale_spec(ghost_fly, ghost_rescale, root_body="thorax")
        for body in ghost_fly.worldbody.bodies:
            utils._recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])

        # Recursively disable collision for ALL ghost geoms (body tree + worldbody)
        def disable_collision_recursive(body):
            """Recursively disable collisions for all geoms in body tree."""
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0
            for child in body.bodies:
                disable_collision_recursive(child)

        # Disable on worldbody-level geoms (e.g., floor in ghost)
        for geom in ghost_fly.worldbody.geoms:
            geom.contype = 0
            geom.conaffinity = 0

        # Disable recursively on body tree
        for body in ghost_fly.worldbody.bodies:
            disable_collision_recursive(body)

        spawn_frame = spec.worldbody.add_frame(pos=(0, 0, 0.0), quat=(1, 0, 0, 0))
        spawn_body = spawn_frame.attach_body(
            ghost_fly.body("thorax"), "", suffix="-ghost"
        )

        mj_model_with_ghost = spec.compile()
        mj_model_with_ghost.vis.global_.offwidth = width
        mj_model_with_ghost.vis.global_.offheight = height
        mj_data_with_ghost = mujoco.MjData(mj_model_with_ghost)

        renderer = mujoco.Renderer(mj_model_with_ghost, height=height, width=width)
        if camera is None:
            camera = self._default_render_camera

        rendered_frames = []
        for i, state in enumerate(trajectory):
            time_in_frames = state.data.time * self._config.mocap_hz
            frame = jp.floor(time_in_frames + state.info["start_frame"]).astype(int)
            clip = state.info["reference_clip"]
            ref = self.reference_clips.at(clip=clip, frame=frame)

            mj_data_with_ghost.qpos = jp.concatenate((state.data.qpos, ref.qpos))
            mj_data_with_ghost.qvel = jp.concatenate((state.data.qvel, ref.qvel))
            mujoco.mj_forward(mj_model_with_ghost, mj_data_with_ghost)
            renderer.update_scene(
                mj_data_with_ghost, camera=camera, scene_option=scene_option
            )
            if modify_scene_fns is not None:
                modify_scene_fns[i](renderer.scene)
            rendered_frame = renderer.render()
            if add_labels:
                import cv2

                label = f"Clip {clip}"
                cv2.putText(
                    rendered_frame,
                    label,
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
                    for name in self._config.termination_criteria.keys():
                        if state.metrics["terminations/" + name] > 0:
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
