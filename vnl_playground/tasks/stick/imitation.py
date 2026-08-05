"""Imitation task for stick bug.

Multi-clip imitation environment where the stick bug must track motion
capture reference data. Uses the legacy H5 format (qpos/qvel/xpos/xquat)
loaded via the unified ReferenceClips class.
"""

import collections
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import brax.math
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks import math_utils
from vnl_playground.tasks.reference_clips import ReferenceClips
from vnl_playground.tasks.reward_registry import RewardRegistry

from .. import utils
from . import base as stick_base
from . import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.STICK_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        joints=consts.JOINTS,
        bodies=consts.BODIES,
        mujoco_impl="jax",
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="newton",
        iterations=5,
        ls_iterations=5,
        naconmax=256,
        njmax=600,
        noslip_iterations=0,
        torque_actuators=False,
        rescale_factor=1.5,  # Match H5 SCALE_FACTOR
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=50,
        clip_length=225,
        clip_set="all",
        reference_length=5,
        start_frame_range=[0, 44],
        qvel_init="zeros",
        keep_clips_idx=None,
        reward_terms={
            "root_pos": {"exp_scale": 0.035, "weight": 1.0},
            "root_quat": {"exp_scale": 40.0, "weight": 1.0},
            "joints": {"exp_scale": 1.4, "weight": 1.0},
            "joints_vel": {"exp_scale": 1.0, "weight": 1.0},
            "bodies_pos": {"exp_scale": 0.25, "weight": 1.0},
            "end_eff": {"exp_scale": 0.032, "weight": 1.0},
            "torso_z_range": {"healthy_z_range": (0.0, 0.1), "weight": 1.0},
            "control_cost": {"weight": 0.02},
            "control_diff_cost": {"weight": 0.02},
            "energy_cost": {"max_value": 50.0, "weight": 0.01},
        },
        termination_criteria={
            "root_too_far": {"max_distance": 0.05},
            "root_too_rotated": {"max_degrees": 120.0},
            "pose_error": {"max_l2_error": 4.5},
            "nan_termination": {},
        },
    )


_registry = RewardRegistry()


class Imitation(stick_base.StickBugEnv):
    """Multi-clip imitation environment for stick bug."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict[str, str | int | list[Any] | dict] | None = None,
        clips: ReferenceClips | None = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self.add_stick(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
            rgba=(0, 0.5, 0.5, 1),
        )
        self.compile()
        if clips is not None:
            self.reference_clips = clips
        else:
            self.reference_clips = ReferenceClips(
                self._config.reference_data_path,
                self._config.clip_length,
                self._config.keep_clips_idx,
                joint_names=self._config.joints,
                body_names=self._config.bodies,
            )
        max_n_clips = self.reference_clips.qpos.shape[0]
        if self._config.clip_set == "all":
            self._clip_set = max_n_clips
        elif isinstance(self._config.clip_set, (list, tuple, jp.ndarray, np.ndarray)):
            self._clip_set = jp.array(self._config.clip_set)
        else:
            raise ValueError(
                "config.clip_set must be 'all' or a list of clip indices."
                f" Got {self._config.clip_set}."
            )

        if (
            self.reference_clips._config is not None
            and "model" in self.reference_clips._config
            and self._config.rescale_factor
            != self.reference_clips._config["model"]["SCALE_FACTOR"]
        ):
            warnings.warn(
                f"Environment `rescale_factor` ({self._config.rescale_factor})"
                f" does not match the reference data `SCALE_FACTOR`"
                f" ({self.reference_clips._config['model']['SCALE_FACTOR']}).",
                stacklevel=2,
            )

    def reset(
        self,
        rng: jax.Array,
        clip_idx: int | None = None,
        start_frame: int | None = None,
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
            self.mj_model,
            impl=self._config.mujoco_impl,
            njmax=self._config.njmax,
            naconmax=self._config.naconmax,
        )
        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)
        _assert_all_are_prefix(
            reference.joint_names,
            self.get_joint_names(),
            "reference joints",
            "model joints",
        )
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
        return self.reference_clips.qpos.shape[1]

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        time_in_frames = data.time * self._config.mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        return self.reference_clips.at(
            clip=info["reference_clip"], frame=self._get_cur_frame(data, info)
        )

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        return self.reference_clips.slice(
            clip=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.reference_length,
        )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        reference = self._get_imitation_reference(data, info)

        root_pos = self.root_body(data).xpos
        root_quat = self.root_body(data).xquat
        root_targets = jax.vmap(
            lambda ref_pos: math_utils.world_point_to_local(
                ref_pos, root_pos, root_quat
            )
        )(reference.root_position)
        quat_targets = jax.vmap(
            lambda ref_quat: brax.math.relative_quat(ref_quat, root_quat)
        )(reference.root_quaternion)

        _assert_all_are_prefix(
            reference.joint_names,
            self.get_joint_names(),
            "reference joints",
            "model joints",
        )
        joint_targets = reference.joints - self._get_joint_angles(data)

        bodies_pos = self._get_bodies_pos(data, flatten=False)
        body_rel_pos = jp.array(
            [reference.body_xpos(name) - bodies_pos[name] for name in bodies_pos]
        )
        to_egocentric = jax.vmap(
            lambda diff_vec: math_utils.world_vector_to_local(diff_vec, root_quat)
        )
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
        reward = math_utils.gaussian_reward(distance, weight=weight, scale=exp_scale)
        metrics["rewards/root_pos"] = reward
        return reward

    @_registry.reward("root_quat")
    def _root_quat_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        ang_dist_degrees = math_utils.quaternion_angle(
            root_quat, target.root_quaternion, degrees=True
        )
        metrics["root_angular_error"] = ang_dist_degrees
        reward = math_utils.gaussian_reward(
            ang_dist_degrees, weight=weight, scale=exp_scale
        )
        metrics["rewards/root_quat"] = reward
        return reward

    @_registry.reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(target.joints - joints)
        metrics["joint_l2_error"] = distance
        reward = math_utils.gaussian_reward(distance, weight=weight, scale=exp_scale)
        metrics["rewards/joints"] = reward
        return reward

    @_registry.reward("joints_vel")
    def _joint_vels_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joint_vels = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(target.joints_velocity - joint_vels)
        metrics["joint_vel_l2_error"] = distance
        reward = math_utils.gaussian_reward(distance, weight=weight, scale=exp_scale)
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
        reward = math_utils.gaussian_reward(total_dist, weight=weight, scale=exp_scale)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_registry.reward("end_eff")
    def _end_eff_reward(self, data, info, metrics, weight, exp_scale) -> float:
        total_dist = self._get_bodies_dist(data, info, metrics, consts.END_EFFECTORS)
        metrics["body_errors/end_eff_total"] = total_dist
        reward = math_utils.gaussian_reward(total_dist, weight=weight, scale=exp_scale)
        metrics["rewards/end_eff"] = reward
        return reward

    @_registry.reward("torso_z_range")
    def _torso_z_range_reward(
        self, data, info, metrics, weight, healthy_z_range
    ) -> float:
        metrics["body_z"] = body_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(body_z >= min_z, body_z <= max_z)
        metrics["in_range"] = in_range.astype(float)
        reward = weight * in_range
        metrics["rewards/torso_z_range"] = reward
        return reward

    @_registry.reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_sqr"] = ctrl_sqr = math_utils.squared_l2_norm(info["action"])
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = -cost
        return -cost

    @_registry.reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = math_utils.squared_l2_norm(
            info["action"] - info["prev_action"]
        )
        cost = weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = -cost
        return -cost

    @_registry.reward("energy_cost")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        energy_use = math_utils.absolute_actuator_power(data.qvel, data.qfrc_actuator)
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
        ang_dist = math_utils.quaternion_angle(root_quat, target.root_quaternion)
        return ang_dist > jp.deg2rad(max_degrees)

    @_registry.termination("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        pose_error = jp.linalg.norm(target.joints - joints)
        return pose_error > max_l2_error

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        return jp.any(jp.isnan(data.qpos))

    def render(
        self,
        trajectory: list[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
        scene_option: mujoco.MjvOption | None = None,
        modify_scene_fns: Sequence[Callable[[mujoco.MjvScene], None]] | None = None,
        add_labels=False,
        termination_extra_frames=0,
        render_ghost: bool = True,
    ) -> Sequence[np.ndarray]:
        """Renders a sequence of states with optional ghost stick bug."""
        if render_ghost:
            spec = self._spec.copy()
            ghost_stick = mujoco.MjSpec.from_file(self._walker_xml_path)
            ghost_rescale = self._config.rescale_factor
            if (
                self.reference_clips._config is not None
                and "model" in self.reference_clips._config
            ):
                ghost_rescale = self.reference_clips._config["model"]["SCALE_FACTOR"]
            if ghost_rescale != 1.0:
                ghost_stick = utils.scale_spec(
                    ghost_stick, ghost_rescale, root_body="reference_base"
                )
            for body in ghost_stick.worldbody.bodies:
                utils._recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])
            spawn_frame = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
            spawn_frame.attach_body(
                ghost_stick.body("reference_base"), "", suffix="-ghost"
            )
            mj_model = spec.compile()
        else:
            mj_model = self.mj_model
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        mj_data = mujoco.MjData(mj_model)

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        if camera is None:
            camera = self._default_render_camera

        rendered_frames = []
        for i, state in enumerate(trajectory):
            time_in_frames = state.data.time * self._config.mocap_hz
            frame = jp.floor(time_in_frames + state.info["start_frame"]).astype(int)
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
            rendered_frames.append(rendered_frame)
            if state.done:
                for t in range(termination_extra_frames):
                    rel_t = t / termination_extra_frames
                    fade_factor = 1 / (1 + np.exp(10 * (rel_t - 0.5)))
                    faded_frame = (rendered_frame * fade_factor).astype(np.uint8)
                    rendered_frames.append(faded_frame)
        return rendered_frames


def _assert_all_are_prefix(a, b, a_name="a", b_name="b"):
    if isinstance(a, map):
        a = list(a)
    if isinstance(b, map):
        b = list(b)
    if len(a) != len(b):
        raise AssertionError(
            f"{a_name} has length {len(a)} but {b_name} has length {len(b)}."
        )
    for a_el, b_el in zip(a, b):
        if not b_el.startswith(a_el):
            raise AssertionError(
                f"Comparing {a_name} and {b_name}. Expected {a_el} to match {b_el}."
            )
