"""Multi-clip imitation task for Fauna Robotics Sprout humanoid.

Follows the same pattern as the rodent imitation task (rodent/imitation.py),
adapted for the Sprout humanoid's body structure, sensors, and physical
dimensions.

The Sprout is initialized to a reference pose from motion capture data and
must track the reference trajectory. Rewards encourage matching root position,
root orientation, joint angles, joint velocities, body positions, and end
effector positions. Termination occurs if the pose diverges too far from the
reference.
"""

import collections
import warnings
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

from .. import utils
from . import base as sprout_base
from . import consts
from vnl_playground.tasks.reference_clips import ReferenceClips


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.SPROUT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        mujoco_impl="jax",
        sim_dt=0.002,
        ctrl_dt=0.02,
        solver="newton",
        iterations=5,
        ls_iterations=5,
        naconmax=16 * 512,
        njmax=400,
        noslip_iterations=0,
        ccd_iterations=75,
        torque_actuators=True,
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=50,
        clip_length=250,
        clip_set="all",
        reference_length=5,
        start_frame_range=[0, 44],
        qvel_init="zeros",
        keep_clips_idx=None,
        reward_terms={
            # Imitation rewards
            "root_pos": {"exp_scale": 0.035, "weight": 1.0},
            "root_quat": {"exp_scale": 20.0, "weight": 1.0},
            "joints": {"exp_scale": 1.4, "weight": 1.0},
            "joints_vel": {"exp_scale": 1.0, "weight": 1.0},
            "bodies_pos": {"exp_scale": 0.25, "weight": 1.0},
            "end_eff": {"exp_scale": 0.032, "weight": 1.0},
            # Costs / regularizers
            "torso_z_range": {"healthy_z_range": (0.35, 0.75), "weight": 1.0},
            "control_cost": {"weight": 0.02},
            "control_diff_cost": {"weight": 0.02},
            "energy_cost": {"max_value": 50.0, "weight": 0.01},
        },
        termination_criteria={
            "root_too_far": {"max_distance": 0.1},
            "root_too_rotated": {"max_degrees": 60.0},
            "pose_error": {"max_l2_error": 4.5},
            "nan_termination": {},
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


class Imitation(sprout_base.SproutEnv):
    """Multi-clip imitation environment for Sprout humanoid."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
        clips: Optional[ReferenceClips] = None,
    ) -> None:
        """Initialize the Sprout imitation environment.

        Args:
            config: Configuration dictionary for the environment.
            config_overrides: Dictionary of configuration overrides.
            clips: Pre-loaded ReferenceClips object. If provided, it overrides
                loading from ``config.reference_data_path``.
        """
        super().__init__(config, config_overrides)
        self.add_sprout(
            torque_actuators=self._config.torque_actuators,
        )
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
        """Reset the environment: draw a new reference clip and initialize pose.

        Args:
            rng: JAX random number generator state.
            clip_idx: If provided, uses this clip index instead of sampling.
            start_frame: If provided, uses this start frame instead of sampling.

        Returns:
            The initial state of the environment after reset.
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
            The new state of the environment.
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
        return collections.OrderedDict(
            imitation_target=self._get_imitation_target(data, info),
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics) -> bool:
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            terminated = termination_fcn(self, data, info, **kwargs)
            any_terminated = jp.logical_or(any_terminated, terminated)
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

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
        elif self._config.qvel_init == "noise":
            raise NotImplementedError("qvel_init='noise' is not yet implemented.")
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
        """Get the reference data at the current frame."""
        return self.reference_clips.at(
            clip=info["reference_clip"], frame=self._get_cur_frame(data, info)
        )

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the reference slice that is part of the observation."""
        return self.reference_clips.slice(
            clip=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.reference_length,
        )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Get the imitation target transformed to egocentric coordinates."""
        reference = self._get_imitation_reference(data, info)

        root_pos = self.root_body(data).xpos
        root_quat = self.root_body(data).xquat
        root_targets = jax.vmap(
            lambda ref_pos: brax.math.rotate(ref_pos - root_pos, root_quat)
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
            lambda diff_vec: brax.math.rotate(diff_vec, root_quat)
        )
        body_targets = jax.vmap(to_egocentric)(body_rel_pos)

        return collections.OrderedDict(
            root=root_targets,
            quat=quat_targets,
            joint=joint_targets,
            body=body_targets,
        )

    # ---- Reward functions ----

    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("root_pos")
    def _root_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        distance = jp.linalg.norm(target.root_position - root_pos)
        metrics["root_pos_distance"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/root_pos"] = reward
        return reward

    @_named_reward("root_quat")
    def _root_quat_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """`exp_scale` is in degrees."""
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        rot_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        ang_dist_degrees = jp.rad2deg(rot_dist)
        metrics["root_angular_error"] = ang_dist_degrees
        reward = weight * jp.exp(-((ang_dist_degrees / exp_scale) ** 2) / 2)
        metrics["rewards/root_quat"] = reward
        return reward

    @_named_reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(target.joints - joints)
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_named_reward("joints_vel")
    def _joint_vels_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joint_vels = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(target.joints_velocity - joint_vels)
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    def _get_bodies_dist(
        self, data, info, metrics, bodies=consts.BODIES
    ) -> float:
        target = self._get_current_target(data, info)
        body_pos = self._get_bodies_pos(data, flatten=False)
        total_dist_sqr = 0.0
        for body_name in bodies:
            dist_sqr = jp.sum((body_pos[body_name] - target.body_xpos(body_name)) ** 2)
            metrics["body_errors/" + body_name] = jp.sqrt(dist_sqr)
            total_dist_sqr += dist_sqr
        return jp.sqrt(total_dist_sqr)

    @_named_reward("bodies_pos")
    def _body_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        total_dist = self._get_bodies_dist(data, info, metrics, consts.BODIES)
        metrics["body_errors/total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_named_reward("end_eff")
    def _end_eff_reward(self, data, info, metrics, weight, exp_scale) -> float:
        total_dist = self._get_bodies_dist(data, info, metrics, consts.END_EFFECTORS)
        metrics["body_errors/end_eff_total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/end_eff"] = reward
        return reward

    @_named_reward("torso_z_range")
    def _torso_z_range_reward(
        self, data, info, metrics, weight, healthy_z_range
    ) -> float:
        metrics["torso_z"] = torso_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        metrics["in_range"] = in_range.astype(float)
        reward = weight * in_range
        metrics["rewards/torso_z_range"] = reward
        return reward

    @_named_reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_sqr"] = ctrl_sqr = jp.sum(jp.square(info["action"]))
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = -cost
        return -cost

    @_named_reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = jp.sum(
            jp.square(info["action"] - info["prev_action"])
        )
        cost = weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = -cost
        return -cost

    @_named_reward("energy_cost")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        energy_use = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = -cost
        return -cost

    # ---- Termination criteria ----

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

    @_named_termination_criterion("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    # ---- Rendering ----

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
        render_ghost: bool = True,
    ) -> Sequence[np.ndarray]:
        """Render a trajectory with an optional ghost showing the imitation target.

        Args:
            trajectory: Sequence of environment states to render.
            height: Height of rendered frames in pixels.
            width: Width of rendered frames in pixels.
            camera: Camera name or index for rendering.
            scene_option: Additional scene rendering options.
            modify_scene_fns: Functions to modify the scene before each frame.
            add_labels: Whether to overlay clip and termination labels.
            termination_extra_frames: Extra frames to render on termination
                (freeze-on-done with fade-out effect).
            render_ghost: Whether to render a transparent ghost of the target.

        Returns:
            List of rendered frames as numpy arrays.
        """
        if render_ghost:
            spec = self._spec.copy()
            ghost_sprout = mujoco.MjSpec.from_file(self._walker_xml_path)
            for body in ghost_sprout.body("torso_link").bodies:
                utils._recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])
            # Also recolour the root body's own geoms
            for geom in ghost_sprout.body("torso_link").geoms:
                utils._recolour_geom(geom, rgba=[1.0, 1.0, 1.0, 0.2])
            spawn_site = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
            spawn_body = spawn_site.attach_body(
                ghost_sprout.body("torso_link"), "", suffix="-ghost"
            )
            spawn_body.add_freejoint()
            mj_model = spec.compile()
        else:
            mj_model = self.mj_model

        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        mj_data = mujoco.MjData(mj_model)

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        if scene_option is None:
            scene_option = mujoco.MjvOption()
        scene_option.geomgroup[1] = False  # Hide collision geoms (red)
        if camera is None:
            camera = -1

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

            if add_labels:
                import cv2

                behavior_label = self.reference_clips.clip_names[clip]
                label = f"Clip {clip} ({behavior_label})"
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
                    fade_factor = 1 / (
                        1 + np.exp(10 * (rel_t - 0.5))
                    )  # Logistic fade-out
                    faded_frame = (rendered_frame * fade_factor).astype(np.uint8)
                    rendered_frames.append(faded_frame)

        return rendered_frames

    # ---- Observation size properties ----

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["proprioception"])[0])

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)

    def verify_reference_data(self, atol: float = 5e-3) -> bool:
        """Verify that reference data matches the environment's model.

        Tests all frames of all clips to check that body positions (xpos) match
        those produced by the model when initialized to the corresponding qpos.

        Args:
            atol: Absolute floating-point tolerance for checks.

        Returns:
            True if all checks passed, False if any check failed.
        """

        def test_frame(clip_idx: int, frame: int) -> dict[str, bool]:
            data = self._reset_data(clip_idx, frame)
            reference = self.reference_clips.at(clip=clip_idx, frame=frame)
            checks = collections.OrderedDict()
            checks["root_pos"] = jp.allclose(
                self.root_body(data).xpos, reference.root_position, atol=atol
            )
            checks["root_quat"] = jp.allclose(
                self.root_body(data).xquat, reference.root_quaternion, atol=atol
            )
            checks["joints"] = jp.allclose(
                self._get_joint_angles(data), reference.joints, atol=atol
            )
            body_pos = self._get_bodies_pos(data, flatten=False)
            for body_name, body_pos_val in body_pos.items():
                checks[f"body_xpos/{body_name}"] = jp.allclose(
                    body_pos_val, reference.body_xpos(body_name), atol=atol
                )
            if self._config.qvel_init == "reference":
                checks["joints_ang_vel"] = jp.allclose(
                    self._get_joint_ang_vels(data),
                    reference.joints_velocity,
                    atol=atol,
                )
            return checks

        @jax.jit
        def test_clip(clip_idx: int):
            return jax.vmap(test_frame, in_axes=(None, 0))(
                clip_idx, jp.arange(self._clip_length())
            )

        _assert_all_are_prefix(
            self.reference_clips.joint_names,
            self.get_joint_names(),
            "reference joints",
            "model joints",
        )
        if isinstance(self._clip_set, int):
            clip_idxs = jp.arange(self._clip_set)
        else:
            clip_idxs = self._clip_set

        any_failed = False
        for clip in clip_idxs:
            if clip < 0 or clip >= self.reference_clips.qpos.shape[0]:
                raise ValueError(
                    f"Clip index {clip} is out of range. Reference "
                    f"data has {self.reference_clips.qpos.shape[0]} clips."
                )
            test_result = test_clip(clip)

            for name, result in test_result.items():
                n_failed = jp.sum(np.logical_not(result))
                if n_failed > 0:
                    first_failed_frame = jp.argmax(np.logical_not(result))
                    clip_label = self.reference_clips.clip_names[clip]
                    warnings.warn(
                        f"Reference data verification failed for {n_failed} frames"
                        f" for check '{name}' for clip {clip} ({clip_label})."
                        f" First failure at frame {first_failed_frame}."
                    )
                    any_failed = True
        return not any_failed


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
