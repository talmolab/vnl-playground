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

from .. import utils
from . import base as rodent_base
from . import consts
from .reference_clips import ReferenceClips


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        mujoco_impl="jax",
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="newton",
        iterations=5,
        ls_iterations=5,
        nconmax=256,
        njmax=128,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=50,
        clip_length=250,
        clip_set="all",  # NOTE: Charles added keep_clips_idx which basically is the same as this for indices to reduce memory usage
        reference_length=5,
        start_frame_range=[0, 44],
        qvel_init="zeros",
        keep_clips_idx=None,
        reward_terms={
            # Imitation rewards
            "root_pos": {"exp_scale": 0.035, "weight": 1.0},  # Meters
            "root_quat": {"exp_scale": 20.0, "weight": 1.0},  # Degrees
            "joints": {"exp_scale": 1.4, "weight": 1.0},  # Joint-space L2 distance
            "joints_vel": {
                "exp_scale": 1.0,
                "weight": 1.0,
            },  # Joint velocity-space L2 distance
            "bodies_pos": {
                "exp_scale": 0.25,
                "weight": 1.0,
            },  # Distance in concatenated euclidean space
            "end_eff": {
                "exp_scale": 0.032,
                "weight": 1.0,
            },  # Distance in concatenated euclidean space
            # Costs / regularizers
            "torso_z_range": {"healthy_z_range": (0.0325, 0.5), "weight": 1.0},
            "control_cost": {"weight": 0.02},
            "control_diff_cost": {"weight": 0.02},
            "energy_cost": {"max_value": 50.0, "weight": 0.01},
        },
        termination_criteria={
            "root_too_far": {"max_distance": 0.1},  # Meters
            "root_too_rotated": {"max_degrees": 60.0},  # Degrees
            "pose_error": {"max_l2_error": 4.5},  # Joint-space L2 distance
            "nan_termination": {},
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
        clips: Optional[ReferenceClips] = None,
    ) -> None:
        """
        Initialize the rodent imitation environment.
        Args:
            config (config_dict.ConfigDict, optional): Configuration dictionary for the environment.
                Defaults to `default_config()`.
            config_overrides (optional):
                Dictionary of configuration overrides.
            clips (optional):
                Pre-loaded ReferenceClips object. If provided, it overrides
                loading from `config.reference_data_path`.
        """
        super().__init__(config, config_overrides)
        self.add_rodent(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
            rgba=(0, 0.5, 0.5, 1),  # Teal color
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
            # Only use clips whose types match the specified set of
            # clips (e.g. "Walk", "LGroom")
            (self._clip_set,) = jp.where(
                self._config.clip_set == self.reference_clips.clip_names
            )
        else:
            raise ValueError(
                "config.clip_set must be 'all', a list of clip indices"
                f" or a behavior name. Got {self._config.clip_set}."
            )

        if (
            self._config.rescale_factor
            != self.reference_clips._config["model"]["SCALE_FACTOR"]
        ):
            warnings.warn(
                f"Environment `rescale_factor` ({self._config.rescale_factor})"
                f" does not match the reference data `SCALE_FACTOR`"
                f" ({self.reference_clips._config['model']['SCALE_FACTOR']})."
            )

    def reset(
            self, 
            rng: jax.Array, 
            clip_idx: Optional[int] = None, 
            start_frame: Optional[int] = None
        ) -> mjx_env.State:
        """
        Resets the environment state: draws a new reference clip and initializes the rodent's pose to match.
        Args:
            rng (jax.Array): JAX random number generator state.
            clip_idx (optional): If provided, uses this clip index instead of sampling randomly.
        Returns:
            mjx_env.State: The initial state of the environment after reset.
        """

        start_rng, clip_rng = jax.random.split(rng)
        if clip_idx is None:
            clip_idx = jax.random.choice(clip_rng, self._clip_set)
        if start_frame is None:
            start_frame = jax.random.randint(start_rng, (), *self._config.start_frame_range)
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
            state (mjx_env.State): Current environment state.
            action (jax.Array): Action to apply.
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

        # Handle nans during sim with this in addition to termination on nans in data
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
            proprioception=self._get_proprioception(data, info, flatten=False),
            imitation_target=self._get_imitation_target(data, info),
        )

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self,
                data,
                info,
                metrics,
                imitation_reference=self._get_current_target(data, info),
                **kwargs,
            )
        return net_reward

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

    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
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
        """Get the reference slice that is to be part of the observation."""
        return self.reference_clips.slice(
            clip=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.reference_length,
        )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Get the imitation target, i.e. the imitation reference transformed to
        egocentric coordinates."""
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
        to_egocentric = jax.vmap(lambda diff_vec: brax.math.rotate(diff_vec, root_quat))
        body_targets = jax.vmap(to_egocentric)(body_rel_pos)

        return collections.OrderedDict(
            root=root_targets,
            quat=quat_targets,
            joint=joint_targets,
            body=body_targets,
        )

    # Rewards
    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("root_pos")
    def _root_pos_reward(
        self, data, info, metrics, imitation_reference, weight, exp_scale
    ) -> float:
        root_pos = self.root_body(data).xpos
        distance = jp.linalg.norm(imitation_reference.root_position - root_pos)
        metrics["root_pos_distance"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/root_pos"] = reward
        return reward

    @_named_reward("root_quat")
    def _root_quat_reward(
        self, data, info, metrics, imitation_reference, weight, exp_scale
    ) -> float:
        """`exp_scale` is in degrees."""
        root_quat = self.root_body(data).xquat
        quat_dist = (
            2.0 * jp.dot(root_quat, imitation_reference.root_quaternion) ** 2 - 1.0
        )
        rot_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        ang_dist_degrees = jp.rad2deg(rot_dist)
        metrics["root_angular_error"] = ang_dist_degrees
        reward = weight * jp.exp(-((ang_dist_degrees / exp_scale) ** 2) / 2)
        metrics["rewards/root_quat"] = reward
        return reward

    @_named_reward("joints")
    def _joints_reward(
        self, data, info, metrics, imitation_reference, weight, exp_scale
    ) -> float:
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(imitation_reference.joints - joints)
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_named_reward("joints_vel")
    def _joint_vels_reward(
        self, data, info, metrics, imitation_reference, weight, exp_scale
    ) -> float:
        target = self._get_current_target(data, info)
        joint_vels = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(imitation_reference.joints_velocity - joint_vels)
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    def _get_bodies_dist(
        self, data, info, metrics, imitation_reference, bodies=consts.BODIES
    ) -> float:
        body_pos = self._get_bodies_pos(data, flatten=False)
        total_dist_sqr = 0.0
        for body_name in bodies:
            dist_sqr = jp.sum(
                (body_pos[body_name] - imitation_reference.body_xpos(body_name)) ** 2
            )
            metrics["body_errors/" + body_name] = jp.sqrt(dist_sqr)
            total_dist_sqr += dist_sqr
        return jp.sqrt(total_dist_sqr)

    @_named_reward("bodies_pos")
    def _body_pos_reward(
        self, data, info, metrics, imitation_reference, weight, exp_scale
    ) -> float:
        total_dist = self._get_bodies_dist(
            data, info, metrics, imitation_reference, consts.BODIES
        )
        metrics["body_errors/total"] = total_dist
        reward = weight * jp.exp(-(((total_dist / exp_scale)) ** 2) / 2)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_named_reward("end_eff")
    def _end_eff_reward(
        self, data, info, metrics, imitation_reference, weight, exp_scale
    ) -> float:
        total_dist = self._get_bodies_dist(
            data, info, metrics, imitation_reference, consts.END_EFFECTORS
        )
        metrics["body_errors/end_eff_total"] = total_dist
        reward = weight * jp.exp(-(((total_dist / exp_scale)) ** 2) / 2)
        metrics["rewards/end_eff"] = reward
        return reward

    @_named_reward("torso_z_range")
    def _torso_z_range_reward(
        self, data, info, metrics, imitation_reference, weight, healthy_z_range
    ) -> float:
        metrics["torso_z"] = torso_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        metrics["in_range"] = in_range.astype(float)
        reward = weight * in_range
        metrics["rewards/torso_z_range"] = reward
        return reward

    @_named_reward("control_cost")
    def _control_cost(self, data, info, metrics, imitation_reference, weight) -> float:
        metrics["ctrl_sqr"] = ctrl_sqr = jp.sum(jp.square(info["action"]))
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = cost
        return -cost

    @_named_reward("control_diff_cost")
    def _control_diff_cost(
        self, data, info, metrics, imitation_reference, weight
    ) -> float:
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = jp.sum(
            jp.square(info["action"] - info["prev_action"])
        )
        cost = weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = cost
        return -cost

    @_named_reward("energy_cost")
    def _energy_cost(
        self, data, info, metrics, imitation_reference, weight, max_value
    ) -> float:
        energy_use = jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.qfrc_actuator[6:]))
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = cost
        return -cost

    @_named_reward("jerk_cost")
    def _jerk_cost(
        self, data, info, weight, metrics, imitation_reference, window_len
    ) -> float:
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
        # Handle nans during sim by resetting env
        flattened_vals, _ = jax.flatten_util.ravel_pytree(data)
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
        Renders a sequence of states (trajectory). The video includes the imitation
        target as a white transparent "ghost".

        Args:
            trajectory (List[mjx_env.State]): Sequence of environment states to render.
            height (int, optional): Height of the rendered frames in pixels. Defaults to 240.
            width (int, optional): Width of the rendered frames in pixels. Defaults to 320.
            camera (str, optional): Camera name or index to use for rendering.
            scene_option (mujoco.MjvOption, optional): Additional scene rendering options.
            modify_scene_fns (Sequence[Callable[[mujoco.MjvScene], None]], optional):
                Sequence of functions to modify the scene before rendering each frame.
                Defaults to None.
            add_labels (bool, optional): Whether to overlay clip and termination cause
                labels on frames. Defaults to False.
            termination_extra_frames (int, optional): If larger than 0, then repeat the
                frame triggering the termination this number of times. This gives
                a freeze-on-done effect that may help debug termination criteria.
                Additionally, a simple fade-out effect is applied during those frames
                to smooth the tranisition between clips. If this is larger than 0, the
                number of returned frames might be larger than `len(trajectory)`.
        Returns:
            Sequence[np.ndarray]: List of rendered frames as numpy arrays.
        """
        # Create a new spec with a ghost, without modifying the existing one
        spec = self._spec.copy()
        ghost_rodent = mujoco.MjSpec.from_file(self._walker_xml_path)
        ghost_rescale = self.reference_clips._config["model"]["SCALE_FACTOR"]
        if ghost_rescale != 1.0:
            ghost_rodent = utils.dm_scale_spec(ghost_rodent, ghost_rescale)
        for body in ghost_rodent.worldbody.bodies:
            utils._recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])
        spawn_site = spec.worldbody.add_frame(pos=(0, 0, 0.05), quat=(1, 0, 0, 0))
        spawn_body = spawn_site.attach_body(ghost_rodent.worldbody, "", suffix="-ghost")
        spawn_body.add_freejoint()
        mj_model_with_ghost = spec.compile()
        mj_model_with_ghost.vis.global_.offwidth = width
        mj_model_with_ghost.vis.global_.offheight = height
        mj_data_with_ghost = mujoco.MjData(mj_model_with_ghost)

        renderer = mujoco.Renderer(mj_model_with_ghost, height=height, width=width)
        if camera is None:
            camera = -1

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

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(jax.flatten_util.ravel_pytree(obs_size["proprioception"])[0])

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(jax.flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)

    def verify_reference_data(self, atol: float = 5e-3) -> bool:
        """A set of non-exhaustive sanity checks that the reference data found in
        `config.REFERENCE_DATA_PATH` matches the environment's model. Most
        importantly, it verifies that the global coordinates of the
        body parts (xpos) match those the model produces when initialized to
        the corresponding qpos from the file. This can catch issues with nested
        free joints, mismatched joint orders, and incorrect scaling of the model
        but it's not exhaustive). This current implementation tests all
        frames of all clips in the reference data, and so is rather slow and
        does not have to be run every time.

        Args:
            atol (float): Absolute floating-point tolerance for the checks.
                          Defaults to 5e-3, because this seems to be the precision
                          of the reference data (TODO: why are there several mm
                          of error?).
        Returns:
            bool: True if all checks passed, False if any check failed.
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
            for body_name, body_pos in body_pos.items():
                checks[f"body_xpos/{body_name}"] = jp.allclose(
                    body_pos, reference.body_xpos(body_name), atol=atol
                )
            if self._config.qvel_init == "reference":
                checks["joints_ang_vel"] = jp.allclose(
                    self._get_joint_ang_vels(data), reference.joints, atol=atol
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
                    f"Clip index {clip} is out of range. Reference"
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
