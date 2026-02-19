"""Mouse arm imitation environment for motion tracking."""

import collections
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.mouse import consts
from vnl_playground.tasks.mouse.base import (
    MouseBaseEnv,
    default_config as base_default_config,
)
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips


def default_config() -> config_dict.ConfigDict:
    """Default configuration for mouse arm imitation.

    Returns:
        config_dict.ConfigDict: Configuration with reference data settings,
            walker body names, reward terms, and termination criteria.
    """
    cfg = base_default_config()
    # Reference data settings
    cfg.reference_data_path = str(consts.MOUSE_REFERENCE_DATA_PATH)
    cfg.mocap_hz = 200  # Frame rate of reference data
    cfg.clip_length = 100  # Frames per clip
    cfg.clip_set = "all"  # Which clips to use
    cfg.reference_length = 1  # Frames of future reference to include in observation
    cfg.start_frame_range = [0, 1]  # Always start at frame 0
    cfg.qvel_init = "reference"  # How to initialize velocities: "zeros", "reference"
    cfg.keep_clips_idx = None  # Indices of clips to keep (None = all)

    # Walker-specific settings (can be overridden via config)
    cfg.tracked_bodies = ["scapula", "humerus", "ulna", "radius", "wrist_body"]
    cfg.end_effector = "wrist_body"

    # Reward terms
    cfg.reward_terms = {
        "joints": {"exp_scale": 0.2, "weight": 5.0},
        "joints_vel": {"exp_scale": 0.2, "weight": 0.5},
        "wrist_pos": {"exp_scale": 0.005, "weight": 0.1},  # End effector tracking
        "bodies_pos": {"exp_scale": 0.01, "weight": 0.1},
        "control_cost": {"weight": 0.01},
        "control_diff_cost": {"weight": 0.00},
    }

    # Termination criteria
    cfg.termination_criteria = {
        "pose_error": {"max_l2_error": 3.0},
        "nan_termination": {},
    }

    return cfg


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


class MouseImitation(MouseBaseEnv):
    """Multi-clip imitation environment for mouse arm."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
        clips: Optional[MouseReferenceClips] = None,
    ) -> None:
        """Initialize the mouse arm imitation environment.

        Args:
            config: Configuration dictionary.
            config_overrides: Optional overrides for config fields.
            clips: Pre-loaded MouseReferenceClips. If None, loads from config path.
        """
        super().__init__(config, config_overrides)

        # Add mouse arm (no freejoint - fixed base)
        self.add_mouse(freejoint=False, pos=(0.0, 0.0, 0.0))
        self.compile()

        # Load reference clips
        if clips is not None:
            self.reference_clips = clips
        else:
            self.reference_clips = MouseReferenceClips(
                self._config.reference_data_path,
                self._config.clip_length,
                self._config.keep_clips_idx,
            )

        # Recompute xpos/xquat using simulation model for consistency
        # (reference data may have been generated with a different model)
        self.reference_clips.recompute_kinematics(self._mj_model)

        # Setup clip set
        max_n_clips = self.reference_clips.qpos.shape[0]
        if self._config.clip_set == "all":
            self._clip_set = max_n_clips
        elif isinstance(self._config.clip_set, (list, tuple, jp.ndarray, np.ndarray)):
            self._clip_set = jp.array(self._config.clip_set)
        else:
            raise ValueError(
                f"config.clip_set must be 'all' or a list of clip indices. "
                f"Got {self._config.clip_set}."
            )

        # Cache body IDs (from config for flexibility)
        self._body_ids = {}
        for name in self._config.tracked_bodies:
            try:
                # Names have "-mouse" suffix after add_mouse()
                self._body_ids[name] = self._mj_model.body(name + "-mouse").id
            except KeyError:
                print(f"Warning: body {name}-mouse not found in model")

        self._wrist_body_id = self._body_ids.get(self._config.end_effector, None)

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        """Reset the environment to a reference pose.

        Args:
            rng: JAX random key.
            clip_idx: Specific clip index (if None, sampled randomly).
            start_frame: Specific start frame (if None, sampled randomly).

        Returns:
            Initial environment state.
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

        # Check if we've reached the end of the clip
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
            New environment state.
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

        # Handle NaNs
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
        """Build observation dictionary.

        Args:
            data: MJX simulation data.
            info: Episode info dict containing reference_clip and start_frame.

        Returns:
            Mapping[str, Any]: OrderedDict with 'task_obs' and
                'proprioception' entries.
        """
        return collections.OrderedDict(
            task_obs=self._get_imitation_target(data, info),
            proprioception=self._get_proprioception(data, info),
        )

    def _get_proprioception(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> jp.ndarray:
        """Get proprioceptive observation (joint positions and velocities).

        Args:
            data: MJX simulation data.
            info: Episode info dict.

        Returns:
            jp.ndarray: Concatenated qpos and qvel.
        """
        return jp.concatenate(
            [
                data.qpos,  # Joint positions
                data.qvel,  # Joint velocities
            ]
        )

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        """Compute total reward from all registered reward terms.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict; individual reward values are written here.

        Returns:
            float: Sum of all weighted reward terms.
        """
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            if name in _REWARD_FCN_REGISTRY:
                net_reward += _REWARD_FCN_REGISTRY[name](
                    self, data, info, metrics, **kwargs
                )
        return net_reward

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics) -> bool:
        """Check if episode should terminate based on registered criteria.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict; termination flags are written here.

        Returns:
            bool: True if any termination criterion is triggered.
        """
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            if name in _TERMINATION_FCN_REGISTRY:
                termination_fcn = _TERMINATION_FCN_REGISTRY[name]
                terminated = termination_fcn(self, data, info, **kwargs)
                any_terminated = jp.logical_or(any_terminated, terminated)
                metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        """Reset simulation data to match reference at given clip/frame.

        Args:
            clip_idx: Index of the reference clip.
            start_frame: Frame index within the clip to initialize from.

        Returns:
            mjx.Data: Simulation data initialized to the reference pose.
        """
        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
        )

        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)
        data = data.replace(qpos=reference.qpos)

        if self._config.qvel_init == "zeros":
            data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel=reference.qvel)

        data = mjx.forward(self.mjx_model, data)
        return data

    def null_action(self) -> jp.ndarray:
        """Return zero action.

        Returns:
            jp.ndarray: Zero array of shape (action_size,).
        """
        return jp.zeros(self.action_size)

    def _clip_length(self):
        """Get number of frames per clip.

        Returns:
            int: Number of frames in each reference clip.
        """
        return self.reference_clips.qpos.shape[1]

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        """Get current frame index based on simulation time.

        Args:
            data: MJX simulation data (uses data.time).
            info: Episode info dict containing 'start_frame'.

        Returns:
            int: Current frame index in the reference clip.
        """
        time_in_frames = data.time * self._config.mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> MouseReferenceClips:
        """Get reference data at the current frame.

        Args:
            data: MJX simulation data.
            info: Episode info dict containing 'reference_clip' and 'start_frame'.

        Returns:
            MouseReferenceClips: Reference clip sliced to the current frame.
        """
        return self.reference_clips.at(
            clip=info["reference_clip"], frame=self._get_cur_frame(data, info)
        )

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> MouseReferenceClips:
        """Get future reference frames for observation.

        Args:
            data: MJX simulation data.
            info: Episode info dict containing 'reference_clip' and 'start_frame'.

        Returns:
            MouseReferenceClips: Slice of reference_length future frames starting
                from current_frame + 1.
        """
        return self.reference_clips.slice(
            clip=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.reference_length,
        )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Get imitation target (future reference poses relative to current state).

        Args:
            data: MJX simulation data.
            info: Episode info dict containing 'reference_clip' and 'start_frame'.

        Returns:
            Mapping[str, jp.ndarray]: OrderedDict with 'joint' (joint angle deltas)
                and 'wrist' (wrist position deltas) targets.
        """
        reference = self._get_imitation_reference(data, info)

        # Joint angle targets (difference from current)
        joint_targets = reference.joints - data.qpos

        # Wrist position targets (in world frame since base is fixed)
        wrist_pos = data.xpos[self._wrist_body_id]
        wrist_targets = jax.vmap(lambda ref_pos: ref_pos - wrist_pos)(
            reference.body_xpos("wrist_body")
        )

        return collections.OrderedDict(
            joint=joint_targets,
            wrist=wrist_targets,
        )

    # ==================== Reward Functions ====================

    def _named_reward(name: str):
        """Decorator to register a method in the reward function registry.

        Args:
            name: Key to register the reward function under.

        Returns:
            Callable: Decorator that registers and returns the reward function.
        """

        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """Reward for matching joint angles.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict.
            weight: Reward weight multiplier.
            exp_scale: Scale parameter for the Gaussian kernel.

        Returns:
            float: Weighted Gaussian reward based on joint angle L2 error.
        """
        target = self._get_current_target(data, info)
        distance = jp.linalg.norm(target.joints - data.qpos)
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_named_reward("joints_vel")
    def _joints_vel_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """Reward for matching joint velocities.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict.
            weight: Reward weight multiplier.
            exp_scale: Scale parameter for the Gaussian kernel.

        Returns:
            float: Weighted Gaussian reward based on joint velocity L2 error.
        """
        target = self._get_current_target(data, info)
        distance = jp.linalg.norm(target.joints_velocity - data.qvel)
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    @_named_reward("wrist_pos")
    def _wrist_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """Reward for matching wrist (end effector) position.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict.
            weight: Reward weight multiplier.
            exp_scale: Scale parameter for the Gaussian kernel.

        Returns:
            float: Weighted Gaussian reward based on wrist position L2 error.
        """
        target = self._get_current_target(data, info)
        wrist_pos = data.xpos[self._wrist_body_id]
        target_wrist = target.body_xpos("wrist_body")
        distance = jp.linalg.norm(wrist_pos - target_wrist)
        metrics["wrist_pos_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/wrist_pos"] = reward
        return reward

    @_named_reward("bodies_pos")
    def _bodies_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """Reward for matching all tracked body positions.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict.
            weight: Reward weight multiplier.
            exp_scale: Scale parameter for the Gaussian kernel.

        Returns:
            float: Weighted Gaussian reward based on total body position L2 error.
        """
        target = self._get_current_target(data, info)
        total_dist_sqr = 0.0
        for body_name, body_id in self._body_ids.items():
            body_pos = data.xpos[body_id]
            target_pos = target.body_xpos(body_name)
            dist_sqr = jp.sum((body_pos - target_pos) ** 2)
            metrics[f"body_errors/{body_name}"] = jp.sqrt(dist_sqr)
            total_dist_sqr += dist_sqr
        total_dist = jp.sqrt(total_dist_sqr)
        metrics["body_errors/total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_named_reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        """Penalty for control magnitude.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict.
            weight: Penalty weight multiplier.

        Returns:
            float: Negative weighted sum of squared action values.
        """
        ctrl_sqr = jp.sum(jp.square(info["action"]))
        metrics["ctrl_sqr"] = ctrl_sqr
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = -cost
        return -cost

    @_named_reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, weight) -> float:
        """Penalty for control rate of change.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            metrics: Mutable metrics dict.
            weight: Penalty weight multiplier.

        Returns:
            float: Negative weighted sum of squared action deltas.
        """
        ctrl_diff_sqr = jp.sum(jp.square(info["action"] - info["prev_action"]))
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr
        cost = weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = -cost
        return -cost

    # ==================== Termination Functions ====================

    def _named_termination_criterion(name: str):
        """Decorator to register a method in the termination function registry.

        Args:
            name: Key to register the termination function under.

        Returns:
            Callable: Decorator that registers and returns the termination function.
        """

        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        """Terminate if pose error is too large.

        Args:
            data: MJX simulation data.
            info: Episode info dict.
            max_l2_error: Maximum allowable L2 joint error before termination.

        Returns:
            bool: True if joint L2 error exceeds max_l2_error.
        """
        target = self._get_current_target(data, info)
        pose_error = jp.linalg.norm(target.joints - data.qpos)
        return pose_error > max_l2_error

    @_named_termination_criterion("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Terminate if NaN values appear in simulation.

        Args:
            data: MJX simulation data.
            info: Episode info dict.

        Returns:
            bool: True if any NaN values are found in the flattened data.
        """
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    # ==================== Rendering ====================

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 480,
        width: int = 640,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        render_ghost: bool = True,
    ) -> Sequence[np.ndarray]:
        """Render a trajectory with optional ghost showing reference motion.

        Args:
            trajectory: List of environment states to render.
            height: Rendered frame height.
            width: Rendered frame width.
            camera: Camera name or None for default.
            scene_option: MuJoCo scene options.
            render_ghost: Whether to render ghost showing reference pose.

        Returns:
            List of rendered frames as numpy arrays.
        """
        if render_ghost:
            # Create model with ghost mouse for visualization
            spec = self._spec.copy()

            # Add ghost mouse
            ghost_spec = mujoco.MjSpec.from_file(self._walker_xml_path)
            spawn_frame = spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
            ghost_body = spawn_frame.attach_body(
                ghost_spec.body("clavicle"), "", "-ghost"
            )

            # Make ghost translucent and non-colliding
            def recolor_geoms(body, rgba):
                for g in body.geoms:
                    g.rgba = rgba
                    g.contype = 0
                    g.conaffinity = 0
                for child in body.bodies:
                    recolor_geoms(child, rgba)

            recolor_geoms(ghost_body, [0.3, 0.8, 1.0, 0.4])
            mj_model = spec.compile()
        else:
            mj_model = self.mj_model

        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

        if camera is None:
            camera = (
                "my_camera-mouse"
                if "my_camera-mouse"
                in [mj_model.camera(i).name for i in range(mj_model.ncam)]
                else -1
            )

        rendered_frames = []
        for state in trajectory:
            frame_idx = self._get_cur_frame(state.data, state.info)
            clip_idx = state.info["reference_clip"]
            ref = self.reference_clips.at(clip=clip_idx, frame=frame_idx)

            if render_ghost:
                mj_data.qpos = jp.concatenate([state.data.qpos, ref.qpos])
                mj_data.qvel = jp.concatenate([state.data.qvel, ref.qvel])
            else:
                mj_data.qpos = state.data.qpos
                mj_data.qvel = state.data.qvel

            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
            rendered_frames.append(renderer.render())

        return rendered_frames

    @property
    def observation_size(self) -> int:
        """Total flattened observation size."""
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> Dict[str, Any]:
        """Observation sizes by component."""
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        return jax.tree_util.tree_map(
            lambda x: jp.prod(jp.array(x.shape)), abstract_state.obs
        )
