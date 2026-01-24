"""Sparse-reward imitation environment for rodent.

This environment provides sparse rewards based on whether joint angles match
the reference clip within a tolerance threshold. Unlike the dense imitation
environment, rewards are binary (1.0 if matched, 0.0 otherwise) and episodes
run for a fixed duration without early termination.
"""

import collections
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from jax import flatten_util

from .. import utils
from . import base as rodent_base
from . import consts
from vnl_playground.tasks.reference_clips import ReferenceClips


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # Model paths
        walker_xml_path=consts.RODENT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        # Simulation params
        mujoco_impl="jax",
        sim_dt=0.002,
        ctrl_dt=0.01,  # 50 Hz control
        solver="cg",
        iterations=5,
        ls_iterations=5,
        naconmax=256,
        njmax=128,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        # Reference data params
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=50,
        clip_length=250,
        clip_set="all",
        reference_length=5,
        qvel_init="zeros",
        keep_clips_idx=None,
        # Episode params
        episode_length=1000,  # 10 seconds at ctrl_dt=0.01
        default_clip_idx=0,  # Fixed clip index to use (None to sample randomly)
        # Reward params
        reward_terms={
            # Small per-frame reward for matching
            "frame_match": {
                "tolerance": 1.0,  # L2 norm threshold for match
                "weight": 0.01,  # Small reward per matched frame
            },
            # Large terminal bonus for matching entire trajectory
            "trajectory_match": {
                "tolerance": 1.0,  # L2 norm threshold (max error must be below)
                "weight": 1.0,  # Large bonus at episode end
            },
        },
        # No termination conditions (empty dict)
        termination_criteria={},
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}


class SparseImitation(rodent_base.RodentEnv):
    """Sparse-reward imitation environment.

    Rewards are binary: 1.0 if joint angle error is below tolerance, 0.0 otherwise.
    Episodes run for a fixed duration (episode_length steps) without early termination.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
        clips: Optional[ReferenceClips] = None,
    ) -> None:
        """Initialize the sparse imitation environment.

        Args:
            config: Configuration dictionary for the environment.
            config_overrides: Dictionary of configuration overrides.
            clips: Pre-loaded ReferenceClips object. If provided, it overrides
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
    ) -> mjx_env.State:
        """Reset the environment state.

        Initializes the rodent to default pose with small joint noise and
        random yaw rotation. Samples a reference clip to track for rewards.

        Args:
            rng: JAX random number generator state.
            clip_idx: If provided, uses this clip index instead of sampling randomly.

        Returns:
            The initial state of the environment after reset.
        """
        rng, clip_rng, reset_rng = jax.random.split(rng, 3)

        # Use fixed clip from config, argument override, or sample randomly
        if clip_idx is None:
            if self._config.default_clip_idx is not None:
                clip_idx = self._config.default_clip_idx
            else:
                clip_idx = jax.random.choice(clip_rng, self._clip_set)

        # Always start from frame 0
        start_frame = 0

        data = self._reset_data(reset_rng)

        info: dict[str, Any] = {
            "start_frame": start_frame,
            "reference_clip": clip_idx,
            "max_error": 0.0,  # Track max joint error across episode
        }

        # Check for truncation (episode length based on frames or clip end)
        last_valid_frame = self._clip_length() - 1
        current_frame = self._get_cur_frame(data, info)
        episode_ended = jp.logical_or(
            current_frame >= start_frame + self._config.episode_length,
            current_frame > last_valid_frame,
        )
        info["truncated"] = jp.astype(episode_ended, float)
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()

        metrics = {
            "current_frame": jp.astype(current_frame, float),
            "max_error": 0.0,
        }
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = episode_ended  # Only truncation, no early termination

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

        info = state.info.copy()

        # Compute current joint error and update max_error
        current_error = self._compute_joint_error(data, info)
        info["max_error"] = jp.maximum(state.info["max_error"], current_error)

        # Check for truncation (episode length based on frames or clip end)
        last_valid_frame = self._clip_length() - 1
        current_frame = self._get_cur_frame(data, info)
        episode_ended = jp.logical_or(
            current_frame >= info["start_frame"] + self._config.episode_length,
            current_frame > last_valid_frame,
        )
        info["truncated"] = jp.astype(episode_ended, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        done = episode_ended  # Only truncation, no early termination

        metrics = state.metrics.copy()
        metrics["current_error"] = current_error
        metrics["max_error"] = info["max_error"]
        reward = self._get_reward(data, info, metrics)

        # Handle nans during sim
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        state.metrics["current_frame"] = jp.astype(current_frame, float)
        return state

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        """Get observations."""
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
            ]
        )
        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        """Compute total reward from configured reward terms."""
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def _reset_data(self, rng: jax.Array) -> mjx.Data:
        """Initialize MuJoCo data with default pose, joint noise, and random yaw.

        Args:
            rng: JAX random number generator state.

        Returns:
            Initialized MuJoCo data.
        """
        rng, yaw_rng, joint_rng = jax.random.split(rng, 3)

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            njmax=self._config.njmax,
            naconmax=self._config.naconmax,
        )

        # Get default qpos from model
        default_qpos = self.mjx_model.qpos0.copy()

        # Random yaw rotation about z-axis (quaternion: [w, x, y, z])
        # Rotation about z: [cos(θ/2), 0, 0, sin(θ/2)]
        yaw_angle = jax.random.uniform(yaw_rng, (), minval=0, maxval=2 * jp.pi)
        yaw_quat = jp.array(
            [
                jp.cos(yaw_angle / 2),
                0.0,
                0.0,
                jp.sin(yaw_angle / 2),
            ]
        )

        # Small noise perturbations to joint angles
        n_joints = self.mjx_model.nq - 7  # Exclude root pos (3) and quat (4)
        joint_noise_scale = 0.05  # Small perturbations
        joint_noise = jax.random.normal(joint_rng, (n_joints,)) * joint_noise_scale

        # Construct new qpos: [root_pos(3), root_quat(4), joints(n_joints)]
        new_qpos = jp.concatenate(
            [
                default_qpos[:3],  # Root position (unchanged)
                yaw_quat,  # Random yaw rotation
                default_qpos[7:] + joint_noise,  # Joints with noise
            ]
        )

        data = data.replace(qpos=new_qpos)
        data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
        data = mjx.forward(self.mjx_model, data)
        return data

    def null_action(self) -> jp.ndarray:
        """Return a zero action."""
        return jp.zeros(self.action_size)

    def _clip_length(self):
        """Return the number of frames per clip."""
        return self.reference_clips.qpos.shape[1]

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        """Compute current frame from simulation time (like dense imitation)."""
        time_in_frames = data.time * self._config.mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the reference data at the current frame."""
        return self.reference_clips.at(
            clip=info["reference_clip"], frame=self._get_cur_frame(data, info)
        )

    def _compute_joint_error(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> jax.Array:
        """Compute L2 norm of joint angle error vs reference."""
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        return jp.linalg.norm(target.joints - joints)

    # Reward function decorator and registry
    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("frame_match")
    def _frame_match_reward(self, data, info, metrics, weight, tolerance) -> float:
        """Small per-frame reward: weight if current frame matches, else 0."""
        current_error = metrics.get(
            "current_error", self._compute_joint_error(data, info)
        )
        matched = current_error < tolerance
        reward = weight * jp.astype(matched, float)
        metrics["rewards/frame_match"] = reward
        metrics["frame_matched"] = jp.astype(matched, float)
        return reward

    @_named_reward("trajectory_match")
    def _trajectory_match_reward(self, data, info, metrics, weight, tolerance) -> float:
        """Terminal bonus: weight if max_error < tolerance at episode end, else 0."""
        del data  # Unused, reward depends only on tracked max_error
        is_terminal = info["truncated"] > 0.5
        max_error = info["max_error"]
        trajectory_matched = max_error < tolerance

        # Only give reward at terminal state
        reward = jp.where(
            is_terminal,
            weight * jp.astype(trajectory_matched, float),
            0.0,
        )
        metrics["rewards/trajectory_match"] = reward
        metrics["trajectory_matched"] = jp.astype(trajectory_matched, float)
        return reward

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        render_ghost: bool = True,
    ) -> Sequence[np.ndarray]:
        """Render a sequence of states (trajectory).

        Args:
            trajectory: Sequence of environment states to render.
            height: Height of the rendered frames in pixels.
            width: Width of the rendered frames in pixels.
            camera: Camera name or index to use for rendering.
            scene_option: Additional scene rendering options.
            modify_scene_fns: Functions to modify the scene before rendering.
            render_ghost: Whether to render the ghost model showing the imitation target.

        Returns:
            List of rendered frames as numpy arrays.
        """
        if render_ghost:
            spec = self._spec.copy()
            ghost_rodent = mujoco.MjSpec.from_file(self._walker_xml_path)
            ghost_rescale = self.reference_clips._config["model"]["SCALE_FACTOR"]
            if ghost_rescale != 1.0:
                ghost_rodent = utils.dm_scale_spec(ghost_rodent, ghost_rescale)
            for body in ghost_rodent.worldbody.bodies:
                utils._recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])
            spawn_site = spec.worldbody.add_frame(pos=(0, 0, 0.05), quat=(1, 0, 0, 0))
            spawn_body = spawn_site.attach_body(
                ghost_rodent.worldbody, "", suffix="-ghost"
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
            # Use time-based frame indexing (like dense imitation)
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

        return rendered_frames

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
