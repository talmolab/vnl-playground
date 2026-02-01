"""Sparse-reward imitation environment for rodent using elastic sequence matching.

This environment provides sparse rewards for producing the reference clip motion
sequence multiple times within each episode using online elastic DP matching.

Key behavior:
- Samples agent joint angles at a fixed rate (mocap_hz) independent of reference phase
- Uses elastic dynamic programming to allow time-warp (agent can be faster or slower)
- Emits sparse reward only when full clip matches, then resets DP for next detection
- Supports configurable speed tolerance via min_ratio/max_ratio parameters
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
from vnl_playground.tasks.task_registry import TaskRegistry

_registry = TaskRegistry()


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
        qvel_init="zeros",
        keep_clips_idx=None,
        clip_range=(
            75,
            200,
        ),  # (start, end) frame indices or None for full clip. defaults to a rear
        default_clip_idx=1,  # Fixed clip index to use (None to sample randomly). defaults idx 1 is a rear
        # Episode params
        episode_length=2500,  # 25 seconds to allow multiple sequence matches
        # Elastic matching params
        min_ratio=0.9,  # Agent can complete clip in 0.9x time (faster)
        max_ratio=1.1,  # Agent can take up to 1.1x time (slower)
        tolerance=1.5,  # Per-frame joint angle L2 threshold (radians) - relaxed for high-level matching
        use_wrapped_angles=True,  # Use atan2(sin,cos) for angle diff
        # Reward params
        # Note: sequence_match must come before dp_progress (order matters for DP state)
        reward_terms={
            "sequence_match": {"weight": 1.0},
            "dp_progress": {"weight": 0.1},
            "survival": {"weight": 0.001},  # Small reward for staying alive
        },
        # Termination conditions
        termination_criteria={
            "fallen": {"min_torso_z": 0.03, "max_torso_angle": 60},
            "nan_termination": {},
        },
    )


class SparseImitation(rodent_base.RodentEnv):
    """Sparse-reward imitation environment with elastic sequence matching.

    Uses online dynamic programming to detect when the agent's joint angles
    match the reference clip, allowing for time-warped playback (faster or slower).
    Episodes run for a fixed duration (episode_length steps) without early termination.
    """

    _registry = _registry

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

        # Compute effective clip range
        full_clip_length = self.reference_clips.qpos.shape[1]
        if self._config.clip_range is not None:
            self._clip_start = self._config.clip_range[0]
            self._clip_end = self._config.clip_range[1]
            if self._clip_start < 0 or self._clip_end > full_clip_length:
                raise ValueError(
                    f"clip_range ({self._clip_start}, {self._clip_end}) out of bounds "
                    f"for clip length {full_clip_length}"
                )
            if self._clip_start >= self._clip_end:
                raise ValueError(
                    f"clip_range start ({self._clip_start}) must be less than "
                    f"end ({self._clip_end})"
                )
        else:
            self._clip_start = 0
            self._clip_end = full_clip_length

        # Compute elastic matching bounds from config
        clip_length = self._clip_length()
        self._min_len = int(np.ceil(self._config.min_ratio * clip_length))
        self._max_len = int(np.floor(self._config.max_ratio * clip_length))

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

        # Initialize elastic DP state
        clip_length = self._clip_length()
        INF = jp.iinfo(jp.int32).max // 4  # Large value for unreachable states

        info: dict[str, Any] = {
            "start_frame": start_frame,
            "reference_clip": clip_idx,
            # Elastic DP state: min/max steps to reach each reference frame
            "min_steps": jp.full((clip_length,), INF, dtype=jp.int32),
            "max_steps": jp.full((clip_length,), -INF, dtype=jp.int32),
            "sample_phase": jp.array(0.0, dtype=jp.float32),
            "prev_final_reachable": jp.array(False),
            "match_count": jp.array(0, dtype=jp.int32),
            "sequence_matched_this_step": jp.array(False),
            "dp_progress": jp.array(0.0, dtype=jp.float32),
        }

        # Check for truncation (episode length based on control steps)
        episode_ended = data.time >= self._config.episode_length * self._config.ctrl_dt
        info["truncated"] = jp.astype(episode_ended, float)
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()

        metrics = {
            "match_count": 0.0,
            "sequence_matched": 0.0,
            "sample_phase": 0.0,
            "dp_progress": 0.0,
        }
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        terminated = self._is_done(data, info, metrics)
        done = jp.logical_or(episode_ended, terminated)

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

        # Check for truncation (episode length based on time)
        episode_ended = data.time >= self._config.episode_length * self._config.ctrl_dt
        info["truncated"] = jp.astype(episode_ended, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)

        metrics = state.metrics.copy()
        reward = self._get_reward(data, info, metrics)

        # Handle nans during sim
        reward = jp.nan_to_num(reward)

        # Check for termination (fallen, NaN, etc.)
        terminated = self._is_done(data, info, metrics)
        done = jp.logical_or(episode_ended, terminated)

        return state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=jp.astype(done, float),
            metrics=metrics,
        )

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
        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(
            state=obs,
            privileged_state=obs,
        )

    @_registry.reward("sequence_match")
    def _sequence_match_reward(self, data, info, metrics, weight) -> jax.Array:
        """Sparse reward for completing a sequence match.

        Performs elastic DP matching: advances sampling phase, runs DP updates
        for new samples, updates info fields, and returns sparse reward only
        when the full sequence is matched.

        Args:
            data: Simulation data (for joint angles).
            info: State info containing DP state (min_steps, max_steps, etc.).
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted sequence match reward (sparse).
        """
        # Advance sampling phase
        sample_phase = (
            info["sample_phase"] + self._config.ctrl_dt * self._config.mocap_hz
        )
        n_new_samples = jp.floor(sample_phase).astype(jp.int32)
        sample_phase = sample_phase - n_new_samples.astype(jp.float32)

        # Pull DP state
        INF = jp.iinfo(jp.int32).max // 4
        min_steps = info["min_steps"]
        max_steps = info["max_steps"]
        prev_final_reachable = info["prev_final_reachable"]
        match_count = info["match_count"]

        current_joints = self._get_joint_angles(data)
        # Slice reference joints to the effective clip range
        ref_joints = self.reference_clips.joints[
            info["reference_clip"], self._clip_start : self._clip_end
        ]

        # Static upper bound per control step (compile-time constant).
        # +2 is a small safety margin for floating error.
        max_updates = int(np.ceil(self._config.ctrl_dt * self._config.mocap_hz) + 2)

        def body(i, carry):
            min_s, max_s, prev_reach, m_count, matched_any = carry

            do_update = i < n_new_samples

            def do(c):
                min_s2, max_s2, prev_reach2, m_count2, matched_any2 = c
                new_min, new_max, complete = self._dp_update(
                    min_s2, max_s2, current_joints, ref_joints
                )

                matched_now = complete & (~prev_reach2)
                m_count2 = m_count2 + jp.where(matched_now, 1, 0)

                # Reset DP state on match to detect next sequence
                new_min = jp.where(matched_now, jp.full_like(new_min, INF), new_min)
                new_max = jp.where(matched_now, jp.full_like(new_max, -INF), new_max)

                return (
                    new_min,
                    new_max,
                    complete,
                    m_count2,
                    matched_any2 | matched_now,
                )

            return jax.lax.cond(
                do_update,
                do,
                lambda c: c,
                (min_s, max_s, prev_reach, m_count, matched_any),
            )

        min_steps, max_steps, prev_final_reachable, match_count, matched_any = (
            jax.lax.fori_loop(
                0,
                max_updates,
                body,
                (
                    min_steps,
                    max_steps,
                    prev_final_reachable,
                    match_count,
                    jp.array(False),
                ),
            )
        )

        # Write back into info
        info["min_steps"] = min_steps
        info["max_steps"] = max_steps
        info["sample_phase"] = sample_phase
        info["prev_final_reachable"] = prev_final_reachable
        info["match_count"] = match_count
        info["sequence_matched_this_step"] = matched_any

        # Update metrics
        metrics["match_count"] = jp.astype(match_count, float)
        metrics["sequence_matched"] = jp.astype(matched_any, float)
        metrics["sample_phase"] = jp.astype(sample_phase, float)

        # Sparse reward
        reward = jp.where(matched_any, 1.0, 0.0) * weight
        metrics["rewards/sequence_match"] = reward

        return reward

    @_registry.reward("dp_progress")
    def _dp_progress_reward(self, data, info, metrics, weight) -> jax.Array:
        """Dense reward for advancing through the reference sequence.

        Computes the furthest reachable reference frame from the DP state and
        rewards positive progress (advancement) through the sequence. Progress
        resets to 0 when a full sequence match occurs.

        Note: This reward function must run AFTER sequence_match in the config
        order, as it depends on the updated DP state from sequence_match.

        Args:
            data: Simulation data (unused, but required by interface).
            info: State info containing DP state (min_steps, sequence_matched_this_step).
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted progress reward (dense).
        """
        del data  # Unused, progress is computed from DP state

        INF = jp.iinfo(jp.int32).max // 4
        min_steps = info["min_steps"]
        matched_any = info["sequence_matched_this_step"]

        # Compute furthest reachable frame index
        L = min_steps.shape[0]
        reachable = min_steps < INF  # already pruned by max_len inside _dp_update
        # Find furthest reachable frame index (-1 if none reachable)
        best_idx = jp.max(jp.where(reachable, jp.arange(L, dtype=jp.int32), -1))
        progress = (best_idx + 1).astype(jp.float32) / jp.array(
            L, jp.float32
        )  # in [0,1]

        # Compute progress delta from previous step
        prev_progress = info["dp_progress"]
        progress_delta = jp.maximum(progress - prev_progress, 0.0)

        # Reset progress on sequence match (so next sequence starts from 0)
        progress = jp.where(matched_any, jp.array(0.0, dtype=jp.float32), progress)

        # Write back into info
        info["dp_progress"] = progress

        # Update metrics
        metrics["dp_progress"] = progress

        # Dense progress reward
        reward = weight * progress_delta
        metrics["rewards/dp_progress"] = reward

        return reward

    @_registry.reward("survival")
    def _survival_reward(self, data, info, metrics, weight) -> jax.Array:
        """Small constant reward for staying alive (not given on termination step).

        Encourages the agent to avoid termination conditions and keep
        attempting to match the sequence.

        Args:
            data: Simulation data.
            info: State info.
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted survival reward (0 if terminated, weight otherwise).
        """
        # Check if this step results in termination
        terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = self._registry.terminations[name]
            terminated = jp.logical_or(
                terminated, termination_fcn(self, data, info, **kwargs)
            )

        reward = jp.where(terminated, 0.0, weight)
        metrics["rewards/survival"] = reward
        return reward

    @_registry.termination("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = 0.03,
        max_torso_angle: float = 60,
    ) -> jax.Array:
        """Check if rodent has fallen.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum z height threshold.
            max_torso_angle: Maximum angle from vertical in degrees.

        Returns:
            Boolean indicating if fallen.
        """
        del info

        torso_body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        torso_z = torso_body.xpos[2]

        below_ground = torso_z < min_torso_z

        # xmat is 3x3 rotation matrix, [-1, -1] is element (2,2) = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_ground, too_tilted)

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> jax.Array:
        """Check for NaN values in simulation data.

        Args:
            data: Simulation data.
            info: State info (unused).

        Returns:
            Boolean indicating if NaN detected.
        """
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    def _wrapped_angle_distance(
        self, angles1: jax.Array, angles2: jax.Array
    ) -> jax.Array:
        """Compute L2 norm of wrapped angle differences.

        Uses atan2(sin(a-b), cos(a-b)) to handle angle wrapping at ±π.
        """
        diff = angles1 - angles2
        wrapped_diff = jp.arctan2(jp.sin(diff), jp.cos(diff))
        return jp.linalg.norm(wrapped_diff)

    def _dp_update(
        self,
        prev_min: jax.Array,
        prev_max: jax.Array,
        current_joints: jax.Array,
        ref_joints: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Single step of elastic sequence matching DP.

        For each reference frame i, we track the min/max number of agent steps
        that could reach that frame. Transitions allowed:
        - stay (i → i): agent is slower, repeats frames
        - advance +1 (i-1 → i): normal speed
        - advance +2 (i-2 → i): agent is faster, skips reference frames

        Args:
            prev_min: Previous min_steps array, shape (L,)
            prev_max: Previous max_steps array, shape (L,)
            current_joints: Current agent joint angles, shape (n_joints,)
            ref_joints: Reference joint angles for all frames, shape (L, n_joints)

        Returns:
            new_min: Updated min_steps array
            new_max: Updated max_steps array
            complete: Boolean, True if full sequence matched
        """
        L = ref_joints.shape[0]
        INF = jp.iinfo(jp.int32).max // 4
        NINF = -INF
        tolerance = self._config.tolerance
        min_len = self._min_len
        max_len = self._max_len

        # Compute per-frame emission match (does current_joints match each ref frame?)
        if self._config.use_wrapped_angles:
            diff = current_joints[None, :] - ref_joints  # (L, n_joints)
            wrapped_diff = jp.arctan2(jp.sin(diff), jp.cos(diff))
            frame_distances = jp.linalg.norm(wrapped_diff, axis=1)  # (L,)
        else:
            frame_distances = jp.linalg.norm(
                current_joints[None, :] - ref_joints, axis=1
            )  # (L,)
        ok = frame_distances < tolerance  # (L,) boolean mask

        # Shift helpers for transitions from i-1 and i-2
        prev_min_m1 = jp.concatenate([jp.array([INF], dtype=jp.int32), prev_min[:-1]])
        prev_min_m2 = jp.concatenate(
            [jp.array([INF, INF], dtype=jp.int32), prev_min[:-2]]
        )
        prev_max_m1 = jp.concatenate([jp.array([NINF], dtype=jp.int32), prev_max[:-1]])
        prev_max_m2 = jp.concatenate(
            [jp.array([NINF, NINF], dtype=jp.int32), prev_max[:-2]]
        )

        # Candidate min/max from each transition (+1 agent step)
        cand_min_stay = jp.where(prev_min < INF, prev_min + 1, INF)
        cand_min_1 = jp.where(prev_min_m1 < INF, prev_min_m1 + 1, INF)
        cand_min_2 = jp.where(prev_min_m2 < INF, prev_min_m2 + 1, INF)

        cand_max_stay = jp.where(prev_max > NINF, prev_max + 1, NINF)
        cand_max_1 = jp.where(prev_max_m1 > NINF, prev_max_m1 + 1, NINF)
        cand_max_2 = jp.where(prev_max_m2 > NINF, prev_max_m2 + 1, NINF)

        # Take min/max across all transitions
        cand_min = jp.minimum(jp.minimum(cand_min_stay, cand_min_1), cand_min_2)
        cand_max = jp.maximum(jp.maximum(cand_max_stay, cand_max_1), cand_max_2)

        # Allow starting a new match at reference frame 0
        cand_min = cand_min.at[0].set(jp.minimum(cand_min[0], 1))
        cand_max = cand_max.at[0].set(jp.maximum(cand_max[0], 1))

        # Gate by emission match (only update if frame matches)
        new_min = jp.where(ok, cand_min, INF)
        new_max = jp.where(ok, cand_max, NINF)

        # Enforce max_len bound (prune paths that are too long)
        new_min = jp.where(new_min <= max_len, new_min, INF)
        new_max = jp.where(new_min <= max_len, jp.minimum(new_max, max_len), NINF)

        # Check completion: final frame reachable within [min_len, max_len]
        complete = (new_min[-1] <= max_len) & (new_max[-1] >= min_len)

        return new_min, new_max, complete

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
        """Return the number of frames in the effective clip range."""
        return self._clip_end - self._clip_start

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        """Compute current frame from simulation time (like dense imitation)."""
        time_in_frames = data.time * self._config.mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the reference data at the current frame (offset by clip_start)."""
        frame = self._get_cur_frame(data, info) + self._clip_start
        return self.reference_clips.at(clip=info["reference_clip"], frame=frame)

    def _compute_joint_error(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> jax.Array:
        """Compute L2 norm of joint angle error vs reference."""
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        return jp.linalg.norm(target.joints - joints)

    def _get_cyclic_ref_frame(self, data: mjx.Data) -> jax.Array:
        """Compute current reference frame with cyclic wrap (for rendering).

        Returns the frame index into the full clip (offset by clip_start).
        """
        time_in_frames = data.time * self._config.mocap_hz
        return (
            jp.floor(time_in_frames).astype(int) % self._clip_length()
            + self._clip_start
        )

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
        clip_length = self._clip_length()
        for i, state in enumerate(trajectory):
            # Use cyclic time-based frame indexing (offset by clip_start)
            time_in_frames = state.data.time * self._config.mocap_hz
            frame = (
                jp.floor(time_in_frames).astype(int) % clip_length + self._clip_start
            )
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
        return jp.sum(flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0])

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
