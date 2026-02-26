"""Online reference trajectory generator using a trained policy.

Loads a trained multi-behavior walker policy and generates reference
trajectories on-the-fly. This replaces static H5 reference clips for
the walker imitation task.

The generator produces trajectories as JAX arrays, compatible with
jax.jit and jax.lax.scan for efficient batched generation.

Behavior schedules use smooth linear blending at segment boundaries
(matching the ``transition_steps`` parameter from MultiBehaviorWalker)
so the Step 1 policy sees the same soft mode vectors it was trained on.

Usage:
    generator = OnlineReferenceGenerator(
        policy_fn=make_inference_fn(params),
        walker_env=MultiBehaviorWalker(...),
        n_frames=200,
    )
    # Generate a trajectory with a specific behavior schedule
    trajectory = generator.generate(rng, behavior_schedule)
    # trajectory.qpos.shape == (200, 9)
"""

from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.walker import consts


class WalkerTrajectory(NamedTuple):
    """A generated reference trajectory.

    All arrays have shape (n_frames, ...) where n_frames is the
    trajectory length.
    """
    qpos: jp.ndarray           # (n_frames, nq=9)
    qvel: jp.ndarray           # (n_frames, nv=9)
    xpos: jp.ndarray           # (n_frames, nbody, 3)
    xquat: jp.ndarray          # (n_frames, nbody, 4)
    behavior_labels: jp.ndarray  # (n_frames, n_modes=8) soft mode vectors


class OnlineReferenceGenerator:
    """Generates reference trajectories using a trained conditional policy.

    The generator rolls out the trained Step 1 policy in MJX to produce
    trajectories with known behavior labels. These trajectories serve as
    reference data for the imitation task.

    Args:
        policy_fn: Inference function from trained policy.
            Signature: policy_fn(obs, rng) -> action
            The obs must include the behavior mode vector (soft or one-hot).
        walker_env: A MultiBehaviorWalker environment instance (used for
            model access and observation computation).
        n_frames: Number of frames per generated trajectory.
        deterministic: If True, use policy mean (no sampling noise).
        warmup_frames: Number of standing-mode frames to prepend before the
            actual behavior schedule. Lets the walker settle from a random
            initial pose into a stable standing posture. Set to 0 to disable.
        warmup_transition_frames: Number of frames to linearly blend from
            standing to the first mode of the actual schedule.
    """

    def __init__(
        self,
        policy_fn: Callable,
        walker_env: Any,
        n_frames: int = 200,
        deterministic: bool = True,
        warmup_frames: int = 0,
        warmup_transition_frames: int = 40,
    ):
        self.policy_fn = policy_fn
        self.env = walker_env
        self.n_frames = n_frames
        self.deterministic = deterministic
        self.warmup_frames = warmup_frames
        self.warmup_transition_frames = warmup_transition_frames

        # Cache model references
        self._mjx_model = walker_env.mjx_model
        self._mj_model = walker_env.mj_model
        self._n_substeps = walker_env.n_substeps
        self._torso_id = walker_env._torso_id

    def generate(
        self,
        rng: jax.Array,
        behavior_schedule: jp.ndarray,
    ) -> WalkerTrajectory:
        """Generate a single reference trajectory.

        Args:
            rng: JAX random key.
            behavior_schedule: (n_frames, N_BEHAVIOR_MODES) soft mode vector
                array specifying the behavior blend at each frame.  Can be
                hard one-hot or smoothly blended (from ``sample_behavior_schedule``).

        Returns:
            WalkerTrajectory with all state data and behavior labels.
        """
        # Initialize walker in a standing pose
        rng, rng_init = jax.random.split(rng)
        init_state = self.env.reset(rng_init)
        init_data = init_state.data

        # Prepend standing warmup + smooth transition to schedule
        if self.warmup_frames > 0:
            stand_idx = consts.BEHAVIOR_MODES["stand"]
            stand_schedule = self.sample_fixed_schedule(
                stand_idx, self.warmup_frames
            )
            stand_vec = jax.nn.one_hot(stand_idx, consts.N_BEHAVIOR_MODES)
            first_mode = behavior_schedule[0]
            alphas = jp.linspace(
                0.0, 1.0, self.warmup_transition_frames
            ).reshape(-1, 1)
            transition = (1.0 - alphas) * stand_vec + alphas * first_mode
            full_schedule = jp.concatenate(
                [stand_schedule, transition, behavior_schedule], axis=0
            )
        else:
            full_schedule = behavior_schedule

        def step_fn(carry, behavior_vec):
            data, prev_action, rng = carry
            rng, action_rng = jax.random.split(rng)

            # Build observation (same as MultiBehaviorWalker._get_obs)
            orientations = data.xmat[1:, [0, 0], [0, 2]].ravel()
            height = data.xpos[self._torso_id, 2].reshape(1)
            velocity = data.qvel
            joint_angles = data.qpos[consts.N_ROOT_QPOS:]
            obs = jp.concatenate([
                orientations, height, velocity, joint_angles,
                prev_action, behavior_vec,
            ])

            # Get action from policy
            action = self.policy_fn(obs, action_rng)

            # Step physics
            data = mjx_env.step(
                self._mjx_model, data, action, self._n_substeps
            )

            # Record state
            frame = {
                "qpos": data.qpos,
                "qvel": data.qvel,
                "xpos": data.xpos,
                "xquat": data.xquat,
            }

            return (data, action, rng), frame

        init_carry = (init_data, jp.zeros(self.env.action_size), rng)
        _, frames = jax.lax.scan(step_fn, init_carry, full_schedule)

        # Trim warmup frames from output
        trim = (
            self.warmup_frames + self.warmup_transition_frames
            if self.warmup_frames > 0
            else 0
        )

        return WalkerTrajectory(
            qpos=frames["qpos"][trim:],
            qvel=frames["qvel"][trim:],
            xpos=frames["xpos"][trim:],
            xquat=frames["xquat"][trim:],
            behavior_labels=full_schedule[trim:],
        )

    @staticmethod
    def sample_behavior_schedule(
        rng: jax.Array,
        n_frames: int,
        n_modes: int = consts.N_BEHAVIOR_MODES,
        transition_steps: int = consts.DEFAULT_TRANSITION_STEPS,
        mode_duration_mean: int = 150,
        mode_duration_min: int = 60,
    ) -> jp.ndarray:
        """Sample a random behavior schedule matching MultiBehaviorWalker timing.

        Replicates the countdown-timer with exponential duration sampling from
        ``MultiBehaviorWalker.step()``'s mode-switching logic:

            duration ~ mode_duration_min + Exp(mode_duration_mean - mode_duration_min)

        Transitions are linearly blended over ``transition_steps`` frames,
        matching the smooth blending the Step 1 policy was trained on.

        Args:
            rng: Random key.
            n_frames: Total number of frames.
            n_modes: Number of behavior modes.
            transition_steps: Frames to linearly blend between modes.
            mode_duration_mean: Average steps per mode (shifted exponential mean).
            mode_duration_min: Minimum steps before mode switch.

        Returns:
            (n_frames, n_modes) soft mode vector array.
        """
        scale = jp.float32(mode_duration_mean - mode_duration_min)

        def step_fn(carry, _):
            (mode_idx, steps_until_switch, transition_progress,
             behavior_mode, transition_source, transition_target, rng) = carry

            rng, rng_mode, rng_dur = jax.random.split(rng, 3)

            # Decrement countdown
            steps_until_switch = steps_until_switch - 1

            # Check if countdown expired AND current transition is complete
            countdown_expired = steps_until_switch <= 0
            transition_done = transition_progress >= transition_steps
            start_new = countdown_expired & transition_done

            # Sample new target mode (ensure different from current)
            new_mode_idx = jax.random.randint(rng_mode, (), 0, n_modes)
            new_mode_idx = jp.where(
                new_mode_idx == mode_idx,
                (new_mode_idx + 1) % n_modes,
                new_mode_idx,
            )
            new_target = jax.nn.one_hot(new_mode_idx, n_modes)

            # Sample new countdown duration (shifted exponential)
            new_duration = mode_duration_min + jp.int32(
                jp.round(jax.random.exponential(rng_dur) * scale)
            )

            # Conditionally start new transition
            transition_source = jp.where(
                start_new, behavior_mode, transition_source
            )
            transition_target = jp.where(
                start_new, new_target, transition_target
            )
            transition_progress = jp.where(
                start_new, jp.int32(0), transition_progress + 1
            )
            mode_idx = jp.where(start_new, new_mode_idx, mode_idx)
            steps_until_switch = jp.where(
                start_new, new_duration, steps_until_switch
            )

            # Compute blended mode vector
            alpha = jp.clip(
                transition_progress / transition_steps, 0.0, 1.0
            )
            behavior_mode = (
                (1.0 - alpha) * transition_source + alpha * transition_target
            )

            new_carry = (
                mode_idx, steps_until_switch, transition_progress,
                behavior_mode, transition_source, transition_target, rng,
            )
            return new_carry, behavior_mode

        # Initialize: random first mode, sample first countdown
        rng, rng_init_mode, rng_init_dur = jax.random.split(rng, 3)
        init_mode_idx = jax.random.randint(rng_init_mode, (), 0, n_modes)
        init_behavior = jax.nn.one_hot(init_mode_idx, n_modes)
        init_duration = mode_duration_min + jp.int32(
            jp.round(jax.random.exponential(rng_init_dur) * scale)
        )

        init_carry = (
            init_mode_idx,
            init_duration,
            jp.int32(transition_steps),   # Start fully transitioned
            init_behavior,
            init_behavior,                # source = current
            init_behavior,                # target = current
            rng,
        )

        _, schedule = jax.lax.scan(step_fn, init_carry, jp.arange(n_frames))
        return schedule

    @staticmethod
    def sample_fixed_schedule(
        mode_idx: int,
        n_frames: int,
        n_modes: int = consts.N_BEHAVIOR_MODES,
    ) -> jp.ndarray:
        """Create a fixed single-mode schedule.

        Args:
            mode_idx: Behavior mode index (0-7).
            n_frames: Number of frames.
            n_modes: Number of modes.

        Returns:
            (n_frames, n_modes) one-hot array with the same mode throughout.
        """
        return jp.tile(
            jax.nn.one_hot(mode_idx, n_modes),
            (n_frames, 1),
        )


class PrecomputedTrajectoryDataset:
    """Pre-generated trajectory pool that replaces OnlineReferenceGenerator.

    Holds N pre-generated trajectories in memory (as a single stacked
    WalkerTrajectory with leading batch dimension). The ``generate()`` method
    randomly indexes into the pool — O(1) array lookup instead of a full
    jax.lax.scan rollout.

    Usage:
        dataset = PrecomputedTrajectoryDataset.load("trajectories.npz")
        # Drop-in replacement for OnlineReferenceGenerator:
        trajectory = dataset.generate(rng, behavior_schedule)

    Args:
        trajectories: WalkerTrajectory with shape (N, n_frames, ...) per field.
    """

    def __init__(self, trajectories: WalkerTrajectory):
        self.trajectories = trajectories
        self.n_trajectories = trajectories.qpos.shape[0]
        self.n_frames = trajectories.qpos.shape[1]

    def generate(
        self,
        rng: jax.Array,
        behavior_schedule: jp.ndarray,
    ) -> WalkerTrajectory:
        """Pick a random trajectory from the pool.

        Args:
            rng: JAX random key (used to select trajectory index).
            behavior_schedule: Ignored — the pre-generated trajectories
                already have their own behavior labels.

        Returns:
            WalkerTrajectory with shape (n_frames, ...).
        """
        idx = jax.random.randint(rng, (), 0, self.n_trajectories)
        return WalkerTrajectory(
            qpos=self.trajectories.qpos[idx],
            qvel=self.trajectories.qvel[idx],
            xpos=self.trajectories.xpos[idx],
            xquat=self.trajectories.xquat[idx],
            behavior_labels=self.trajectories.behavior_labels[idx],
        )

    @classmethod
    def load(cls, path: str) -> "PrecomputedTrajectoryDataset":
        """Load a dataset from an NPZ file.

        Expected keys: qpos, qvel, xpos, xquat, behavior_labels,
        each with shape (N, n_frames, ...).
        """
        path = Path(path)
        data = np.load(path)
        trajectories = WalkerTrajectory(
            qpos=jp.array(data["qpos"]),
            qvel=jp.array(data["qvel"]),
            xpos=jp.array(data["xpos"]),
            xquat=jp.array(data["xquat"]),
            behavior_labels=jp.array(data["behavior_labels"]),
        )
        return cls(trajectories)

    @staticmethod
    def save(
        path: str,
        trajectories: WalkerTrajectory,
    ) -> None:
        """Save stacked trajectories to an NPZ file.

        Args:
            path: Output file path (.npz).
            trajectories: WalkerTrajectory with (N, n_frames, ...) arrays.
        """
        np.savez_compressed(
            path,
            qpos=np.asarray(trajectories.qpos),
            qvel=np.asarray(trajectories.qvel),
            xpos=np.asarray(trajectories.xpos),
            xquat=np.asarray(trajectories.xquat),
            behavior_labels=np.asarray(trajectories.behavior_labels),
        )
