"""Online reference trajectory generator using a trained policy.

Loads a trained multi-behavior walker policy and generates reference
trajectories on-the-fly. This replaces static H5 reference clips for
the walker imitation task.

The generator produces trajectories as JAX arrays, compatible with
jax.jit and jax.lax.scan for efficient batched generation.

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

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jp
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
    behavior_labels: jp.ndarray  # (n_frames, n_modes=4) one-hot


class OnlineReferenceGenerator:
    """Generates reference trajectories using a trained conditional policy.

    The generator rolls out the trained Step 1 policy in MJX to produce
    trajectories with known behavior labels. These trajectories serve as
    reference data for the imitation task.

    Args:
        policy_fn: Inference function from trained policy.
            Signature: policy_fn(obs, rng) -> action
            The obs must include the behavior mode one-hot.
        walker_env: A MultiBehaviorWalker environment instance (used for
            model access and observation computation).
        n_frames: Number of frames per generated trajectory.
        deterministic: If True, use policy mean (no sampling noise).
    """

    def __init__(
        self,
        policy_fn: Callable,
        walker_env: Any,
        n_frames: int = 200,
        deterministic: bool = True,
    ):
        self.policy_fn = policy_fn
        self.env = walker_env
        self.n_frames = n_frames
        self.deterministic = deterministic

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
            behavior_schedule: (n_frames, N_BEHAVIOR_MODES) one-hot array
                specifying which behavior mode to use at each frame.

        Returns:
            WalkerTrajectory with all state data and behavior labels.
        """
        # Initialize walker in a standing pose
        rng, rng_init = jax.random.split(rng)
        init_state = self.env.reset(rng_init)
        init_data = init_state.data

        def step_fn(carry, behavior_onehot):
            data, prev_action, rng = carry
            rng, action_rng = jax.random.split(rng)

            # Build observation (same as MultiBehaviorWalker._get_obs)
            orientations = data.xmat[1:, [0, 0], [0, 2]].ravel()
            height = data.xpos[self._torso_id, 2].reshape(1)
            velocity = data.qvel
            joint_angles = data.qpos[consts.N_ROOT_QPOS:]
            obs = jp.concatenate([
                orientations, height, velocity, joint_angles,
                prev_action, behavior_onehot,
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
        _, frames = jax.lax.scan(step_fn, init_carry, behavior_schedule)

        return WalkerTrajectory(
            qpos=frames["qpos"],
            qvel=frames["qvel"],
            xpos=frames["xpos"],
            xquat=frames["xquat"],
            behavior_labels=behavior_schedule,
        )

    @staticmethod
    def sample_behavior_schedule(
        rng: jax.Array,
        n_frames: int,
        n_transitions: int = 1,
        n_modes: int = consts.N_BEHAVIOR_MODES,
    ) -> jp.ndarray:
        """Sample a random behavior schedule with specified transitions.

        Args:
            rng: Random key.
            n_frames: Total number of frames.
            n_transitions: Number of behavior transitions (0 = single mode,
                1 = one transition, etc.).
            n_modes: Number of behavior modes.

        Returns:
            (n_frames, n_modes) one-hot array.
        """
        rng_modes, rng_splits = jax.random.split(rng)
        n_segments = n_transitions + 1

        # Sample mode for each segment
        modes = jax.random.randint(rng_modes, (n_segments,), 0, n_modes)

        # Ensure consecutive segments have different modes
        def fix_consecutive(prev_mode, mode):
            fixed = jp.where(mode == prev_mode, (mode + 1) % n_modes, mode)
            return fixed, fixed

        _, fixed_modes = jax.lax.scan(fix_consecutive, modes[0], modes[1:])
        modes = jp.concatenate([modes[:1], fixed_modes])

        # Build per-frame mode indices using equal-width segments
        frames_per_segment = n_frames // n_segments
        frame_indices = jp.arange(n_frames)
        # Each frame belongs to segment (frame_idx // frames_per_segment),
        # clamped to the last segment
        segment_ids = jp.minimum(
            frame_indices // frames_per_segment, n_segments - 1
        )
        frame_modes = modes[segment_ids]

        return jax.nn.one_hot(frame_modes, n_modes)

    @staticmethod
    def sample_fixed_schedule(
        mode_idx: int,
        n_frames: int,
        n_modes: int = consts.N_BEHAVIOR_MODES,
    ) -> jp.ndarray:
        """Create a fixed single-mode schedule.

        Args:
            mode_idx: Behavior mode index (0-3).
            n_frames: Number of frames.
            n_modes: Number of modes.

        Returns:
            (n_frames, n_modes) one-hot array with the same mode throughout.
        """
        return jp.tile(
            jax.nn.one_hot(mode_idx, n_modes),
            (n_frames, 1),
        )
