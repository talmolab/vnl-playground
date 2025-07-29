"""Environment wrapper for VAE training."""

import jax
import jax.numpy as jnp
from typing import Any, Dict, Tuple, Optional
import functools

from vnl_mjx.tasks.rodent.base import RodentEnv
from vnl_mjx.utils.track_mjx_loader import TrackMJXTeacher, compute_reference_observations


class VAETrainingWrapper:
    """Environment wrapper that coordinates teacher/student training."""
    
    def __init__(
        self,
        base_env: RodentEnv,
        teacher: TrackMJXTeacher,
        reference_clip: Any,
        traj_length: int = 5,
    ):
        """Initialize VAE training wrapper.
        
        Args:
            base_env: Base environment (e.g., BowlEscape, FlatArena)
            teacher: Loaded teacher network
            reference_clip: Reference trajectory data
            traj_length: Length of reference trajectory to use
        """
        self.base_env = base_env
        self.teacher = teacher
        self.reference_clip = reference_clip
        self.traj_length = traj_length
        
        # Create teacher inference functions
        from vnl_mjx.utils.track_mjx_loader import create_teacher_student_inference_fns
        self.teacher_inference_fn, self.obs_processor = create_teacher_student_inference_fns(teacher)
        
        # Get dimensions
        self.reference_obs_size = teacher.reference_obs_size
        self.proprioceptive_obs_size = teacher.proprioceptive_obs_size
        
    def reset(self, rng: jnp.ndarray) -> Dict[str, Any]:
        """Reset environment and return initial state."""
        state = self.base_env.reset(rng)
        
        # Extract proprioceptive observations (last part of obs)
        proprioceptive_obs = state.obs[..., self.reference_obs_size:]
        
        # Compute reference observations
        # Note: For simplicity, we'll use the first frame of reference
        # In practice, you might want to randomly sample start frames
        reference_obs = state.obs[..., :self.reference_obs_size]
        
        # Get teacher actions
        normalized_obs = self.obs_processor(state.obs)
        teacher_actions = self.teacher_inference_fn(normalized_obs)
        
        return {
            "state": state,
            "proprioceptive_obs": proprioceptive_obs,
            "reference_obs": reference_obs,
            "teacher_actions": teacher_actions,
            "frame_idx": 0,  # Track current frame in reference
        }
    
    def step(
        self, 
        wrapped_state: Dict[str, Any], 
        student_actions: jnp.ndarray
    ) -> Dict[str, Any]:
        """Step environment with student actions and get next teacher actions.
        
        Args:
            wrapped_state: Current wrapped state
            student_actions: Actions from VAE student
        
        Returns:
            Next wrapped state
        """
        # Step physics with student actions
        next_state = self.base_env.step(wrapped_state["state"], student_actions)
        
        # Update frame index
        next_frame_idx = wrapped_state["frame_idx"] + 1
        
        # Extract proprioceptive observations from next state
        proprioceptive_obs = next_state.obs[..., self.reference_obs_size:]
        reference_obs = next_state.obs[..., :self.reference_obs_size]
        
        # Get teacher actions for next state
        normalized_obs = self.obs_processor(next_state.obs)
        teacher_actions = self.teacher_inference_fn(normalized_obs)
        
        return {
            "state": next_state,
            "proprioceptive_obs": proprioceptive_obs,
            "reference_obs": reference_obs,
            "teacher_actions": teacher_actions,
            "frame_idx": next_frame_idx,
        }
    
    @property
    def action_size(self) -> int:
        """Get action dimension."""
        return self.base_env.action_size


def create_vae_training_environment(
    env_name: str,
    teacher_checkpoint_path: str,
    env_config: Dict[str, Any],
    reference_clip: Optional[Any] = None,
) -> VAETrainingWrapper:
    """Factory function to create VAE training environment.
    
    Args:
        env_name: Name of environment ('bowl_escape' or 'flat_arena')
        teacher_checkpoint_path: Path to teacher checkpoint
        env_config: Environment configuration
        reference_clip: Reference trajectory data
    
    Returns:
        VAE training environment wrapper
    """
    # Create base environment
    if env_name == "bowl_escape":
        from vnl_mjx.tasks.rodent import bowl_escape
        base_env = bowl_escape.BowlEscape(config=env_config)
    elif env_name == "flat_arena":
        from vnl_mjx.tasks.rodent import flat_arena
        base_env = flat_arena.FlatArena(config=env_config)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Load teacher
    teacher = TrackMJXTeacher(teacher_checkpoint_path)
    
    # Create wrapper
    return VAETrainingWrapper(
        base_env=base_env,
        teacher=teacher,
        reference_clip=reference_clip,
    )
