"""
VAE Distillation Task Environment.

This environment is specifically designed for teacher-student distillation training.
It loads a track-mjx environment and extracts observations/actions for VAE training,
without being tied to specific task rewards or environments.
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict, Tuple, Optional, Callable
import functools
from ml_collections import config_dict

from vnl_mjx.utils.track_mjx_loader import TrackMJXTeacher
from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.io.load import make_multiclip_data, load_reference_clip_data

from brax import envs
from omegaconf import DictConfig


class VAEDistillationTask:
    """
    Pure distillation task environment for VAE training.
    
    This environment:
    1. Loads the same track-mjx environment as the teacher
    2. Provides observations in the same format as teacher training
    3. Focuses purely on action prediction accuracy
    4. Is agnostic to specific task rewards (bowl_escape, flat_arena, etc.)
    """
    
    def __init__(
        self,
        teacher: TrackMJXTeacher,
        reference_data_path: str,
        episode_length: int = 1000,
        rng: Optional[jax.Array] = None,
    ):
        """Initialize VAE distillation task.
        
        Args:
            teacher: Loaded teacher network
            reference_data_path: Path to reference trajectory data
            episode_length: Length of training episodes
            rng: Optional random number generator for initialization
        """
        self.teacher = teacher
        self.teacher_config = teacher.config
        self.episode_length = episode_length
        self.teacher_cfg = teacher.get_config()
        
        # Load reference data
        self.reference_clip = load_reference_clip_data(reference_data_path)
        
        # Create the underlying track-mjx environment
        self._create_tracking_environment()
        
        # Get observation dimensions
        self.reference_obs_size = teacher.reference_obs_size
        self.proprioceptive_obs_size = teacher.proprioceptive_obs_size
        self.total_obs_size = self.reference_obs_size + self.proprioceptive_obs_size

        # Get action size from teacher
        self.action_size = teacher.action_size
        
        # Create teacher inference function
        self.teacher_inference_fn = teacher.create_inference_fn(rng=rng)
        
        # Create observation processing functions
        self._create_observation_functions()
    
    def _create_tracking_environment(self):
        """Create the underlying track-mjx environment from teacher config."""
        
        # Create walker
        walker = Rodent(**self.teacher_cfg["walker_config"])

        # Create task environment
        self.env = MultiClipTracking(
            reference_clip=self.reference_clip,
            walker=walker,
            reward_config=None,  # We don't need rewards for distillation
            **self.teacher_cfg["env_config"]["env_args"],
            **self.teacher_cfg["reference_config"],
        )
    
    def _create_observation_functions(self):
        """Create observation processing functions."""
        
        @jax.jit
        def normalize_observations(obs: jnp.ndarray) -> jnp.ndarray:
            """Normalize observations using teacher's normalizer."""
            return self.teacher.normalize_observations(obs)
        
        @jax.jit
        def split_observations(obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Split observations into reference and proprioceptive parts."""
            return self.teacher.split_observations(obs)
        
        self.normalize_observations = normalize_observations
        self.split_observations = split_observations
    
    def reset(self, rng: jnp.ndarray, clip_idx: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment and return initial state for VAE training.
        
        Args:
            rng: Random number generator
            clip_idx: Optional clip index for multi-clip data
        
        Returns:
            Dictionary containing state information for VAE training
        """
        # Reset the underlying environment
        if hasattr(self.env, 'reset') and clip_idx is not None:
            # Multi-clip environment
            state = self.env.reset(rng, clip_idx=clip_idx)
        else:
            # Single-clip environment
            state = self.env.reset(rng)
        
        # Process observations
        normalized_obs = self.normalize_observations(state.obs)
        reference_obs, proprioceptive_obs = self.split_observations(state.obs)
        
        # Get teacher action for this state
        teacher_action = self.teacher_inference_fn(normalized_obs)
        
        return {
            "state": state,
            "obs": state.obs,
            "normalized_obs": normalized_obs,
            "reference_obs": reference_obs,
            "proprioceptive_obs": proprioceptive_obs,
            "teacher_action": teacher_action,
            "step_count": 0,
        }
    
    def step(
        self, 
        distill_state: Dict[str, Any], 
        student_action: jnp.ndarray
    ) -> Dict[str, Any]:
        """Step environment with student action and get next teacher action.
        
        Args:
            distill_state: Current distillation state
            student_action: Action from VAE student
        
        Returns:
            Next distillation state
        """
        # Step the underlying environment with student action
        next_state = self.env.step(distill_state["state"], student_action)
        
        # Process next observations
        normalized_obs = self.normalize_observations(next_state.obs)
        reference_obs, proprioceptive_obs = self.split_observations(next_state.obs)
        
        # Get teacher action for next state
        teacher_action = self.teacher_inference_fn(normalized_obs)
        
        # Update step count
        next_step_count = distill_state["step_count"] + 1
        
        return {
            "state": next_state,
            "obs": next_state.obs,
            "normalized_obs": normalized_obs,
            "reference_obs": reference_obs,
            "proprioceptive_obs": proprioceptive_obs,
            "teacher_action": teacher_action,
            "step_count": next_step_count,
        }
    
    def is_done(self, distill_state: Dict[str, Any]) -> bool:
        """Check if episode is done."""
        return (
            distill_state["state"].done or 
            distill_state["step_count"] >= self.episode_length
        )


def create_vae_distillation_task(
    teacher_checkpoint_path: str,
    reference_data_path: str,
    episode_length: int = 1000,
) -> VAEDistillationTask:
    """Factory function to create VAE distillation task.
    
    Args:
        teacher_checkpoint_path: Path to track-mjx teacher checkpoint
        reference_data_path: Path to reference trajectory data
        episode_length: Length of training episodes
    
    Returns:
        VAE distillation task environment
    """
    # Load teacher
    teacher = TrackMJXTeacher(teacher_checkpoint_path)
    
    # Create distillation task
    return VAEDistillationTask(
        teacher=teacher,
        reference_data_path=reference_data_path,
        episode_length=episode_length,
    )
