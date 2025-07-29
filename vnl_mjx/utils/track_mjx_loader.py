"""Utilities for loading and interfacing with track-mjx networks."""

import logging
import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, Tuple, Optional
import functools

# Import track-mjx components
from track_mjx.agent import checkpointing
from brax.training.acme import running_statistics

from omegaconf import DictConfig



class TrackMJXTeacher:
    """Wrapper for frozen track-mjx teacher network."""
    
    def __init__(self, checkpoint_path: str):
        """Initialize teacher from checkpoint.
        
        Args:
            checkpoint_path: Path to track-mjx checkpoint
        """
        self.checkpoint_path = checkpoint_path

        # Load teacher network and normalizer from checkpoint.
        logging.info(f"Loading teacher checkpoint from: {self.checkpoint_path}")
        
        # Load checkpoint using track-mjx utilities
        checkpoint_data = checkpointing.load_checkpoint_for_eval(self.checkpoint_path)
        
        self.config = checkpoint_data["cfg"]
        self.policy_params = checkpoint_data["policy"]
        
        # Extract normalizer and network parameters
        self.normalizer_params = self.policy_params[0]  # First element is normalizer
        self.network_params = self.policy_params[1]     # Second element is network
        
        # Get observation sizes from config
        network_config = self.config["network_config"]
        self.reference_obs_size = network_config["reference_obs_size"]
        self.proprioceptive_obs_size = network_config["proprioceptive_obs_size"]
        self.total_obs_size = network_config["observation_size"]

        # Get action size from config
        self.action_size = network_config["action_size"]
        
        logging.info(f"Teacher loaded - Ref obs size: {self.reference_obs_size}, "
              f"Proprioceptive obs size: {self.proprioceptive_obs_size}")
        
    def get_config(self) -> DictConfig:
        """Get the configuration of the teacher network."""
        return self.config
    
    def normalize_observations(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Normalize observations using teacher's normalizer."""
        return running_statistics.normalize(obs, self.normalizer_params)
    
    def split_observations(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Split observations into reference and proprioceptive parts.
        
        Args:
            obs: Full observations [reference_obs, proprioceptive_obs]
        
        Returns:
            (reference_obs, proprioceptive_obs)
        """
        reference_obs = obs[..., :self.reference_obs_size]
        proprioceptive_obs = obs[..., self.reference_obs_size:]
        return reference_obs, proprioceptive_obs

    def create_inference_fn(self, rng: Optional[jax.Array] = None) -> Callable:
        """Create jitted inference function for teacher network."""
        
        # Import the network factory used by the teacher
        if self.config["train_setup"]["train_config"].get("use_lstm", False):
            from track_mjx.agent.lstm_ppo import ppo_networks
        else:
            from track_mjx.agent.mlp_ppo import ppo_networks
        
        # Create the network with same config as teacher
        network = ppo_networks.make_ppo_networks(
            **self.config["network_config"]
        )

        # Create a fixed key if none is provided
        if rng is None:
            inference_key = jax.random.PRNGKey(0)
        else:
            inference_key = rng

        @jax.jit
        def teacher_inference(obs: jnp.ndarray) -> jnp.ndarray:
            """Inference function for teacher network.
            
            Args:
                obs: Normalized observations
            
            Returns:
                Teacher actions
            """
            # Get policy logits
            logits, _ = network.policy_network.apply(self.network_params, obs, inference_key, deterministic=True)
            
            # For deterministic evaluation, use mode of distribution
            action_distribution = network.parametric_action_distribution
            actions = action_distribution.mode(logits)
            
            return actions
        
        return teacher_inference


def compute_reference_observations(
    current_state: Any,
    reference_trajectory: Any,
    walker: Any,
    current_frame: int,
    traj_length: int = 5
) -> jnp.ndarray:
    """Compute reference observations as done in track-mjx.
    
    This replicates the reference observation computation from track-mjx's
    SingleClipTracking environment.
    
    Args:
        current_state: Current MuJoCo state
        reference_trajectory: Reference clip data
        walker: Walker object with utility methods
        current_frame: Current frame index in reference
        traj_length: Length of reference trajectory to use
    
    Returns:
        Reference observations
    """
    # Get reference trajectory slice
    start_frame = current_frame
    end_frame = min(start_frame + traj_length, len(reference_trajectory.position))
    
    ref_traj = jax.tree.map(
        lambda x: x[start_frame:end_frame],
        reference_trajectory
    )
    
    # Compute local tracking positions
    track_pos_local = walker.compute_local_track_positions(
        ref_traj.position, current_state.qpos
    )
    
    # Compute quaternion distances
    quat_dist = walker.compute_quat_distances(
        ref_traj.quaternion, current_state.qpos
    )
    
    # Compute joint distances
    joint_dist = walker.compute_local_joint_distances(
        ref_traj.joints, current_state.qpos
    )
    
    # Compute body position distances
    body_pos_dist_local = walker.compute_local_body_positions(
        ref_traj.body_positions,
        current_state.xpos[1:],  # Skip floor body
        current_state.qpos,
    )
    
    # Concatenate all reference observations
    reference_obs = jnp.concatenate([
        track_pos_local,
        quat_dist,
        joint_dist,
        body_pos_dist_local,
    ])
    
    return reference_obs


def create_teacher_student_inference_fns(teacher: TrackMJXTeacher) -> Tuple[Callable, Callable]:
    """Create inference functions for teacher and utilities for student.
    
    Args:
        teacher: Loaded teacher network
    
    Returns:
        (teacher_inference_fn, observation_processing_fn)
    """
    teacher_inference_fn = teacher.create_inference_fn()
    
    @jax.jit
    def process_observations_for_teacher(obs: jnp.ndarray) -> jnp.ndarray:
        """Process observations for teacher inference."""
        normalized_obs = teacher.normalize_observations(obs)
        return normalized_obs
    
    return teacher_inference_fn, process_observations_for_teacher
