"""VAE checkpointing utilities."""

import orbax.checkpoint as ocp
from flax.training import orbax_utils
from typing import Dict, Any
import jax
import logging

from vnl_mjx.utils.track_mjx_loader import TrackMJXTeacher


class VAECheckpointer:
    """Checkpointing utilities for VAE training."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.mgr_options = ocp.CheckpointManagerOptions(
            step_prefix="VAE"
        )
        self.ckpt_mgr = ocp.CheckpointManager(checkpoint_dir, options=self.mgr_options)
    
    def save_checkpoint(
        self, 
        step: int, 
        vae_params: Any, 
        optimizer_state: Any, 
        config: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """Save VAE checkpoint.
        
        Args:
            step: Training step
            vae_params: VAE network parameters
            optimizer_state: Optimizer state
            config: Training configuration
            metrics: Current metrics
        """
        logging.info(f"Saving checkpoint at step {step}")
        
        self.ckpt_mgr.save(
            step=step,
            args=ocp.args.Composite(
                vae_params=ocp.args.StandardSave(vae_params),
                optimizer_state=ocp.args.StandardSave(optimizer_state),
                config=ocp.args.JsonSave(config),
                metrics=ocp.args.JsonSave(metrics),
            ),
        )
    
    def restore_checkpoint(self, step: int = None) -> Dict[str, Any]:
        """Restore VAE checkpoint.
        
        Args:
            step: Specific step to restore (None for latest)
        
        Returns:
            Restored checkpoint data
        """
        if step is None:
            step = self.ckpt_mgr.latest_step()
        
        if step is None:
            logging.info("No checkpoint found, starting from scratch")
            return None
        
        logging.info(f"Restoring checkpoint from step {step}")
        
        # Create abstract pytrees for restoration
        abstract_vae_params = None  # Will be filled by training script
        abstract_optimizer_state = None  # Will be filled by training script
        
        restored = self.ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(
                vae_params=ocp.args.StandardRestore(abstract_vae_params),
                optimizer_state=ocp.args.StandardRestore(abstract_optimizer_state),
                config=ocp.args.JsonRestore(),
                metrics=ocp.args.JsonRestore(),
            ),
        )
        
        return restored
    
    def load_teacher_checkpoint(self, teacher_ckpt_path: str) -> TrackMJXTeacher:
        """Load track-mjx teacher checkpoint.
        
        Args:
            teacher_ckpt_path: Path to teacher checkpoint
        
        Returns:
            Loaded teacher network
        """
        return TrackMJXTeacher(teacher_ckpt_path)
