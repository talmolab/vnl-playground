"""
VAE distillation training script.

This script trains a VAE to imitate a frozen teacher policy. The VAE learns to map 
from proprioceptive + reference observations to actions, while regularizing with a 
prior that only uses proprioceptive observations.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

# Set up environment variables
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from vnl_mjx.models.vae import create_vae_network
from vnl_mjx.tasks.vae_distillation import create_vae_distillation_task
from vnl_mjx.training.training_loop import train_vae
from vnl_mjx.utils.vae_checkpointing import VAECheckpointer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    

def create_experiment_name(cfg: DictConfig, run_id: str) -> str:
    """Create unique experiment name."""
    env_name = cfg.env_config.env_name
    return f"Distill-{env_name}-{run_id}"


def setup_wandb(cfg: DictConfig, exp_name: str):
    """Setup Weights & Biases logging."""
    if cfg.wandb_config.project:
        wandb.init(
            project=cfg.wandb_config.project,
            config=dict(OmegaConf.to_container(cfg, resolve=True)),
            id=exp_name,
            group=cfg.wandb_config.group_name,
        )
        return wandb
    return None


@hydra.main(version_base=None, config_path="config", config_name="vae_distillation")
def main(cfg: DictConfig):
    """Main training function."""
    setup_logging()
    logging.info("Starting VAE distillation training")
    
    # Check JAX devices
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Using CPU")
    
    # Create run ID
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_path = f"{cfg.training_config.checkpoint_dir}/{run_id}"
    logging.info(f"Checkpoint directory: {checkpoint_path}")

    # Create experiment name
    exp_name = create_experiment_name(cfg, run_id)
    logging.info(f"Experiment name: {exp_name}")
    
    # Setup wandb
    wandb_logger = setup_wandb(cfg, exp_name)
    
    # Setup random number generator
    rng = jax.random.PRNGKey(cfg.training_config.seed)

    # Initialize checkpointer
    checkpointer = VAECheckpointer(
        checkpoint_dir=str(checkpoint_path),
    )
    
    # Load teacher and create distillation task
    logging.info(f"Loading teacher from: {cfg.teacher_config.checkpoint_path}")
    
    # Split the rng key
    rng, key_teacher = jax.random.split(rng)

    # Create VAE distillation task
    distillation_task = create_vae_distillation_task(
        teacher_checkpoint_path=cfg.teacher_config.checkpoint_path,
        reference_data_path=cfg.teacher_config.reference_data_path,
        episode_length=cfg.training_config.episode_length,
        random_init_range=cfg.env_config.random_init_range,
        physics_steps_per_control_step=cfg.env_config.physics_steps_per_control_step,
        rng=key_teacher,
    )
    
    logging.info(f"Distillation task created - Action size: {distillation_task.action_size}")
    logging.info(f"Reference obs size: {distillation_task.reference_obs_size}")
    logging.info(f"Proprioceptive obs size: {distillation_task.proprioceptive_obs_size}")
    
    # Create VAE network
    vae_network = create_vae_network(
        encoder_hidden_sizes=cfg.vae_config.encoder_hidden_sizes,
        decoder_hidden_sizes=cfg.vae_config.decoder_hidden_sizes,
        prior_hidden_sizes=cfg.vae_config.prior_hidden_sizes,
        latent_dim=cfg.vae_config.latent_dim,
        action_dim=distillation_task.action_size,
        activation=cfg.vae_config.activation,
    )
    
    logging.info(f"VAE network created with latent dim: {cfg.vae_config.latent_dim}")

    # Split the rng key for training
    rng, key_train = jax.random.split(rng)
    
    # Train VAE
    final_state = train_vae(
        vae_network=vae_network,
        distillation_task=distillation_task,
        num_envs=cfg.training_config.num_envs,
        num_steps=cfg.training_config.num_steps,
        batch_size=cfg.training_config.batch_size,
        learning_rate=cfg.training_config.learning_rate,
        kl_weight=cfg.training_config.kl_weight,
        eval_every=cfg.training_config.eval_every,
        checkpoint_every=cfg.training_config.checkpoint_every,
        checkpointer=checkpointer,
        wandb_logger=wandb_logger,
        rng=key_train,
    )
    
    # Save final checkpoint
    logging.info("Saving final checkpoint")
    checkpointer.save_checkpoint(
        step=cfg.training_config.num_steps,
        vae_params=final_state.params,
        optimizer_state=final_state.opt_state,
        config=dict(OmegaConf.to_container(cfg, resolve=True)),
        metrics=final_state.metrics,
    )
    
    if wandb_logger:
        wandb.finish()
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()
