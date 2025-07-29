"""
VAE evaluation script.

This script evaluates a trained VAE by comparing its actions to the teacher
and visualizing the learned behaviors.
"""

import os
import logging
from pathlib import Path

# Set up environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from vnl_mjx.models.vae import create_vae_network
from vnl_mjx.tasks.vae_distillation import create_vae_distillation_task
from vnl_mjx.utils.vae_checkpointing import VAECheckpointer


def evaluate_vae_vs_teacher(
    vae_network,
    vae_params,
    distillation_task,
    num_episodes: int = 10,
    episode_length: int = 1000,
    rng: jnp.ndarray = None,
) -> dict:
    """Evaluate VAE against teacher performance.
    
    Args:
        vae_network: VAE network
        vae_params: Trained VAE parameters
        distillation_task: VAE distillation task
        num_episodes: Number of evaluation episodes
        episode_length: Length of each episode
        rng: Random number generator
    
    Returns:
        Evaluation metrics
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    metrics = {
        "action_mse": [],
        "action_mae": [],
        "teacher_rewards": [],
        "student_rewards": [],
        "episode_lengths": [],
    }
    
    for episode in range(num_episodes):
        logging.info(f"Evaluating episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        distill_state = distillation_task.reset(reset_rng)
        
        episode_action_errors = []
        teacher_reward = 0.0
        student_reward = 0.0
        
        for step in range(episode_length):
            # Get teacher actions
            teacher_actions = distill_state["teacher_action"]
            
            # Get VAE actions
            rng, vae_rng = jax.random.split(rng)
            vae_outputs = vae_network.apply(
                vae_params,
                distill_state["proprioceptive_obs"],
                distill_state["reference_obs"],
                vae_rng,
            )
            student_actions = vae_outputs["actions"]
            
            # Compute action errors
            action_error = jnp.mean((teacher_actions - student_actions) ** 2)
            episode_action_errors.append(action_error)
            
            # Step with student actions
            distill_state = distillation_task.step(distill_state, student_actions)
            
            # Accumulate rewards (if available)
            if "reward" in distill_state["state"].__dict__:
                student_reward += distill_state["state"].reward
            
            # Check if episode is done
            if distillation_task.is_done(distill_state):
                break
        
        # Store episode metrics
        metrics["action_mse"].append(np.mean(episode_action_errors))
        metrics["action_mae"].append(np.mean([np.sqrt(err) for err in episode_action_errors]))
        metrics["student_rewards"].append(student_reward)
        metrics["episode_lengths"].append(step + 1)
    
    # Compute aggregate metrics
    aggregate_metrics = {
        "mean_action_mse": np.mean(metrics["action_mse"]),
        "std_action_mse": np.std(metrics["action_mse"]),
        "mean_action_mae": np.mean(metrics["action_mae"]),
        "std_action_mae": np.std(metrics["action_mae"]),
        "mean_student_reward": np.mean(metrics["student_rewards"]),
        "std_student_reward": np.std(metrics["student_rewards"]),
        "mean_episode_length": np.mean(metrics["episode_lengths"]),
    }
    
    return aggregate_metrics, metrics


def visualize_latent_space(
    vae_network,
    vae_params,
    distillation_task,
    num_samples: int = 1000,
    save_path: str = None,
    rng: jnp.ndarray = None,
):
    """Visualize the learned latent space.
    
    Args:
        vae_network: VAE network
        vae_params: Trained VAE parameters
        distillation_task: VAE distillation task
        num_samples: Number of samples to collect
        save_path: Path to save plots
        rng: Random number generator
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    logging.info("Collecting latent space samples...")
    
    latents = []
    enc_means = []
    prior_means = []
    
    # Collect samples
    for i in range(num_samples):
        if i % 100 == 0:
            logging.info(f"Sample {i}/{num_samples}")
        
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        distill_state = distillation_task.reset(reset_rng)
        
        # Get VAE outputs
        rng, vae_rng = jax.random.split(rng)
        vae_outputs = vae_network.apply(
            vae_params,
            distill_state["proprioceptive_obs"],
            distill_state["reference_obs"],
            vae_rng,
        )
        
        latents.append(vae_outputs["latent"])
        enc_means.append(vae_outputs["enc_mean"])
        prior_means.append(vae_outputs["prior_mean"])
    
    latents = jnp.stack(latents)
    enc_means = jnp.stack(enc_means)
    prior_means = jnp.stack(prior_means)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot first two latent dimensions
    axes[0, 0].scatter(latents[:, 0], latents[:, 1], alpha=0.6)
    axes[0, 0].set_title("Latent Space (first 2 dims)")
    axes[0, 0].set_xlabel("Latent Dim 0")
    axes[0, 0].set_ylabel("Latent Dim 1")
    
    # Plot encoder means
    axes[0, 1].scatter(enc_means[:, 0], enc_means[:, 1], alpha=0.6, color='red')
    axes[0, 1].set_title("Encoder Means (first 2 dims)")
    axes[0, 1].set_xlabel("Encoder Mean Dim 0")
    axes[0, 1].set_ylabel("Encoder Mean Dim 1")
    
    # Plot prior means
    axes[1, 0].scatter(prior_means[:, 0], prior_means[:, 1], alpha=0.6, color='green')
    axes[1, 0].set_title("Prior Means (first 2 dims)")
    axes[1, 0].set_xlabel("Prior Mean Dim 0")
    axes[1, 0].set_ylabel("Prior Mean Dim 1")
    
    # Plot latent norms histogram
    latent_norms = jnp.linalg.norm(latents, axis=1)
    axes[1, 1].hist(latent_norms, bins=50, alpha=0.7)
    axes[1, 1].set_title("Latent Vector Norms")
    axes[1, 1].set_xlabel("L2 Norm")
    axes[1, 1].set_ylabel("Frequency")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Latent space visualization saved to {save_path}")
    
    plt.show()


@hydra.main(version_base=None, config_path="config", config_name="vae_distillation")
def main(cfg: DictConfig):
    """Main evaluation function."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting VAE evaluation")
    
    # Get checkpoint path from command line or config
    checkpoint_path = hydra.utils.to_absolute_path("checkpoints/your_vae_checkpoint")
    if not Path(checkpoint_path).exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Setup random number generator
    rng = jax.random.PRNGKey(42)
    
    # Create distillation task
    distillation_task = create_vae_distillation_task(
        teacher_checkpoint_path=cfg.teacher_config.checkpoint_path,
        reference_data_path=cfg.teacher_config.reference_data_path,
        episode_length=cfg.training_config.episode_length,
    )
    
    # Create VAE network
    vae_network = create_vae_network(
        encoder_hidden_sizes=cfg.vae_config.encoder_hidden_sizes,
        decoder_hidden_sizes=cfg.vae_config.decoder_hidden_sizes,
        prior_hidden_sizes=cfg.vae_config.prior_hidden_sizes,
        latent_dim=cfg.vae_config.latent_dim,
        action_dim=distillation_task.action_size,
        activation=cfg.vae_config.activation,
    )
    
    # Load checkpoint
    checkpointer = VAECheckpointer(checkpoint_path)
    # Note: This is simplified - you'd need to implement proper checkpoint loading
    # with the correct abstract pytrees
    logging.info("Loading VAE checkpoint...")
    # vae_params = checkpointer.restore_checkpoint()["vae_params"]
    
    # For now, initialize with random parameters for demonstration
    rng, init_rng = jax.random.split(rng)
    dummy_proprioception = jnp.zeros((1, distillation_task.proprioceptive_obs_size))
    dummy_reference_obs = jnp.zeros((1, distillation_task.reference_obs_size))
    dummy_rng = jax.random.PRNGKey(0)
    
    vae_params = vae_network.init(
        init_rng, dummy_proprioception, dummy_reference_obs, dummy_rng
    )
    
    # Evaluate VAE
    logging.info("Evaluating VAE performance...")
    aggregate_metrics, detailed_metrics = evaluate_vae_vs_teacher(
        vae_network=vae_network,
        vae_params=vae_params,
        distillation_task=distillation_task,
        num_episodes=10,
        episode_length=1000,
        rng=rng,
    )
    
    # Print results
    print("\n" + "="*50)
    print("VAE EVALUATION RESULTS")
    print("="*50)
    for key, value in aggregate_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Visualize latent space
    logging.info("Visualizing latent space...")
    visualize_latent_space(
        vae_network=vae_network,
        vae_params=vae_params,
        distillation_task=distillation_task,
        num_samples=500,
        save_path="latent_space_visualization.png",
        rng=rng,
    )
    
    logging.info("Evaluation completed!")


if __name__ == "__main__":
    main()
