"""Loss functions for VAE imitation learning."""

import jax.numpy as jnp
import jax
from typing import Dict, Any


def action_mse_loss(teacher_actions: jnp.ndarray, student_actions: jnp.ndarray) -> jnp.ndarray:
    """MSE loss between teacher and student actions.
    
    Args:
        teacher_actions: Actions from frozen teacher network
        student_actions: Actions from VAE decoder
    
    Returns:
        Mean squared error loss
    """
    return jnp.mean((teacher_actions - student_actions) ** 2)


def kl_divergence_loss(
    enc_mean: jnp.ndarray, 
    enc_logvar: jnp.ndarray, 
    prior_mean: jnp.ndarray, 
    prior_logvar: jnp.ndarray
) -> jnp.ndarray:
    """KL divergence between encoder and prior distributions.
    
    Args:
        enc_mean: Mean of encoder distribution
        enc_logvar: Log variance of encoder distribution
        prior_mean: Mean of prior distribution
        prior_logvar: Log variance of prior distribution
    
    Returns:
        KL divergence loss
    """
    # KL(q||p) = -0.5 * sum(1 + log(var_q/var_p) - (mu_q - mu_p)^2/var_p - var_q/var_p)
    var_ratio = jnp.exp(enc_logvar - prior_logvar)
    mean_diff_sq = (enc_mean - prior_mean) ** 2 / jnp.exp(prior_logvar)
    
    kl = -0.5 * jnp.sum(
        1 + enc_logvar - prior_logvar - mean_diff_sq - var_ratio,
        axis=-1
    )
    return jnp.mean(kl)


def combined_vae_loss(
    teacher_actions: jnp.ndarray,
    student_actions: jnp.ndarray,
    enc_mean: jnp.ndarray,
    enc_logvar: jnp.ndarray,
    prior_mean: jnp.ndarray,
    prior_logvar: jnp.ndarray,
    kl_weight: float = 1.0,
    action_weight: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """Combined VAE loss function.
    
    Args:
        teacher_actions: Actions from teacher network
        student_actions: Actions from VAE
        enc_mean: Encoder mean
        enc_logvar: Encoder log variance
        prior_mean: Prior mean
        prior_logvar: Prior log variance
        kl_weight: Weight for KL divergence term
        action_weight: Weight for action MSE term
    
    Returns:
        Dictionary with individual losses and total loss
    """
    action_loss = action_mse_loss(teacher_actions, student_actions)
    kl_loss = kl_divergence_loss(enc_mean, enc_logvar, prior_mean, prior_logvar)
    
    total_loss = action_weight * action_loss + kl_weight * kl_loss
    
    return {
        "total_loss": total_loss,
        "action_loss": action_loss,
        "kl_loss": kl_loss,
        "action_mse": action_loss,  # For logging
        "kl_divergence": kl_loss,  # For logging
    }


def compute_vae_metrics(vae_outputs: Dict[str, jnp.ndarray], teacher_actions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Compute additional metrics for monitoring training.
    
    Args:
        vae_outputs: Outputs from VAE forward pass
        teacher_actions: Teacher actions for comparison
    
    Returns:
        Dictionary of metrics
    """
    student_actions = vae_outputs["actions"]
    
    # Action statistics
    action_mean_error = jnp.mean(jnp.abs(teacher_actions - student_actions))
    action_max_error = jnp.max(jnp.abs(teacher_actions - student_actions))
    
    # Latent statistics
    latent_norm = jnp.mean(jnp.linalg.norm(vae_outputs["latent"], axis=-1))
    enc_var = jnp.mean(jnp.exp(vae_outputs["enc_logvar"]))
    prior_var = jnp.mean(jnp.exp(vae_outputs["prior_logvar"]))
    
    return {
        "action_mean_error": action_mean_error,
        "action_max_error": action_max_error,
        "latent_norm": latent_norm,
        "encoder_variance": enc_var,
        "prior_variance": prior_var,
    }
