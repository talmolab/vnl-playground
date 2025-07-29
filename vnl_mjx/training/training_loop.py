"""Main VAE distillation training loop."""

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training.train_state import TrainState
from typing import Dict, Any, Tuple, Callable
import functools
import logging
from tqdm import tqdm

from vnl_mjx.models.vae import VAE
from vnl_mjx.training.losses import combined_vae_loss, compute_vae_metrics
from vnl_mjx.tasks.vae_distillation import VAEDistillationTask


class VAETrainingState(TrainState):
    """Training state for VAE with additional fields."""
    metrics: Dict[str, float] = flax.struct.field(default_factory=dict)


def create_vae_train_state(
    vae_network: VAE,
    learning_rate: float,
    proprioceptive_obs_size: int,
    reference_obs_size: int,
    rng: jnp.ndarray,
) -> VAETrainingState:
    """Create VAE training state.
    
    Args:
        vae_network: VAE network
        learning_rate: Learning rate
        proprioceptive_obs_size: Size of proprioceptive observations
        reference_obs_size: Size of reference observations
        rng: Random number generator
    
    Returns:
        VAE training state
    """
    # Initialize network with dummy inputs
    dummy_proprioception = jnp.zeros((1, proprioceptive_obs_size))
    dummy_reference_obs = jnp.zeros((1, reference_obs_size))
    dummy_rng = jax.random.PRNGKey(0)
    
    params = vae_network.init(
        rng, 
        dummy_proprioception, 
        dummy_reference_obs, 
        dummy_rng
    )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    return VAETrainingState.create(
        apply_fn=vae_network.apply,
        params=params,
        tx=optimizer,
    )


def create_training_step_fn(
    kl_weight: float,
    action_weight: float = 1.0,
) -> Callable:
    """Create training step function.
    
    Args:
        kl_weight: Weight for KL divergence loss
        action_weight: Weight for action MSE loss
    
    Returns:
        Training step function
    """
    
    def loss_fn(params, vae_network_apply_fn, batch, rng):
        """Compute VAE loss for a batch."""
        proprioceptive_obs = batch["proprioceptive_obs"]
        reference_obs = batch["reference_obs"]
        teacher_actions = batch["teacher_actions"]
        
        # VAE forward pass
        vae_outputs = vae_network_apply_fn(
            params, proprioceptive_obs, reference_obs, rng
        )
        
        # Compute losses
        losses = combined_vae_loss(
            teacher_actions=teacher_actions,
            student_actions=vae_outputs["actions"],
            enc_mean=vae_outputs["enc_mean"],
            enc_logvar=vae_outputs["enc_logvar"],
            prior_mean=vae_outputs["prior_mean"],
            prior_logvar=vae_outputs["prior_logvar"],
            kl_weight=kl_weight,
            action_weight=action_weight,
        )
        
        # Compute additional metrics
        metrics = compute_vae_metrics(vae_outputs, teacher_actions)
        losses.update(metrics)
        
        return losses["total_loss"], losses
    
    @jax.jit
    def training_step(state: VAETrainingState, batch: Dict[str, jnp.ndarray], rng: jnp.ndarray):
        """Single training step."""
        loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = loss_and_grad_fn(
            state.params, state.apply_fn, batch, rng
        )
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(metrics=metrics)
        
        return new_state, metrics
    
    return training_step

def create_parallel_distillation_functions(
    task: VAEDistillationTask,
) -> Tuple[Callable, Callable]:
    """Create parallel functions for VAE distillation training.
    
    Args:
        task: VAE distillation task
    
    Returns:
        (parallel_reset, parallel_step) functions
    """
    
    @jax.jit
    def parallel_reset(rng_keys: jnp.ndarray, clip_indices: Optional[jnp.ndarray] = None):
        """Reset all environments in parallel."""
        if clip_indices is not None:
            return jax.vmap(task.reset, in_axes=(0, 0))(rng_keys, clip_indices)
        else:
            return jax.vmap(task.reset, in_axes=(0, None))(rng_keys, None)
    
    @jax.jit
    def parallel_step(distill_states, student_actions):
        """Step all environments in parallel."""
        return jax.vmap(task.step)(distill_states, student_actions)
    
    return parallel_reset, parallel_step


def train_vae(
    vae_network: VAE,
    distillation_task: VAEDistillationTask,
    num_envs: int,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    kl_weight: float,
    eval_every: int,
    checkpoint_every: int,
    checkpointer: Any,
    wandb_logger: Any = None,
    rng: jnp.ndarray | None = None,
) -> VAETrainingState:
    """Main VAE training loop.
    
    Args:
        vae_network: VAE network to train
        distillation_task: VAE distillation task environment
        num_envs: Number of parallel environments
        num_steps: Total training steps
        batch_size: Batch size for training
        learning_rate: Learning rate
        kl_weight: Weight for KL divergence loss
        eval_every: Evaluation frequency
        checkpoint_every: Checkpointing frequency
        checkpointer: Checkpoint manager
        wandb_logger: Weights & Biases logger
        rng: Random number generator
    
    Returns:
        Final training state
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Split rng key
    rng, key_init = jax.random.split(rng)
    
    # Initialize training state
    train_state = create_vae_train_state(
        vae_network=vae_network,
        learning_rate=learning_rate,
        proprioceptive_obs_size=distillation_task.proprioceptive_obs_size,
        reference_obs_size=distillation_task.reference_obs_size,
        rng=key_init,
    )
    
    # Create training step function
    training_step_fn = create_training_step_fn(kl_weight=kl_weight)
    
    # Create parallel environment functions
    parallel_reset, parallel_step = create_parallel_distillation_functions(distillation_task)
    
    # Initialize environments
    rng, env_rng = jax.random.split(rng)
    env_rngs = jax.random.split(env_rng, num_envs)
    distill_states = parallel_reset(env_rngs)
    
    # Training loop
    logging.info(f"Starting VAE training for {num_steps} steps with {num_envs} environments")
    
    for step in tqdm(range(num_steps), desc="Training VAE"):
        # Collect batch of data
        batch_data = {
            "proprioceptive_obs": [],
            "reference_obs": [],
            "teacher_actions": [],
        }
        
        # Collect trajectories
        for _ in range(batch_size // num_envs):
            # Store current data
            batch_data["proprioceptive_obs"].append(
                jnp.stack([ds["proprioceptive_obs"] for ds in distill_states])
            )
            batch_data["reference_obs"].append(
                jnp.stack([ds["reference_obs"] for ds in distill_states])
            )
            batch_data["teacher_actions"].append(
                jnp.stack([ds["teacher_action"] for ds in distill_states])
            )
            
            # Generate VAE actions for stepping
            rng, vae_rng = jax.random.split(rng)
            vae_rngs = jax.random.split(vae_rng, num_envs)
            
            # Get VAE actions (without gradients for environment stepping)
            proprioceptive_batch = jnp.stack([ds["proprioceptive_obs"] for ds in distill_states])
            reference_batch = jnp.stack([ds["reference_obs"] for ds in distill_states])
            
            vae_outputs = jax.vmap(train_state.apply_fn, in_axes=(None, 0, 0, 0))(
                train_state.params, proprioceptive_batch, reference_batch, vae_rngs
            )
            student_actions = vae_outputs["actions"]
            
            # Step environments
            distill_states = parallel_step(distill_states, student_actions)
        
        # Concatenate batch data
        batch = {
            key: jnp.concatenate(value_list, axis=0)
            for key, value_list in batch_data.items()
        }
        
        # Training step
        rng, train_rng = jax.random.split(rng)
        train_state, metrics = training_step_fn(train_state, batch, train_rng)
        
        # Logging
        if step % 100 == 0:
            logging.info(f"Step {step}: Loss = {metrics['total_loss']:.4f}, "
                        f"Action Loss = {metrics['action_loss']:.4f}, "
                        f"KL Loss = {metrics['kl_loss']:.4f}")
            
            if wandb_logger:
                wandb_logger.log(metrics, step=step)
        
        # Checkpointing
        if step > 0 and step % checkpoint_every == 0:
            checkpointer.save_checkpoint(
                step=step,
                vae_params=train_state.params,
                optimizer_state=train_state.opt_state,
                config={},  # Add config as needed
                metrics=metrics,
            )
        
        # Evaluation
        if step % eval_every == 0:
            # Add evaluation logic here if needed
            pass
    
    return train_state
