"""VAE models for imitation learning."""

from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from jax import random


class VAEEncoder(nn.Module):
    """MLP encoder that takes proprioception + reference_obs -> latent mean/logvar."""
    
    hidden_layer_sizes: Sequence[int]
    latent_dim: int
    activation: str = "tanh"
    
    @nn.compact
    def __call__(self, proprioception: jnp.ndarray, reference_obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            proprioception: Proprioceptive observations
            reference_obs: Reference observations (goal state)
        
        Returns:
            (mean, logvar): Latent distribution parameters
        """
        # Concatenate inputs
        x = jnp.concatenate([proprioception, reference_obs], axis=-1)
        
        # MLP layers
        for hidden_size in self.hidden_layer_sizes:
            x = nn.Dense(
                hidden_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = getattr(nn, self.activation)(x)
        
        # Output layers for mean and logvar
        mean = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        
        logvar = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        
        return mean, logvar


class VAEDecoder(nn.Module):
    """MLP decoder that takes proprioception + latent -> actions."""
    
    hidden_layer_sizes: Sequence[int]
    action_dim: int
    activation: str = "tanh"
    
    @nn.compact
    def __call__(self, proprioception: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            proprioception: Proprioceptive observations
            latent: Sampled latent variable
        
        Returns:
            actions: Predicted actions
        """
        # Concatenate inputs
        x = jnp.concatenate([proprioception, latent], axis=-1)
        
        # MLP layers
        for hidden_size in self.hidden_layer_sizes:
            x = nn.Dense(
                hidden_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = getattr(nn, self.activation)(x)
        
        # Output layer for actions
        actions = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        
        return actions


class VAEPrior(nn.Module):
    """MLP prior that takes proprioception -> latent mean/logvar."""
    
    hidden_layer_sizes: Sequence[int]
    latent_dim: int
    activation: str = "tanh"
    
    @nn.compact
    def __call__(self, proprioception: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            proprioception: Proprioceptive observations
        
        Returns:
            (mean, logvar): Prior distribution parameters
        """
        x = proprioception
        
        # MLP layers
        for hidden_size in self.hidden_layer_sizes:
            x = nn.Dense(
                hidden_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = getattr(nn, self.activation)(x)
        
        # Output layers for mean and logvar
        mean = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        
        logvar = nn.Dense(
            self.latent_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        
        return mean, logvar


class VAE(nn.Module):
    """Complete VAE combining encoder, decoder, and prior."""
    
    encoder_hidden_sizes: Sequence[int]
    decoder_hidden_sizes: Sequence[int]
    prior_hidden_sizes: Sequence[int]
    latent_dim: int
    action_dim: int
    activation: str = "tanh"
    
    def setup(self):
        self.encoder = VAEEncoder(
            hidden_layer_sizes=self.encoder_hidden_sizes,
            latent_dim=self.latent_dim,
            activation=self.activation,
        )
        self.decoder = VAEDecoder(
            hidden_layer_sizes=self.decoder_hidden_sizes,
            action_dim=self.action_dim,
            activation=self.activation,
        )
        self.prior = VAEPrior(
            hidden_layer_sizes=self.prior_hidden_sizes,
            latent_dim=self.latent_dim,
            activation=self.activation,
        )
    
    def encode(self, proprioception: jnp.ndarray, reference_obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode to latent distribution."""
        return self.encoder(proprioception, reference_obs)
    
    def decode(self, proprioception: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:
        """Decode to actions."""
        return self.decoder(proprioception, latent)
    
    def prior_distribution(self, proprioception: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get prior distribution."""
        return self.prior(proprioception)
    
    def reparameterize(self, mean: jnp.ndarray, logvar: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        """Reparameterization trick."""
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, mean.shape)
        return mean + eps * std
    
    def __call__(self, proprioception: jnp.ndarray, reference_obs: jnp.ndarray, rng: jnp.ndarray) -> dict:
        """Full forward pass."""
        # Encoder
        enc_mean, enc_logvar = self.encode(proprioception, reference_obs)
        
        # Prior
        prior_mean, prior_logvar = self.prior_distribution(proprioception)
        
        # Sample latent
        latent = self.reparameterize(enc_mean, enc_logvar, rng)
        
        # Decoder
        actions = self.decode(proprioception, latent)
        
        return {
            "actions": actions,
            "enc_mean": enc_mean,
            "enc_logvar": enc_logvar,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "latent": latent,
        }


def create_vae_network(
    encoder_hidden_sizes: Sequence[int],
    decoder_hidden_sizes: Sequence[int],
    prior_hidden_sizes: Sequence[int],
    latent_dim: int,
    action_dim: int,
    activation: str = "tanh",
) -> VAE:
    """Factory function to create VAE network."""
    return VAE(
        encoder_hidden_sizes=encoder_hidden_sizes,
        decoder_hidden_sizes=decoder_hidden_sizes,
        prior_hidden_sizes=prior_hidden_sizes,
        latent_dim=latent_dim,
        action_dim=action_dim,
        activation=activation,
    )
