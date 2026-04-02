"""Intention network architectures for VAE-style imitation learning.

This module provides encoder-decoder neural network architectures.
The key components are:

- Encoder: Maps reference trajectory observations to a latent intention space
- Decoder: Maps latent intentions + proprioceptive state to action parameters
- IntentionNetwork: Full VAE combining encoder and decoder with reparameterization

The architecture enables learning from motion capture data by encoding trajectory
information into a compact latent space that conditions the policy.

The outer policy API (make_intention_policy) accepts nested observations
{'state': {...}, 'privileged_state': {...}} and handles key selection and
normalization. The IntentionNetwork module itself receives the already-selected
inner dict {'task_obs': ..., 'proprioception': ...}.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from brax.training import networks, types
from brax.training.acme import running_statistics
from flax import linen as nn

from track_mjx.agent.observation_utils import (
    normalizer_select,
)


class Encoder(nn.Module):
    """VAE encoder that maps observations to latent distribution parameters.

    Processes reference trajectory observations through an MLP with LayerNorm
    to produce mean and log-variance of the latent intention distribution.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latents: Dimension of the latent intention space.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[tuple[jnp.ndarray, jnp.ndarray], dict]:
        activations = {}
        # For each layer in the sequence
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        if get_activation:
            activations["mean"] = mean_x
            activations["logvar"] = logvar_x
            return (mean_x, logvar_x), activations
        return mean_x, logvar_x


class Decoder(nn.Module):
    """VAE decoder that maps latent intentions to action distribution parameters.

    Processes concatenated latent intention and proprioceptive observations
    through an MLP to produce action distribution parameters.

    Attributes:
        layer_sizes: Layer dimensions including final output size.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        activate_final: Whether to apply activation after final layer.
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> tuple[jnp.ndarray, dict]:
        if get_activation:
            activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
                if get_activation:
                    activations[f"layer_{i}"] = x
        if get_activation:
            return x, activations
        return x, {}


def reparameterize_single(
    rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """Sample from Gaussian using reparameterization trick (single sample).

    Enables backpropagation through stochastic sampling by expressing the
    sample as a deterministic function of the parameters plus noise.

    Args:
        rng: JAX random key for sampling (shape [2]).
        mean: Mean of the Gaussian distribution (shape [latent_dim]).
        logvar: Log-variance of the Gaussian distribution (shape [latent_dim]).

    Returns:
        Sampled latent vector: mean + std * epsilon, where epsilon ~ N(0, I).
    """
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


def reparameterize(
    rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """Sample from Gaussian using reparameterization trick (batched).

    Supports both single key (broadcasted) and per-sample keys for
    deterministic replay.

    Args:
        rng: JAX random key(s) for sampling. Either shape [2] (single key
            for all samples) or [batch_size, 2] (per-sample keys).
        mean: Mean of the Gaussian distribution, shape [latent_dim] (unbatched)
            or [batch_size, latent_dim] (batched).
        logvar: Log-variance of the Gaussian distribution, same shape as mean.

    Returns:
        Sampled latent vectors with same shape as mean.
    """
    if rng.ndim == 1:
        # Single key - use original behavior
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, logvar.shape)
        return mean + eps * std
    elif mean.ndim == 1:
        # Per-sample keys but unbatched mean/logvar - use first key
        # This is a fallback safety for shape mismatches
        return reparameterize_single(rng[0], mean, logvar)
    else:
        # Per-sample keys with batched mean/logvar - vmap over batch dimension
        return jax.vmap(reparameterize_single)(rng, mean, logvar)


class IntentionNetwork(nn.Module):
    """Full VAE model combining encoder and decoder for intention-based policy.

    The network receives inner observation dicts (already selected and normalized
    by the outer apply function). The encoder processes obs['task_obs'] to produce
    latent intentions, which are then concatenated with obs['proprioception'] and
    decoded into action distribution parameters.

    Attributes:
        encoder_layers: Hidden layer sizes for the encoder MLP.
        decoder_layers: Layer sizes for decoder (excluding action output).
        latents: Dimension of the latent intention space.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int = 60
    proprioception_noise_std: float = 0.0

    def setup(self):
        """Initialize encoder and decoder submodules."""
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key: jax.Array,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        # Access inner observations: already normalized and selected for 'state'
        # obs is the inner dict: {'task_obs': ..., 'proprioception': ...}
        traj = obs["task_obs"]
        egocentric_obs = obs["proprioception"]

        # Check if observations are actually batched (based on normalized obs shape)
        obs_is_batched = traj.ndim >= 2

        # Handle key splitting based on both key shape AND observation shape
        if key.ndim == 1:
            # Single key - split for encoder
            encoder_rng, noise_rng = jax.random.split(key)
        elif not obs_is_batched:
            # Per-sample keys but unbatched observation - use first key
            # This can happen when key batching was determined from nested obs structure
            # before normalization flattened it to unbatched
            encoder_rng, noise_rng = jax.random.split(key[0])
        else:
            # Per-sample keys [batch_size, 2] - vmap split over batch
            encoder_rng, noise_rng = jax.vmap(jax.random.split)(key).swapaxes(0, 1)

        if not deterministic and self.proprioception_noise_std > 0.0:
            if noise_rng.ndim == 1:
                noise = jax.random.normal(noise_rng, egocentric_obs.shape)
            elif not obs_is_batched:
                noise = jax.random.normal(noise_rng[0], egocentric_obs.shape)
            else:
                noise = jax.vmap(
                    lambda rng_key, obs_i: jax.random.normal(rng_key, obs_i.shape)
                )(noise_rng, egocentric_obs)
            egocentric_obs = egocentric_obs * (
                1.0 + self.proprioception_noise_std * noise
            )

        if get_activation:
            (latent_mean, latent_logvar), encoder_activations = self.encoder(
                traj, get_activation=True
            )
            # Uses mean in the case of deterministic evaluation
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            concatenated = jnp.concatenate([z, egocentric_obs], axis=-1)
            action, decoder_activations = self.decoder(
                concatenated, get_activation=True
            )
            return (
                action,
                latent_mean,
                latent_logvar,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "intention": z,
                },
            )
        else:
            latent_mean, latent_logvar = self.encoder(traj, get_activation=False)
            # Uses mean in the case of deterministic evaluation
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            action, _ = self.decoder(jnp.concatenate([z, egocentric_obs], axis=-1))
            return action, latent_mean, latent_logvar


def make_intention_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    proprioception_noise_std: float = 0.0,
    policy_obs_key: str = "state",
) -> networks.FeedForwardNetwork:
    """Create an intention-based policy network.

    Constructs an encoder-decoder VAE policy where the encoder processes
    obs[policy_obs_key]['task_obs'] and the decoder generates action
    parameters conditioned on latent intentions and obs[policy_obs_key]['proprioception'].

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        latent_size: Dimension of the latent intention space.
        obs_sizes: Dict with 'task_obs' and 'proprioception' sizes.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        proprioception_noise_std: Stddev for multiplicative Gaussian noise on
            proprioception during stochastic training passes.
        policy_obs_key: Top-level observation key for policy (default: 'state').

    Returns:
        FeedForwardNetwork with init and apply methods. The apply function
        returns (action_params, latent_mean, latent_logvar).
    """

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes)
        + [action_param_size],  # add action size to the last layer
        latents=latent_size,
        proprioception_noise_std=proprioception_noise_std,
    )

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, Mapping[str, jnp.ndarray]],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply policy with observation normalization.

        Args:
            processor_params: RunningStatisticsState with pytree structure matching obs.
            policy_params: Policy network parameters.
            obs: Nested observation dict.
            key: Random key for sampling.
            deterministic: If True, use mean instead of sampling.
            get_activation: If True, return activations.
        """
        policy_normalizer = normalizer_select(processor_params, policy_obs_key)
        normalized_obs = running_statistics.normalize(
            obs[policy_obs_key], policy_normalizer
        )
        return policy_module.apply(
            policy_params,
            obs=normalized_obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    # Create dummy inner observation for initialization (already selected/normalized)
    dummy_obs = {
        "task_obs": jnp.zeros((1, obs_sizes["task_obs"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )


def make_decoder_policy(
    param_size: int,
    decoder_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> networks.FeedForwardNetwork:
    """Create a decoder-only policy network for downstream tasks.

    Creates a standalone decoder policy that can be used with externally
    provided latent intentions. Useful for transfer learning or hierarchical
    control where intentions come from a separate module.

    The normalizer params only apply to the proprioceptive portion of the
    observation (latent intentions are not normalized).

    Args:
        param_size: Output dimension for action distribution parameters.
        decoder_obs_size: Input dimension (latent_size + proprioceptive_size).
        preprocess_observations_fn: Normalization function for proprioceptive obs.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.

    Returns:
        FeedForwardNetwork with init and apply methods.
    """
    policy_module = Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [param_size],
    )

    def apply(processor_params, policy_params, obs):
        """Apply decoder with selective normalization of proprioceptive obs."""
        # Split obs into latent (unnormalized) and proprioceptive (normalized)
        latent_obs = obs[..., : -processor_params.mean.shape[-1]]
        proprio_obs = preprocess_observations_fn(
            obs[..., -processor_params.mean.shape[-1] :], processor_params
        )
        obs = jnp.concatenate([latent_obs, proprio_obs], axis=-1)
        return policy_module.apply(policy_params, x=obs, get_activation=False)

    dummy_obs = jnp.zeros((1, decoder_obs_size))

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=apply,
    )


    # Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO loss functions for intention-based imitation learning.

This module provides loss computation for PPO training with VAE-style
intention networks, including:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate policy loss
- Value function loss
- Entropy bonus
- KL divergence loss for VAE latent space (with autoregressive prior)
"""

from collections.abc import Callable
from typing import Any

import flax
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params


@flax.struct.dataclass
class PPONetworkParams:
    """Container for PPO network parameters.

    Attributes:
        policy: Policy network parameters (encoder + decoder for intention networks).
        value: Value function network parameters.
    """

    policy: Params
    value: Params


def compute_kl_to_gaussian_prior(
    latent_mean: jnp.ndarray,
    latent_logvar: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL divergence from latent distribution to standard Gaussian prior.

    Computes KL(q(z|x) || p(z)) where q(z|x) = N(latent_mean, exp(latent_logvar))
    and p(z) = N(0, I). This encourages the latent space to be regularized
    towards a standard Gaussian.

    Args:
        latent_mean: Mean of the latent distribution, shape [T, B, D].
        latent_logvar: Log variance of the latent distribution, shape [T, B, D].

    Returns:
        Scalar KL divergence loss (mean over all dimensions and batch).
    """
    return -0.5 * jnp.mean(
        1 + latent_logvar - jnp.square(latent_mean) - jnp.exp(latent_logvar)
    )


def compute_ar1_temporal_loss(
    latent_mean: jnp.ndarray,
    discount: jnp.ndarray,
    truncation: jnp.ndarray,
) -> jnp.ndarray:
    """Compute AR(1) temporal smoothness loss for latent intentions.

    Encourages temporal consistency in the latent space by penalizing
    large changes between consecutive timesteps. Pairs that cross episode
    boundaries (done or truncated) are masked out.

    Args:
        latent_mean: Mean of the latent distribution, shape [T, B, D].
        discount: Discount signal from transitions, shape [T, B].
            0 indicates episode end.
        truncation: Truncation signal, shape [T, B].
            1 indicates episode was truncated (not terminated).

    Returns:
        Scalar L2 loss between consecutive latent means (masked and normalized).
    """
    z_prev = latent_mean[:-1]  # z_0, ..., z_{T-2}
    z_curr = latent_mean[1:]  # z_1, ..., z_{T-1}
    # Valid if episode continues: not done (discount=1) AND not truncated (truncation=0)
    valid_mask = discount[:-1] * (1 - truncation[:-1])
    l2_diff = jnp.mean(jnp.square(z_curr - z_prev), axis=-1)  # [T-1, B]
    masked_l2 = l2_diff * valid_mask
    return jnp.sum(masked_l2) / jnp.maximum(jnp.sum(valid_mask), 1.0)


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a variance-bias tradeoff for advantage estimation controlled
    by lambda_. When lambda_=0, this reduces to 1-step TD. When lambda_=1,
    this uses the full Monte Carlo return.

    Args:
        truncation: Truncation signal, shape [T, B]. 1 if episode was truncated
            (not terminated) at this step.
        termination: Termination signal, shape [T, B]. 1 if episode ended.
        rewards: Rewards, shape [T, B].
        values: Value function estimates V(s_t), shape [T, B].
        bootstrap_value: Value estimate at time T for bootstrapping, shape [B].
        lambda_: GAE lambda parameter (0=TD(0), 1=Monte Carlo). Defaults to 1.0.
        discount: Discount factor (gamma). Defaults to 0.99.

    Returns:
        Tuple of:
            - vs: Value targets for training, shape [T, B].
            - advantages: GAE advantages, shape [T, B].
    """
    truncation_mask = 1 - truncation

    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), acc

    (_, _), vs_minus_v_xs = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    # Add V(x_s) to get v_s
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (
        rewards + discount * (1 - termination) * vs_t_plus_1 - values
    ) * truncation_mask

    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: ppo_networks.PPONetworks,
    entropy_cost: float = 1e-4,
    latent_kl_weight: float = 1e-3,
    latent_ar1_weight: float = 1e-3,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    latent_kl_schedule: Callable[[int], float] | None = None,
    latent_ar1_schedule: Callable[[int], float] | None = None,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute PPO loss with VAE KL divergence for intention networks.

    Computes the standard PPO clipped surrogate loss plus:
    - Value function MSE loss
    - Entropy bonus for exploration
    - KL divergence loss for VAE latent space regularization
    - L2 temporal smoothness loss between consecutive latent means,
      encouraging temporal consistency in the latent intentions.

    Args:
        params: PPO network parameters (policy and value).
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch with shape [B, T]. Required extra fields:
            - data.extras["state_extras"]["truncation"]
            - data.extras["policy_extras"]["raw_action"]
            - data.extras["policy_extras"]["log_prob"]
        rng: JAX random key.
        step: Current training step (for KL schedule).
        ppo_network: PPO network container with policy, value, and distribution.
        entropy_cost: Entropy bonus coefficient (higher = more exploration).
        latent_kl_weight: Weight for KL divergence to standard Gaussian prior.
        latent_ar1_weight: Weight for AR(1) temporal smoothness loss.
        discounting: Discount factor (gamma) for GAE.
        reward_scaling: Multiplier applied to rewards.
        gae_lambda: GAE lambda parameter.
        clipping_epsilon: PPO clipping range for policy ratio.
        normalize_advantage: Whether to normalize advantages to zero mean/unit std.
        vf_coefficient: Coefficient for value function loss
        latent_kl_schedule: Optional schedule function(step) -> latent_kl_weight.
        latent_ar1_schedule: Optional schedule function(step) -> latent_ar1_weight.

    Returns:
        Tuple of:
            - total_loss: Scalar loss value.
            - metrics: Dict with individual loss components for logging.
    """

    _, policy_key, entropy_key = jax.random.split(rng, 3)
    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Check for stored policy_rng for deterministic stochastic layer replay
    policy_rng = data.extras["policy_extras"].get("policy_rng")

    if policy_rng is None:
        # Standard path: use fresh RNG
        policy_logits, latent_mean, latent_logvar = policy_apply(
            normalizer_params, params.policy, data.observation, policy_key
        )
    else:
        # Deterministic replay: use stored per-sample RNGs
        # policy_rng has shape [T, B, 2] after swapaxes
        # data.observation is a dict where each leaf has shape [T, B, obs_dim]
        # Flatten [T, B] into single batch dim to avoid vmap (better for buffer donation)
        obs_leaf = jax.tree_util.tree_leaves(data.observation)[0]
        T, B = obs_leaf.shape[:2]

        # Flatten observations [T, B, ...] -> [T*B, ...]
        flat_obs = jax.tree_util.tree_map(
            lambda x: x.reshape((T * B,) + x.shape[2:]),
            data.observation,
        )
        # Flatten keys [T, B, 2] -> [T*B, 2]
        flat_rng = policy_rng.reshape((T * B, 2))

        # Single call with combined batch (policy handles per-sample keys natively)
        flat_logits, flat_mean, flat_logvar = policy_apply(
            normalizer_params, params.policy, flat_obs, flat_rng
        )

        # Reshape outputs back to [T, B, ...]
        policy_logits = flat_logits.reshape((T, B) + flat_logits.shape[1:])
        latent_mean = flat_mean.reshape((T, B) + flat_mean.shape[1:])
        latent_logvar = flat_logvar.reshape((T, B) + flat_logvar.shape[1:])

    baseline = value_apply(normalizer_params, params.value, data.observation)

    # Get the last timestep from dict observation (tree_map handles dict structure)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    # Entropy reward
    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, entropy_key)
    )
    entropy_loss = entropy_cost * -entropy

    # KL Divergence for latent layer
    current_kl_weight = latent_kl_weight
    current_ar1_weight = latent_ar1_weight
    if latent_kl_schedule is not None:
        current_kl_weight = latent_kl_schedule(step)
    if latent_ar1_schedule is not None:
        current_ar1_weight = latent_ar1_schedule(step)

    # KL divergence to standard Gaussian prior N(0, I)
    kl_gaussian = compute_kl_to_gaussian_prior(latent_mean, latent_logvar)

    # L2 loss to previous latent mean (AR(1) temporal smoothness)
    ar1_loss = compute_ar1_temporal_loss(latent_mean, data.discount, truncation)

    # Apply weights to each loss term
    kl_gaussian_weighted = current_kl_weight * kl_gaussian
    ar1_loss_weighted = current_ar1_weight * ar1_loss
    latent_loss = kl_gaussian_weighted + ar1_loss_weighted

    total_loss = policy_loss + v_loss + entropy_loss + latent_loss

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "total_latent_loss": latent_loss,
        "latent_ar1_loss": ar1_loss_weighted,
        "latent_kl_loss": kl_gaussian_weighted,
        "entropy_loss": entropy_loss,
        "latent_kl_weight": current_kl_weight,
        "latent_ar1_weight": current_ar1_weight,
    }


def create_ramp_schedule(
    max_value: float = 0.1,
    min_value: float = 0.0001,
    ramp_steps: int = 1000,
    warmup_steps: int = 0,
    schedule: str = "linear",
    period: int = 45,
) -> Callable[[int], float]:
    """Create a schedule function for KL weight annealing.

    Supports three schedule types:
    - "linear": Ramps from min_value to max_value over ramp_steps after warmup.
    - "cosine": Cyclic cosine oscillation between min_value and max_value.
    - "sine": Cyclic sine oscillation between min_value and max_value.

    Args:
        max_value: Maximum schedule value. Defaults to 0.1.
        min_value: Minimum schedule value. Defaults to 0.0001.
        ramp_steps: Number of steps for linear ramp (linear schedule only).
        warmup_steps: Steps to hold at min_value before ramping (linear only).
        schedule: Schedule type - "linear", "cosine", or "sine".
        period: Period of oscillation in steps (cyclic schedules only).

    Returns:
        A function mapping step -> schedule value.

    Raises:
        ValueError: If schedule is not "linear", "cosine", or "sine".
    """

    def schedule_fn(step: int) -> float:
        step = jnp.asarray(step, dtype=jnp.float32)

        if schedule == "linear":
            progress = jnp.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0)
            is_warmup = step < warmup_steps
            return jnp.where(
                is_warmup, min_value, min_value + progress * (max_value - min_value)
            )
        elif schedule == "cosine":
            angle = (2 * jnp.pi * step) / period
            amplitude = (max_value - min_value) / 2
            midpoint = (max_value + min_value) / 2
            return midpoint + amplitude * jnp.cos(angle)
        elif schedule == "sine":
            # Subtract π/2 so sine starts at min_value (sin(-π/2) = -1)
            angle = (2 * jnp.pi * step) / period - jnp.pi / 2
            amplitude = (max_value - min_value) / 2
            midpoint = (max_value + min_value) / 2
            return midpoint + amplitude * jnp.sin(angle)
        else:
            raise ValueError(
                f"schedule must be 'linear', 'cosine', or 'sine', not {schedule}"
            )

    return schedule_fn