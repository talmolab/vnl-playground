"""Utilities for loading SCAMPER prior checkpoints and creating inference functions.

Provides functions to load frozen prior/decoder networks from SCAMPER's
mlp_prior training stage and create inference callables for use in
PriorHighLevelWrapper during task transfer training.
"""

from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from brax.training import distribution
from brax.training.acme import running_statistics

from scamper.agent.imitation import intention_network
from scamper.agent.mlp_prior.prior_networks import Prior
from scamper.agent.observation_utils import DictRunningStatisticsState


def load_prior_checkpoint(
    checkpoint_path: str,
    step: int | None = None,
) -> Tuple[Dict, Dict, Dict, DictRunningStatisticsState, Dict]:
    """Load encoder, prior, and decoder parameters from an mlp_prior checkpoint.

    Re-exports ``scamper.agent.task_transfer.walk_rear.checkpoint_utils.load_prior_checkpoint``
    for convenient access from vnl-playground.

    Args:
        checkpoint_path: Path to the mlp_prior checkpoint directory.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (encoder_params, prior_params, decoder_params,
                  normalizer_params, config_dict).

        - normalizer_params is a ``DictRunningStatisticsState``.
        - ``config_dict["network_config"]["intention_size"]`` gives the latent size.
        - ``config_dict["env_config"]["ctrl_dt"]`` gives the control timestep.
    """
    from scamper.agent.task_transfer.walk_rear.checkpoint_utils import (
        load_prior_checkpoint as _load_prior_checkpoint,
    )

    return _load_prior_checkpoint(checkpoint_path, step=step)


def make_decoder_inference_fn(
    decoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    config: Dict,
) -> Callable:
    """Create decoder inference function for use in high-level wrappers.

    The decoder takes ``[latent, proprio]`` concatenated input and outputs an
    action (the mode of a ``NormalTanhDistribution``).

    Args:
        decoder_params: Frozen decoder parameters from the mlp_prior checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics.
        config: Checkpoint config dict (as returned by ``load_prior_checkpoint``).

    Returns:
        Function ``(latent_proprio) -> (action, extras)`` where *latent_proprio*
        is ``[latent, proprio]`` concatenated along the last axis.
    """
    action_size = config["network_config"]["action_size"]
    latent_size = config["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = tuple(config["network_config"]["decoder_layer_sizes"])

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
    )

    proprio_normalizer = normalizer_params.proprioception

    def decoder_fn(latent_proprio: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
        """Apply decoder to latent + proprio input.

        Args:
            latent_proprio: Concatenated ``[latent, proprio]`` array.

        Returns:
            Tuple of ``(action, extras_dict)``.
        """
        latent = latent_proprio[..., :latent_size]
        proprio = latent_proprio[..., latent_size:]

        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        decoder_input = jnp.concatenate([latent, normalized_proprio], axis=-1)

        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)

        action = parametric_action_distribution.mode(logits)

        return action, {"logits": logits}

    return decoder_fn


def make_decoder_logits_fn(
    decoder_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    config: Dict,
) -> Callable:
    """Create a decoder-logits inference function (raw pre-tanh, no mode).

    Returns the **raw** decoder logits (shape ``(..., 2*action_size)`` —
    concatenation of pre-tanh ``mu`` and ``log_std``). Used by the
    KL-anchor wrapper to expose `(mu_imit, log_std_imit)` to the policy
    loss for closed-form KL.

    Args:
        decoder_params: Frozen decoder parameters from the imit checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics.
        config: Checkpoint config dict (as returned by
            ``load_prior_checkpoint``).

    Returns:
        Function ``(latent_proprio) -> (logits, extras)`` where
        ``latent_proprio`` is ``[latent, proprio]`` concatenated and
        ``logits`` has shape ``(..., 2 * action_size)``.
    """
    action_size = config["network_config"]["action_size"]
    latent_size = config["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = tuple(config["network_config"]["decoder_layer_sizes"])

    # Use NormalTanhDistribution.param_size for the output layer to stay in
    # lockstep with make_decoder_inference_fn — both functions construct the
    # SAME decoder module shape; only the postprocessing differs (mode vs
    # raw logits). Hardcoding `2 * action_size` would silently desync if
    # brax ever ships a NormalTanh variant with a different param_size.
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    decoder_module = intention_network.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
    )

    proprio_normalizer = normalizer_params.proprioception

    def decoder_logits_fn(latent_proprio: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
        latent = latent_proprio[..., :latent_size]
        proprio = latent_proprio[..., latent_size:]
        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)
        decoder_input = jnp.concatenate([latent, normalized_proprio], axis=-1)
        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)
        return logits, {"logits": logits}

    return decoder_logits_fn


def make_prior_inference_fn(
    prior_params: Dict,
    normalizer_params: DictRunningStatisticsState,
    config: Dict,
) -> Callable:
    """Create prior inference function for use in high-level wrappers.

    The prior takes proprioceptive observations and outputs the mean and
    log-variance of the latent distribution.

    Args:
        prior_params: Frozen prior parameters from the mlp_prior checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics.
        config: Checkpoint config dict (as returned by ``load_prior_checkpoint``).

    Returns:
        Function ``(proprio) -> (mean, logvar)``.
    """
    latent_size = config["network_config"]["intention_size"]
    prior_hidden_layer_sizes = tuple(
        config["network_config"].get("prior_layer_sizes", [1024, 1024])
    )

    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    proprio_normalizer = normalizer_params.proprioception

    def prior_fn(proprio: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply prior to proprioceptive observations.

        Args:
            proprio: Proprioceptive observations.

        Returns:
            Tuple of ``(mean, logvar)`` for the latent distribution.
        """
        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        mean, logvar = prior_module.apply({"params": prior_params}, normalized_proprio)

        return mean, logvar

    return prior_fn
