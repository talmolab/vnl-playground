"""Smoke test for make_decoder_logits_fn — must return raw pre-tanh logits."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_make_decoder_logits_fn_returns_raw_logits():
    """The new fn returns logits of shape (..., 2*action_size) — concat of
    [mu_pretanh, log_std_pretanh] — without applying tanh.
    """
    from brax.training.acme import running_statistics, specs
    from scamper.agent.imitation.intention_network import Decoder as ScamperDecoder
    from scamper.agent.observation_utils import DictRunningStatisticsState

    from vnl_playground.tasks.prior_utils import make_decoder_logits_fn

    proprio_size = 8
    latent_size = 4
    action_size = 6
    decoder_hidden = (16,)
    rng = jax.random.PRNGKey(0)
    decoder = ScamperDecoder(layer_sizes=list(decoder_hidden) + [2 * action_size])
    decoder_params = decoder.init(
        rng, jnp.zeros((1, latent_size + proprio_size))
    )["params"]
    proprio_norm = running_statistics.RunningStatisticsState(
        mean=jnp.zeros((proprio_size,)),
        std=jnp.ones((proprio_size,)),
        count=jnp.array(1_000_000.0),
        summed_variance=jnp.zeros((proprio_size,)),
        std_eps=1e-6,
        mode=running_statistics.NormalizationMode.WELFORD,
    )
    target_norm = running_statistics.RunningStatisticsState(
        mean=jnp.zeros((1,)),
        std=jnp.ones((1,)),
        count=jnp.array(1_000_000.0),
        summed_variance=jnp.zeros((1,)),
        std_eps=1e-6,
        mode=running_statistics.NormalizationMode.WELFORD,
    )
    normalizer_params = DictRunningStatisticsState(
        imitation_target=target_norm, proprioception=proprio_norm
    )
    cfg = {
        "network_config": {
            "action_size": action_size,
            "intention_size": latent_size,
            "decoder_layer_sizes": list(decoder_hidden),
        }
    }

    fn = make_decoder_logits_fn(decoder_params, normalizer_params, cfg)
    latent_proprio = jnp.zeros((1, latent_size + proprio_size))
    logits, extras = fn(latent_proprio)

    # Shape check: (1, 2 * action_size).
    assert logits.shape[-1] == 2 * action_size, logits.shape
    # Must NOT have tanh applied — so the output range can exceed [-1, 1].
    # We can't easily prove "no tanh" with random params; but we can verify
    # logits == decoder.apply directly.
    expected, _ = decoder.apply({"params": decoder_params}, latent_proprio)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected), atol=1e-6)
