"""KLAnchorPriorDecoderWrapper unit test — verifies the new KL-in-loss surface.

The wrapper must:
  - Store anchor_mu_imit and anchor_log_std_imit (pre-tanh) in state.info.
  - Set state.reward = r_task only (NOT r_task + alpha*r_anchor).
  - Keep diagnostic anchor/r_anchor and anchor/action_mse in state.metrics.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vnl_playground.tasks.wrappers_kl_anchor import KLAnchorPriorDecoderWrapper


class _StubState:
    """Minimal state pytree mirroring brax/playground state surface."""

    def __init__(self, obs, reward, metrics=None, info=None):
        self.obs = obs
        self.reward = reward
        self.metrics = metrics if metrics is not None else {}
        self.info = info if info is not None else {}
        self.done = jnp.asarray(0.0, dtype=jnp.float32)

    def replace(self, **kwargs):
        new = _StubState(self.obs, self.reward, self.metrics, self.info)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new


class _StubEnv:
    """Stub base env that returns a fixed proprio + r_task = 0.7."""

    action_size = 4
    proprio_size = 6
    task_obs_size = 3

    @property
    def non_flattened_observation_size(self):
        return {
            "state": {
                "proprioception": self.proprio_size,
                "task_obs": self.task_obs_size,
            },
            "vision": (8, 8, 2),
        }

    def reset(self, rng, **kwargs):
        obs = {
            "state": OrderedDict([
                ("proprioception", jnp.ones((self.proprio_size,))),
                ("task_obs", jnp.zeros((self.task_obs_size,))),
            ]),
            "vision": jnp.zeros((8, 8, 2)),
        }
        return _StubState(obs=obs, reward=jnp.float32(0.0))

    def step(self, state, action):
        # r_task = 0.7 always (so we can assert it ends up unchanged in
        # the wrapper's reward pass-through).
        return _StubState(
            obs=state.obs,
            reward=jnp.float32(0.7),
            metrics={},
            info={},
        )

    @property
    def action_size(self):
        return self.__class__.action_size


def test_wrapper_stores_pretanh_distribution_params_and_passes_through_r_task():
    action_size = 4
    latent_size = 3

    def fake_prior_fn(proprio):
        # Returns deterministic prior_mean of dim=latent_size.
        return jnp.full((latent_size,), 0.123, dtype=jnp.float32), jnp.zeros((latent_size,))

    def fake_decoder_logits_fn(latent_proprio):
        # Returns logits of shape (..., 2*action_size). Pretend mu = 0.5,
        # log_std = -1.0 across all dims.
        batch_shape = latent_proprio.shape[:-1]
        mu = jnp.full(batch_shape + (action_size,), 0.5)
        log_std = jnp.full(batch_shape + (action_size,), -1.0)
        return jnp.concatenate([mu, log_std], axis=-1), {}

    env = _StubEnv()
    wrapped = KLAnchorPriorDecoderWrapper(
        env=env,
        prior_fn=fake_prior_fn,
        decoder_logits_fn=fake_decoder_logits_fn,
        action_size=action_size,
        w_anchor=0.5,
        alpha_anchor=0.0,  # alpha is now ONLY used for diagnostic; loss-side is in cfg
    )

    state0 = wrapped.reset(jax.random.PRNGKey(0))
    # mu_imit and log_std_imit must be in state.info.
    assert "anchor_mu_imit" in state0.info
    assert "anchor_log_std_imit" in state0.info
    np.testing.assert_allclose(np.asarray(state0.info["anchor_mu_imit"]),
                               np.full((action_size,), 0.5), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state0.info["anchor_log_std_imit"]),
                               np.full((action_size,), -1.0), atol=1e-6)
    # The diagnostic anchor/r_anchor should also be in metrics.
    assert "anchor/r_anchor" in state0.metrics

    # Step and confirm reward is r_task only (0.7), NOT r_task + alpha*r_anchor.
    # Action matches a_imit = tanh(mu_imit) = tanh(0.5) so MSE = 0.
    action = jnp.tanh(jnp.full((action_size,), 0.5))
    state1 = wrapped.step(state0, action)
    # reward must equal env's r_task (0.7), not augmented.
    np.testing.assert_allclose(float(state1.reward), 0.7, atol=1e-6)
    # anchor/r_task in metrics (still 0.7).
    np.testing.assert_allclose(
        float(state1.metrics["anchor/r_task"]), 0.7, atol=1e-6
    )
    # anchor/r_anchor in metrics — should be 1.0 (action == mu_imit, MSE=0).
    np.testing.assert_allclose(
        float(state1.metrics["anchor/r_anchor"]), 1.0, atol=1e-5
    )


def test_wrapper_anchor_distribution_params_present_after_step():
    """After step(), state.info still has anchor_mu_imit and anchor_log_std_imit."""
    action_size = 4
    latent_size = 3

    def fake_prior_fn(proprio):
        return jnp.zeros((latent_size,)), jnp.zeros((latent_size,))

    def fake_decoder_logits_fn(latent_proprio):
        bs = latent_proprio.shape[:-1]
        mu = jnp.full(bs + (action_size,), 0.0)
        log_std = jnp.full(bs + (action_size,), 0.0)
        return jnp.concatenate([mu, log_std], axis=-1), {}

    env = _StubEnv()
    wrapped = KLAnchorPriorDecoderWrapper(
        env=env, prior_fn=fake_prior_fn, decoder_logits_fn=fake_decoder_logits_fn,
        action_size=action_size, w_anchor=0.5, alpha_anchor=0.0,
    )
    state0 = wrapped.reset(jax.random.PRNGKey(0))
    state1 = wrapped.step(state0, jnp.zeros((action_size,)))
    assert "anchor_mu_imit" in state1.info
    assert "anchor_log_std_imit" in state1.info
