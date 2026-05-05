"""Wrapper for DMPO kl-anchor mode.

Computes the frozen anchor pipeline (frozen Prior -> frozen Decoder) at
every reset/step using *only* proprioception. After the base env step
returns, computes:
    a_imit = anchor_decoder_fn(prior_mean(proprio), proprio)  # mode action
    r_anchor = exp(-w_anchor * ||a_taken - a_imit||^2 / action_size)
    r_total = r_task + alpha_anchor * r_anchor
and replaces state.reward with r_total. Also stores a_imit + the per-step
anchor MSE in state.info / state.metrics for diagnostic logging.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jp
from mujoco_playground._src import wrapper, mjx_env


class KLAnchorPriorDecoderWrapper(wrapper.Wrapper):
    """Adds an action-anchor reward bonus on top of a base env."""

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_fn: Callable,
        decoder_fn: Callable,
        action_size: int,
        w_anchor: float = 0.01,
        alpha_anchor: float = 10.0,
        anchor_obs_key: str = "state",
        proprio_obs_key: str = "proprioception",
    ):
        super().__init__(env)
        self._prior_fn = prior_fn
        self._decoder_fn = decoder_fn
        self._action_size = action_size
        self._w_anchor = float(w_anchor)
        self._alpha_anchor = float(alpha_anchor)
        self._anchor_obs_key = anchor_obs_key
        self._proprio_obs_key = proprio_obs_key

    def _flatten_proprio(self, full_obs):
        if isinstance(full_obs, dict):
            if self._anchor_obs_key in full_obs and isinstance(
                full_obs[self._anchor_obs_key], dict
            ):
                proprio = full_obs[self._anchor_obs_key].get(
                    self._proprio_obs_key, None
                )
                if proprio is not None:
                    if isinstance(proprio, dict):
                        return jp.concatenate(
                            [proprio[k] for k in sorted(proprio.keys())], axis=-1
                        )
                    return proprio
            if self._proprio_obs_key in full_obs:
                return full_obs[self._proprio_obs_key]
        raise KeyError(
            f"Cannot find proprio key {self._proprio_obs_key} under "
            f"{self._anchor_obs_key} in obs"
        )

    def _compute_anchor(self, proprio):
        prior_mean, _ = self._prior_fn(proprio)
        latent_proprio = jp.concatenate([prior_mean, proprio], axis=-1)
        a_imit, _ = self._decoder_fn(latent_proprio)
        return a_imit, prior_mean

    def reset(self, rng, **kwargs):
        state = self.env.reset(rng, **kwargs)
        full_obs = state.info.get("_full_obs", state.obs)
        proprio = self._flatten_proprio(full_obs)
        a_imit, prior_mean = self._compute_anchor(proprio)
        state.info["anchor_a_imit"] = a_imit
        state.info["anchor_prior_mean"] = prior_mean
        m = dict(state.metrics) if state.metrics else {}
        m["anchor/r_anchor"] = jp.float32(1.0)
        m["anchor/action_mse"] = jp.float32(0.0)
        m["anchor/r_task"] = jp.float32(0.0)
        return state.replace(metrics=m)

    def step(self, state, action):
        next_state = self.env.step(state, action)
        full_obs = next_state.info.get("_full_obs", next_state.obs)
        proprio = self._flatten_proprio(full_obs)
        a_imit, prior_mean = self._compute_anchor(proprio)

        diff = action[..., : self._action_size] - a_imit[..., : self._action_size]
        action_mse = jp.mean(diff * diff)
        r_anchor = jp.exp(-self._w_anchor * action_mse * self._action_size)
        r_task = next_state.reward
        r_total = r_task + self._alpha_anchor * r_anchor

        next_state.info["anchor_a_imit"] = a_imit
        next_state.info["anchor_prior_mean"] = prior_mean

        m = dict(next_state.metrics) if next_state.metrics else {}
        m["anchor/r_anchor"] = r_anchor.astype(jp.float32)
        m["anchor/action_mse"] = action_mse.astype(jp.float32)
        m["anchor/r_task"] = r_task.astype(jp.float32)

        return next_state.replace(reward=r_total, metrics=m)

    @property
    def action_size(self) -> int:
        return self.env.action_size if hasattr(self.env, "action_size") else self._action_size
