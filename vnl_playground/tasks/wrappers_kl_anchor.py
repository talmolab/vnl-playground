"""Wrapper for DMPO kl-anchor mode (KL-in-loss surface).

Computes the frozen anchor pipeline (frozen Prior -> frozen Decoder) at
every reset/step using *only* proprioception. The decoder now returns
raw pre-tanh logits, which are split into the Gaussian distribution
params (mu_imit, log_std_imit). These are written to ``state.info`` so
that the policy loss can read them at SGD time and compute a closed-form
KL term (see DMPOConfig.kl_anchor_alpha). The env reward is set to
``r_task`` only — the anchor signal flows through the *policy loss*, not
through the critic via env reward.

The diagnostic ``anchor/r_anchor`` (MSE-based, post-tanh) and
``anchor/action_mse`` are still emitted to ``state.metrics`` for
monitoring. The constructor keeps ``w_anchor`` and ``alpha_anchor`` for
this diagnostic only — the loss-side weight lives in DMPOConfig.

Also flattens the env's nested observation dict into the canonical
{"vision", "imitation_target", "proprioception"} shape that the kl-anchor
policy network and the eval rollout helper expect. This mirrors what
HighLevelWrapper does in the PPO pipeline. The vision field is preserved
as a placeholder (zeros from the registry env) and is replaced with real
binocular renders by BinocularVisionRenderWrapper downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
from jax import flatten_util as _flatten_util
from mujoco_playground._src import mjx_env, wrapper

if TYPE_CHECKING:
    from collections.abc import Callable


def _flatten_nested_obs(nested):
    """Flatten a possibly nested observation, preserving leading batch dims."""
    if isinstance(nested, jp.ndarray):
        return nested
    leaves = jax.tree.leaves(nested)
    if not leaves:
        return jp.array([])
    ref = min(leaves, key=lambda x: x.ndim)
    n_batch = 0
    for i in range(ref.ndim - 1):
        if len({leaf.shape[i] for leaf in leaves}) == 1:
            n_batch += 1
        else:
            break
    if n_batch == 0:
        flat, _ = _flatten_util.ravel_pytree(nested)
        return flat
    batch_shape = ref.shape[:n_batch]
    return jp.concatenate([leaf.reshape(*batch_shape, -1) for leaf in leaves], axis=-1)


def flatten_obs_dict(obs):
    """Flatten each top-level key (proprioception / imitation_target|task_obs / vision).

    Deliberate copy of track-mjx's ``observation_utils.flatten_obs_dict``, kept in
    sync here to avoid a cross-repo import (track_mjx/agent/observation_utils.py).
    """
    # "vision" may live at the top level alongside "state" (vnl_playground's
    # layout), so capture it before unwrapping in case "state" doesn't carry
    # its own copy.
    vision = obs.get("vision")

    if "state" in obs and "proprioception" not in obs:
        obs = obs["state"]
        if vision is None:
            vision = obs.get("vision")

    flat_proprio = _flatten_nested_obs(obs["proprioception"])
    if "imitation_target" in obs:
        flat_imit = _flatten_nested_obs(obs["imitation_target"])
    elif "task_obs" in obs:
        flat_imit = _flatten_nested_obs(obs["task_obs"])
    else:
        flat_imit = jp.zeros((*flat_proprio.shape[:-1], 0))
    result = {"imitation_target": flat_imit, "proprioception": flat_proprio}
    if vision is not None:
        result["vision"] = vision
    return result


class KLAnchorPriorDecoderWrapper(wrapper.Wrapper):
    """Exposes the frozen anchor's pre-tanh Gaussian distribution params via
    state.info for downstream KL-in-loss; emits MSE-based diagnostics on
    state.metrics. Does NOT modify state.reward — the env reward stays r_task.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_fn: Callable,
        decoder_logits_fn: Callable,
        action_size: int,
        w_anchor: float = 0.5,
        alpha_anchor: float = 0.0,  # diagnostic-only after KL-in-loss port
        anchor_obs_key: str = "state",
        proprio_obs_key: str = "proprioception",
    ):
        super().__init__(env)
        self._prior_fn = prior_fn
        self._decoder_logits_fn = decoder_logits_fn
        self._action_size = action_size
        self._w_anchor = float(w_anchor)
        self._alpha_anchor = float(alpha_anchor)  # kept for diagnostics only
        self._anchor_obs_key = anchor_obs_key
        self._proprio_obs_key = proprio_obs_key

    def _flatten_proprio(self, full_obs):
        proprio = None
        if isinstance(full_obs, dict):
            if self._anchor_obs_key in full_obs and isinstance(
                full_obs[self._anchor_obs_key], dict
            ):
                proprio = full_obs[self._anchor_obs_key].get(
                    self._proprio_obs_key, None
                )
            if proprio is None and self._proprio_obs_key in full_obs:
                proprio = full_obs[self._proprio_obs_key]
        if proprio is None:
            raise KeyError(
                f"Cannot find proprio key {self._proprio_obs_key} under "
                f"{self._anchor_obs_key} in obs"
            )
        # Proprio may be a (possibly nested) dict of arrays; ravel the
        # whole pytree to a flat 1-D vector matching the decoder's input.
        if isinstance(proprio, dict):
            flat, _ = _flatten_util.ravel_pytree(proprio)
            return flat
        return proprio

    def _compute_anchor(self, proprio):
        prior_mean, _ = self._prior_fn(proprio)
        latent_proprio = jp.concatenate([prior_mean, proprio], axis=-1)
        logits, _ = self._decoder_logits_fn(latent_proprio)
        mu = logits[..., : self._action_size]
        # The decoder emits raw pre-softplus log_std (NormalTanhDistribution
        # convention: scale = softplus(raw) + min_std). The loss-side and
        # the startup probe pull `log_std_theta = log(online_dist.stddev())`
        # which equals `log(softplus(raw) + 1e-3)`. To put both sides in the
        # SAME units (so KL is honest at warm-start), apply the same transform
        # here before exposing log_std_imit to state.info.
        raw_log_std = logits[..., self._action_size :]
        log_std = jp.log(jax.nn.softplus(raw_log_std) + 1e-3)
        return mu, log_std, prior_mean

    def _flatten_for_policy(self, obs):
        """Convert nested env obs to {vision, imitation_target, proprioception}.

        Mirrors what HighLevelWrapper does so the kl-anchor policy and eval
        rollout receive obs in the same flat shape they expect. Preserves
        the `vision` placeholder so BinocularVisionRenderWrapper downstream
        can replace it with the real render.
        """
        return flatten_obs_dict(obs)

    def reset(self, rng, **kwargs):
        state = self.env.reset(rng, **kwargs)
        full_obs = state.info.get("_full_obs", state.obs)
        proprio = self._flatten_proprio(full_obs)
        mu_imit, log_std_imit, prior_mean = self._compute_anchor(proprio)
        a_imit = jp.tanh(mu_imit[..., : self._action_size])
        state.info["anchor_mu_imit"] = mu_imit[..., : self._action_size]
        state.info["anchor_log_std_imit"] = log_std_imit[..., : self._action_size]
        state.info["anchor_a_imit"] = a_imit
        m = dict(state.metrics) if state.metrics else {}
        m["anchor/r_anchor"] = jp.float32(1.0)
        m["anchor/action_mse"] = jp.float32(0.0)
        m["anchor/r_task"] = jp.float32(0.0)
        # Emit diagnostic alpha so wandb sees the YAML knob even though the
        # env reward is no longer scaled by it.
        m["anchor/alpha_anchor_diag"] = jp.float32(self._alpha_anchor)
        return state.replace(obs=self._flatten_for_policy(state.obs), metrics=m)

    def step(self, state, action):
        next_state = self.env.step(state, action)
        full_obs = next_state.info.get("_full_obs", next_state.obs)
        proprio = self._flatten_proprio(full_obs)
        mu_imit, log_std_imit, prior_mean = self._compute_anchor(proprio)
        a_imit = jp.tanh(mu_imit[..., : self._action_size])

        # Diagnostic anchor/r_anchor: MSE between sampled action and the imit
        # mode (post-tanh). KEPT for monitoring only; NOT added to reward.
        diff = action[..., : self._action_size] - a_imit
        action_mse = jp.mean(diff * diff)
        r_anchor = jp.exp(-self._w_anchor * action_mse * self._action_size)
        r_task = next_state.reward
        # r_total = r_task ONLY. The anchor signal flows through the policy
        # loss in the learner (NOT through the env reward + critic).
        r_total = r_task

        next_state.info["anchor_mu_imit"] = mu_imit[..., : self._action_size]
        next_state.info["anchor_log_std_imit"] = log_std_imit[..., : self._action_size]
        next_state.info["anchor_a_imit"] = a_imit

        m = dict(next_state.metrics) if next_state.metrics else {}
        m["anchor/r_anchor"] = r_anchor.astype(jp.float32)
        m["anchor/action_mse"] = action_mse.astype(jp.float32)
        m["anchor/r_task"] = r_task.astype(jp.float32)
        m["anchor/alpha_anchor_diag"] = jp.float32(self._alpha_anchor)

        return next_state.replace(
            obs=self._flatten_for_policy(next_state.obs),
            reward=r_total,
            metrics=m,
        )

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def observation_size(self):
        """Return the observation-size dict expected by the kl-anchor entry.

        The kl-anchor policy consumes ``vision`` (HxWxC), ``imitation_target``,
        and ``proprioception``. We mirror the structure produced by
        ``flatten_obs_dict``: flattened element counts for proprio, task_obs,
        and vision.

        Every vnl_playground leaf env builds ``non_flattened_observation_size``
        as ``jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)``
        (see e.g. ``RunGap.non_flattened_observation_size``), so by the time we
        read it here every leaf -- including vision -- has ALREADY been
        collapsed to a scalar element count. There is no per-axis shape left to
        recover from it (a 0-d scalar's own ``.shape`` is always ``()``), so
        ``vision`` is returned as its flattened pixel count (H*W*C), exactly
        like ``proprioception`` and ``imitation_target`` -- not as a shape
        tuple. A caller that needs the real (H, W, C) axes must get them from
        the env's own vision config, not from this size dict.
        """
        obs = self.env.non_flattened_observation_size

        # "vision" may live at the top level alongside "state" (vnl_playground's
        # layout), so capture it before unwrapping in case "state" doesn't
        # carry its own copy. Mirrors flatten_obs_dict's capture-before /
        # second-chance-after-unwrap logic exactly -- same guard, because
        # non_flattened_observation_size has the same nesting as the real obs
        # pytree it was derived from (just with sizes at the leaves).
        vision = obs.get("vision")
        inner = obs
        if "state" in inner and "proprioception" not in inner:
            inner = inner["state"]
            if vision is None:
                vision = inner.get("vision")
        proprio = inner.get("proprioception", 0)
        task_obs = inner.get("task_obs", inner.get("imitation_target", 0))

        def _size(x):
            if isinstance(x, dict):
                flat, _ = _flatten_util.ravel_pytree(x)
                return int(jp.sum(flat))
            try:
                return int(jp.sum(jp.array(x)))
            except Exception:
                return int(x)

        result = {
            "proprioception": _size(proprio),
            "imitation_target": _size(task_obs),
        }
        if vision is not None:
            result["vision"] = _size(vision)
        return result
