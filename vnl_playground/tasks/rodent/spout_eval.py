"""Eval-time statistics for the spout-maze task (`RodentSpoutMazeVision`).

The DMPO eval rollout returns per-step ``[T, N]`` arrays for every env
(``allenv``), but carries only ``rewards/*``, ``terminations/*`` and
``anchor/*`` env metrics through to it; the ``spout/*`` trial events of this
task would otherwise exist only for env 0, an n=1 estimator.  `SpoutEvalMetricAlias`
therefore wraps the EVAL env (never the training env) and mirrors every
``spout/<name>`` under ``SPOUT_ALIAS + <name>`` -- an ``anchor/`` key, the one
carried channel that no HUD or termination code reads.  The mirrored keys
should be stripped from the generic eval metrics before logging.  Once the
rollout carries ``spout/*`` keys itself, `spout_eval_summary` can read
``spout/<name>`` directly and the wrapper can be dropped.

* `SpoutEvalMetricAlias`: the eval-only mirroring wrapper;
* `spout_eval_summary`: hit rates, trial counts and fall fractions from the
  all-env arrays;
* `trial_index_stats`: hit rates by a trial's index within its episode segment;
* `spout_eval_log_line`: the one-line ``SPOUT_EVAL ...`` log record.
"""

from __future__ import annotations

import logging

import numpy as np
from mujoco_playground._src import wrapper as mp_wrapper

__all__ = [
    "SPOUT_ALIAS",
    "SpoutEvalMetricAlias",
    "spout_eval_log_line",
    "spout_eval_summary",
    "trial_index_stats",
]

log = logging.getLogger(__name__)

#: Prefix under which `SpoutEvalMetricAlias` mirrors each ``spout/<name>`` metric.
SPOUT_ALIAS = "anchor/_spout_eval/"


class SpoutEvalMetricAlias(mp_wrapper.Wrapper):
    """Mirrors ``spout/<name>`` metrics to ``SPOUT_ALIAS + <name>`` (eval only)."""

    def __init__(self, env, names):
        super().__init__(env)
        self._names = tuple(names)

    def reset(self, rng):
        return self._alias(self.env.reset(rng))

    def step(self, state, action):
        # Strip the mirrored keys before going down: brax's EpisodeWrapper
        # accumulates info["episode_metrics"][k] for EVERY key of the incoming
        # state.metrics, and that dict (built at reset, below this wrapper)
        # has no entry for them. The inner stack must see exactly what it
        # would without this shim.
        inner = state.replace(
            metrics={
                k: v for k, v in state.metrics.items() if not k.startswith(SPOUT_ALIAS)
            }
        )
        return self._alias(self.env.step(inner, action))

    def _alias(self, st):
        metrics = dict(st.metrics)
        for name in self._names:
            metrics[SPOUT_ALIAS + name] = metrics[f"spout/{name}"]
        return st.replace(metrics=metrics)


def spout_eval_summary(allenv: dict) -> dict:
    """Spout-maze eval statistics from the all-env ``[T, N]`` eval arrays.

    Sums over every env and the whole eval window. ``episodes`` counts episode
    SEGMENTS in the window: each env starts one at t=0 and one after every
    ``done`` except a done on the last step (whose episode has no steps here),
    so partial episodes count and ``*_per_ep`` is per segment.

    * ``hit_rate`` = Σhit / Σtrial_end; ``hit_rate_after_first`` =
      Σhit_after_first / Σtrial_after_first (both denominators floored at 1);
    * ``treats_per_ep`` = steps with ``rewards/treat_collected > 0`` per
      episode (weight-independent; NaN when the reward term is absent);
    * ``timeout_frac`` = Σtimeout / Σtrial_end (timeouts are trial ends);
    * ``fall_frac`` = episodes ending on a ``belly_up`` / ``torso_too_low``
      done, per episode;
    * the per-trial-index keys (``hit_t0..t2``, ``hit_t3p``, ``n_*``,
      ``hit_ge1_envmean``, ``n_ge1_segments``) are `trial_index_stats`: they
      do NOT pool trials across envs the way ``hit_rate_after_first`` does
      (see that function for what that removes and what it does not).
    """
    done = np.asarray(allenv["done"]) > 0.5  # [T, N]
    _, n_envs = done.shape

    def _sum(name):
        return float(np.asarray(allenv[SPOUT_ALIAS + name], np.float64).sum())

    episodes = n_envs + int(done[:-1].sum())
    trials = _sum("trial_end")
    treat = allenv.get("rewards/treat_collected")
    treats = float((np.asarray(treat) > 0).sum()) if treat is not None else float("nan")
    fell = np.zeros_like(done)
    for key in ("terminations/belly_up", "terminations/torso_too_low"):
        if key in allenv:
            fell |= np.asarray(allenv[key]) > 0.5
    falls = int((fell & done).sum())
    by_index = trial_index_stats(
        done,
        np.asarray(allenv[SPOUT_ALIAS + "trial_end"]),
        np.asarray(allenv[SPOUT_ALIAS + "hit"]),
    )
    # Cross-check of the done-based episode boundaries against the env's own
    # trial counter: `trial_after_first` fires on a trial end with
    # trial_idx >= 1, so it must equal the number of index >= 1 trials found
    # here. A mismatch means the boundary logic no longer matches the env.
    n_ge1 = by_index["n_t1"] + by_index["n_t2"] + by_index["n_t3p"]
    if n_ge1 != round(_sum("trial_after_first")):
        log.warning(
            "SPOUT_EVAL: %d trials with in-episode index >= 1 by done-based "
            "boundaries but sum(trial_after_first) = %g; the per-trial-index "
            "keys do not line up with the env's trial counter",
            n_ge1,
            _sum("trial_after_first"),
        )
    return {
        "episodes": episodes,
        "trials_per_ep": trials / episodes,
        "hit_rate": _sum("hit") / max(trials, 1.0),
        "hit_rate_after_first": (
            _sum("hit_after_first") / max(_sum("trial_after_first"), 1.0)
        ),
        "treats_per_ep": treats / episodes,
        "timeout_frac": _sum("timeout") / max(trials, 1.0),
        "fall_frac": falls / episodes,
        **by_index,
    }


def trial_index_stats(done, trial_end, hit) -> dict:
    """Hit rates by a trial's index within its episode segment (``[T, N]`` in).

    A completed trial (``trial_end`` > 0.5) has index ``k`` = the number of
    trials already completed in its episode segment, counted per env. A segment
    starts at t=0 and at ``t + 1`` after every ``done`` at ``t``: the ``done``
    step is the last step of the OLD episode (the env auto-resets inside that
    same step), so a trial ending on it belongs to the old segment.

    ``hit_rate_after_first`` pools every index >= 1 trial of every env, so an
    env that completes more trials in the window weighs more; when trial count
    correlates with hit rate (e.g. an env whose bait matches a memoryless
    policy's favoured arm finishes more trials) the pooled average is inflated.
    The keys below drop that weighting:

    * ``hit_t0`` / ``hit_t1`` / ``hit_t2`` / ``hit_t3p`` (index >= 3) = mean
      ``hit`` over the trials with that index, ``n_t*`` = their counts; an
      index with no trials is NaN. Index 0 is at most one trial per segment, so
      ``hit_t0`` is the unweighted reference rate of a policy with no memory
      of earlier trials;
    * ``hit_ge1_envmean`` = mean over segments with >= 1 index >= 1 trial of
      the fraction of that segment's index >= 1 trials that are hits (each
      segment weighs the same), ``n_ge1_segments`` = their number; NaN when
      there are none.

    Window censoring is NOT removed: index ``k`` only contains segments that
    finished ``k`` earlier trials before the window ended, so for a window
    shorter than the episode compare ``n_t*`` against ``n_t0`` before reading
    ``hit_t1`` and up as unselected (index 0 is the least exposed).
    """
    done = np.asarray(done) > 0.5
    te = np.asarray(trial_end) > 0.5
    hit = np.asarray(hit, np.float64)
    t_len, n_envs = done.shape
    # Trials completed in the whole window BEFORE step t, per env (exclusive).
    before = np.cumsum(te, axis=0) - te
    # `before` is non-decreasing in t, so the running max of its value at the
    # segment-start steps is the count before the CURRENT segment started.
    starts = np.zeros_like(done)
    starts[0] = True
    starts[1:] = done[:-1]
    base = np.maximum.accumulate(np.where(starts, before, 0), axis=0)
    index = (before - base)[te]
    hits = hit[te]
    # Unique (env, segment) id of every step; a segment id is the number of
    # dones strictly before the step, at most t_len - 1.
    segment = np.zeros((t_len, n_envs), np.int64)
    segment[1:] = np.cumsum(done[:-1], axis=0)
    seg_key = (segment + t_len * np.arange(n_envs, dtype=np.int64)[None, :])[te]

    out = {}
    for name, sel in (
        ("t0", index == 0),
        ("t1", index == 1),
        ("t2", index == 2),
        ("t3p", index >= 3),
    ):
        n = int(sel.sum())
        out[f"hit_{name}"] = float(hits[sel].mean()) if n else float("nan")
        out[f"n_{name}"] = n
    later = index >= 1
    if later.any():
        _, inv = np.unique(seg_key[later], return_inverse=True)
        n_trials = np.bincount(inv)
        frac = np.bincount(inv, weights=hits[later]) / n_trials
        out["hit_ge1_envmean"] = float(frac.mean())
        out["n_ge1_segments"] = int(n_trials.size)
    else:
        out["hit_ge1_envmean"] = float("nan")
        out["n_ge1_segments"] = 0
    return {
        k: out[k]
        for k in (
            "hit_t0",
            "hit_t1",
            "hit_t2",
            "hit_t3p",
            "hit_ge1_envmean",
            "n_t0",
            "n_t1",
            "n_t2",
            "n_t3p",
            "n_ge1_segments",
        )
    }


def spout_eval_log_line(env_steps: int, sp: dict) -> str:
    """The ``SPOUT_EVAL ...`` log line; a NaN field prints as ``nan``."""
    return (
        "SPOUT_EVAL env_steps=%d episodes=%d trials_per_ep=%.4g "  # noqa: UP031
        "hit_rate=%.4g hit_rate_after_first=%.4g "
        "treats_per_ep=%.4g timeout_frac=%.4g fall_frac=%.4g "
        "hit_t0=%.4g hit_t1=%.4g hit_t2=%.4g hit_t3p=%.4g "
        "hit_ge1_envmean=%.4g n_t0=%d n_t1=%d n_t2=%d n_t3p=%d "
        "n_ge1_segments=%d"
    ) % (
        int(env_steps),
        sp["episodes"],
        sp["trials_per_ep"],
        sp["hit_rate"],
        sp["hit_rate_after_first"],
        sp["treats_per_ep"],
        sp["timeout_frac"],
        sp["fall_frac"],
        sp["hit_t0"],
        sp["hit_t1"],
        sp["hit_t2"],
        sp["hit_t3p"],
        sp["hit_ge1_envmean"],
        sp["n_t0"],
        sp["n_t1"],
        sp["n_t2"],
        sp["n_t3p"],
        sp["n_ge1_segments"],
    )
