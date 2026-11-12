"""Eval-time statistics for the full maze (`MazeForageVision`, not the spout task).

The DMPO eval rollout returns per-step ``[T, N]`` arrays for every env
(``allenv``), but carries only ``rewards/*``, ``terminations/*`` and
``anchor/*`` env metrics through to it.  ``maze/visited_cells`` (an env metric
that exists only when ``maze_critic_priv`` is on) would therefore never reach
``allenv``.  `MazeEvalMetricAlias` wraps the EVAL env (never the training env)
of a ``maze_critic_priv`` env and mirrors it under
``MAZE_ALIAS + "visited_cells"`` -- an ``anchor/`` key, the one carried channel
that no HUD or termination code reads.  The mirrored keys should be stripped
from the generic eval metrics before logging.  `maze_eval_summary` reads
either spelling, so once the rollout carries ``maze/*`` keys itself the
wrapper can be dropped.

* `MazeEvalMetricAlias`: the eval-only mirroring wrapper;
* `maze_eval_summary`: per-env episode-1 treats, length, end causes and
  visited cells from the all-env arrays;
* `maze_eval_log_line`: the one-line ``MAZE_EVAL ...`` log record.
"""

from __future__ import annotations

import numpy as np
from mujoco_playground._src import wrapper as mp_wrapper

__all__ = [
    "FALL_TERMINATIONS",
    "MAZE_ALIAS",
    "MazeEvalMetricAlias",
    "maze_eval_log_line",
    "maze_eval_summary",
]

#: Prefix under which `MazeEvalMetricAlias` mirrors each ``maze/<name>`` metric.
MAZE_ALIAS = "anchor/_maze_eval/"

#: Termination flags that count as a fall, whichever of them the config carries.
#: ``contact_termination`` is the opt-in dm_control-style floor contact of a
#: non-foot geom (`MazeForageVision._contact_termination`).
FALL_TERMINATIONS = (
    "terminations/belly_up",
    "terminations/torso_too_low",
    "terminations/fallen",
    "terminations/contact_termination",
)


class MazeEvalMetricAlias(mp_wrapper.Wrapper):
    """Mirrors ``maze/<name>`` metrics to ``MAZE_ALIAS + <name>`` (eval only)."""

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
                k: v for k, v in state.metrics.items() if not k.startswith(MAZE_ALIAS)
            }
        )
        return self._alias(self.env.step(inner, action))

    def _alias(self, st):
        metrics = dict(st.metrics)
        for name in self._names:
            metrics[MAZE_ALIAS + name] = metrics[f"maze/{name}"]
        return st.replace(metrics=metrics)


def maze_eval_summary(allenv: dict, ctrl_dt: float) -> dict:
    """Full-maze eval statistics of each env's FIRST episode (``[T, N]`` in).

    "Episode 1" of an env is its steps up to and including its first ``done``
    (the whole window when it never ends). The env auto-resets inside the done
    step, so the ``rewards/*`` / ``terminations/*`` / ``maze/*`` values of that
    step still belong to the OLD episode.

    * ``treats_ep1`` / ``treats_ep1_sd``: mean / population sd over envs of the
      treats collected in episode 1 (steps with ``rewards/treat_collected`` > 0:
      weight-independent, and exact because two treats are never within reach
      of the rat on one step); NaN when the reward term is absent;
    * ``len_ep1_s``: mean episode-1 length in seconds, ``ctrl_dt`` per step;
    * ``fall_frac`` / ``alltreats_frac``: fraction of envs whose episode 1 ended
      on one of `FALL_TERMINATIONS` (whichever the config carries) / on
      ``all_treats_collected``; NaN when none of the needed ``terminations/*``
      keys is in ``allenv``;
    * ``timeout_frac``: fraction of envs whose episode 1 ran to the end of the
      window or ended on a done with no termination flag (``EpisodeWrapper``'s
      time-limit truncation); NaN without ``terminations/any``. A done on a
      NaN termination is in none of the three fractions;
    * ``visited_ep1``: mean over envs of ``maze/visited_cells`` (or its
      ``MAZE_ALIAS`` mirror) at the last step of episode 1; NaN when the metric
      is absent (``maze_critic_priv`` off).

    Args:
        allenv: The all-env eval arrays; ``done`` is required.
        ctrl_dt: Seconds per eval step (the env's control dt times the action
            repeat).

    Returns:
        The statistics above plus ``envs``, the number of envs.
    """
    done = np.asarray(allenv["done"]) > 0.5  # [T, N]
    t_len, n_envs = done.shape
    has_done = done.any(axis=0)
    last = np.where(has_done, done.argmax(axis=0), t_len - 1)  # [N]
    alive = np.arange(t_len)[:, None] <= last[None, :]  # [T, N]
    env_idx = np.arange(n_envs)
    nan = float("nan")

    def _ended_by(*keys):
        """Bool ``[N]``: done at the last step with any of ``keys`` flagged.

        None when none of ``keys`` is in ``allenv``.
        """
        present = [np.asarray(allenv[k]) > 0.5 for k in keys if k in allenv]
        if not present:
            return None
        return np.logical_or.reduce(present)[last, env_idx] & has_done

    def _frac(mask):
        return float(mask.mean()) if mask is not None else nan

    treat = allenv.get("rewards/treat_collected")
    if treat is not None:
        treats = ((np.asarray(treat) > 0) & alive).sum(axis=0).astype(np.float64)
        treats_mean, treats_sd = float(treats.mean()), float(treats.std())
    else:
        treats_mean = treats_sd = nan
    any_term = _ended_by("terminations/any")
    visited = next(
        (
            allenv[k]
            for k in ("maze/visited_cells", MAZE_ALIAS + "visited_cells")
            if k in allenv
        ),
        None,
    )
    return {
        "envs": n_envs,
        "treats_ep1": treats_mean,
        "treats_ep1_sd": treats_sd,
        "len_ep1_s": float((last + 1).mean() * ctrl_dt),
        "fall_frac": _frac(_ended_by(*FALL_TERMINATIONS)),
        "alltreats_frac": _frac(_ended_by("terminations/all_treats_collected")),
        "timeout_frac": _frac(~any_term if any_term is not None else None),
        "visited_ep1": (
            float(np.asarray(visited, np.float64)[last, env_idx].mean())
            if visited is not None
            else nan
        ),
    }


def maze_eval_log_line(env_steps: int, sm: dict) -> str:
    """The ``MAZE_EVAL ...`` log line; a NaN field prints as ``nan``.

    Log parsers count the exact prefix ``MAZE_EVAL env_steps=``: keep it.
    """
    return (
        "MAZE_EVAL env_steps=%d envs=%d treats_ep1=%.4g treats_ep1_sd=%.4g "  # noqa: UP031
        "len_ep1_s=%.4g fall_frac=%.4g alltreats_frac=%.4g timeout_frac=%.4g "
        "visited_ep1=%.4g"
    ) % (
        int(env_steps),
        sm["envs"],
        sm["treats_ep1"],
        sm["treats_ep1_sd"],
        sm["len_ep1_s"],
        sm["fall_frac"],
        sm["alltreats_frac"],
        sm["timeout_frac"],
        sm["visited_ep1"],
    )
