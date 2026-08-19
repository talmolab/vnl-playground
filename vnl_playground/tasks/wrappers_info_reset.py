"""Per-episode ``state.info`` reset for ``full_reset=False`` auto-reset stacks.

Why this exists. ``mujoco_playground``'s ``BraxAutoResetWrapper`` with
``full_reset=False`` (the mode both DMPO entry points use) swaps ``data`` and
``obs`` back to the cached first reset state when an episode ends, but leaves
``state.info`` untouched — its own docstring says "only data and obs are
reset, not the environment info". For RunGap this silently corrupts every
info-derived reward and termination signal during TRAINING (eval calls a real
``reset`` and is unaffected):

  * ``info["gaps_crossed"]`` keeps the env's all-time high-water mark, so
    after the first auto-reset ``gap_crossing_bonus`` only fires when the
    agent exceeds its personal-best corridor position EVER — the sparse
    reward becomes a cross-episode ratchet whose inflow decays to zero as
    records saturate. Measured consequence: every sparse arm's on-policy
    reward dried up, Q went flat, and the MPO temperature dual ran away
    (see arm_m1..m8, 2026-08-18/19).
  * ``info["max_x_reached"]`` persists the same way (silences new_progress).
  * ``info["stale_ref_x"]/["stale_steps"]`` persist (stale_location).
  * ``info["prev_action"]/["action"]`` leak the pre-reset action into the
    first post-reset observation.

This wrapper caches the listed keys at reset and restores them wherever
``done`` fires. It must wrap OUTSIDE ``wrap_for_brax_training`` so that it
sees the post-swap state (data already back at spawn, done flag preserved),
which keeps the restored info exactly consistent with the swapped data:
with ``full_reset=False`` the cached first state IS the layout every episode
of that env replays.

Note the in-episode high-water-mark semantics of ``gaps_crossed`` are
deliberate (no bonus re-collection by rocking back and forth across a gap
boundary) and are preserved — only the cross-episode persistence is removed.
"""
from __future__ import annotations

from typing import Sequence

import jax.numpy as jp
from mujoco_playground._src import wrapper


# The per-episode keys RunGap writes at reset (run_gap.py reset(): info = {...}).
DEFAULT_RUN_GAP_KEYS = (
    "prev_action",
    "action",
    "stale_ref_x",
    "stale_steps",
    "gaps_crossed",
    "just_crossed_gap",
    "max_x_reached",
)


class InfoResetOnDoneWrapper(wrapper.Wrapper):
    """Restores selected ``state.info`` keys to their reset-time values on done."""

    _CACHE_KEY = "InfoResetOnDone_first"

    def __init__(self, env, keys: Sequence[str] = DEFAULT_RUN_GAP_KEYS):
        super().__init__(env)
        if not keys:
            raise ValueError("InfoResetOnDoneWrapper needs at least one info key")
        self._keys = tuple(keys)

    def reset(self, rng, **kwargs):
        state = self.env.reset(rng, **kwargs)
        missing = [k for k in self._keys if k not in state.info]
        if missing:
            raise KeyError(
                f"InfoResetOnDoneWrapper: keys {missing} not present in "
                f"state.info at reset. Present keys: {sorted(state.info)}"
            )
        state.info[self._CACHE_KEY] = {k: state.info[k] for k in self._keys}
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        first = state.info[self._CACHE_KEY]
        done = state.done
        for k in self._keys:
            state.info[k] = _where_done(done, first[k], state.info[k], key=k)
        return state


def _where_done(done, first, current, *, key):
    """``where(done, first, current)`` with BraxAutoResetWrapper's broadcasting.

    Unlike BraxAutoResetWrapper's ``where_done`` (which silently skips
    shape-mismatched leaves), a mismatch here is a config error on an
    explicitly listed key, so it fails loudly at trace time.
    """
    if done.shape:
        if not current.shape or done.shape[0] != current.shape[0]:
            raise ValueError(
                f"InfoResetOnDoneWrapper: info[{key!r}] has shape "
                f"{current.shape} which does not lead with the batch dim of "
                f"done {done.shape}; refusing to silently skip a listed key."
            )
        done = jp.reshape(done, [current.shape[0]] + [1] * (current.ndim - 1))
    return jp.where(done, first, current)
