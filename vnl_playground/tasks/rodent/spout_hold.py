"""Inter-trial hold and start gate for the T-maze spout task (pure JAX).

Mirrors a mouse start-box hold (and the hold / delay epoch of RNN
working-memory tasks): after every trial the rat is put back in ``start``
behind a closed gate, waits a hold drawn per trial, and then the gate drops into
the floor (the "go") and the trial clock starts.

Everything here works on a *single* env (batch it with ``jax.vmap``) and is
written with ``jnp.where`` only, so it runs under ``jit`` / ``vmap`` and on the
CPU.  `RodentSpoutMazeVision` calls it every step when ``hold_enabled``:

1. ``frozen = holding(hs)`` (the state at the START of the step);
2. ``trial_update`` runs, and `freeze_trial` keeps the old trial state and
   zeroes every event while ``frozen``: the trial clock and its timeout count
   only from the release, and no trial event fires during the hold;
3. ``hold_update(hs, trial_end, rng, p)`` counts the hold down, animates the
   gate after the release, and on a trial end re-arms the hold (new duration,
   new boundary, gate closed);
4. ``gate_z(hs, p)`` gives the gate's height offset for this step's pose.

Timing (``H`` = the drawn hold): the steps that start holding are the ``H``
calls after a reset or a trial end, so the trial is frozen for exactly ``H``
steps (``hold_active`` = 1 on each).  ``gate_release`` fires on the last of
them, where ``hold_left`` reaches 0 and ``open_t`` becomes 1.  ``open_t`` then
counts the steps since the release up to ``gate_open_steps`` and saturates.

Gate height offset (`gate_z`): 0 (closed, standing on the floor) while holding;
``-gate_height * open_t / gate_open_steps`` while it slides down, so the drop
is linear over ``gate_open_steps`` steps; ``park_z`` (at or below
``-gate_height``: fully under the floor) from ``open_t == gate_open_steps`` on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "HOLD_EVENTS",
    "HoldParams",
    "HoldState",
    "freeze_trial",
    "gate_z",
    "hold_init",
    "hold_update",
    "holding",
    "make_hold_params",
]

#: Event names `hold_update` returns (float32 scalars).
HOLD_EVENTS = ("hold_active", "gate_release")


class HoldParams(NamedTuple):
    """Static hold / gate parameters (build them with `make_hold_params`).

    Attributes:
      hold_min_steps: shortest hold, in control steps (>= 1).
      hold_max_steps: longest hold, in control steps; the hold is uniform over
        the integers in ``[hold_min_steps, hold_max_steps]``.
      gate_boundaries: stem boundary indices the gate is drawn from, uniformly
        per trial.  Boundary ``k`` sits between stem segment ``k - 1`` and
        ``k`` (1 = ``start`` | ``seg1``).
      gate_open_steps: steps the gate takes to slide from closed to fully below
        the floor after the release.
      gate_height: height of the gate (m); the slide drops it by this much.
      park_z: gate height offset once open (m), ``<= -gate_height``.
    """

    hold_min_steps: int = 100
    hold_max_steps: int = 300
    gate_boundaries: tuple[int, ...] = (1,)
    gate_open_steps: int = 20
    gate_height: float = 0.3
    park_z: float = -1.3


class HoldState(NamedTuple):
    """Per-env hold state.  All fields are int32 scalars (batch with ``jax.vmap``).

    A `NamedTuple` is already a pytree; for ``state.info`` use
    ``hs._asdict()`` and ``HoldState(**d)``.

    Attributes:
      hold_left: control steps of hold left; the env is holding while > 0.
      gate_k: the stem boundary the gate stands on this trial.
      open_t: steps since the release (0 while holding), saturating at
        ``gate_open_steps``; drives the open animation.
    """

    hold_left: jnp.ndarray
    gate_k: jnp.ndarray
    open_t: jnp.ndarray


def make_hold_params(
    *,
    hold_min_s: float,
    hold_max_s: float,
    ctrl_dt: float,
    gate_boundaries: Sequence[int],
    gate_open_steps: int,
    gate_height: float,
    park_z: float,
    n_stem_segments: int,
) -> HoldParams:
    """`HoldParams` from the config's seconds, with validation.

    The hold bounds are ``round(hold_s / ctrl_dt)`` control steps.

    Raises:
      ValueError: if the shortest hold rounds to < 1 step, the longest is
        shorter than the shortest, ``gate_boundaries`` is empty or holds an
        index outside ``[1, n_stem_segments]``, ``gate_open_steps < 1``,
        ``gate_height <= 0`` or ``park_z > -gate_height``.
    """
    hold_min_steps = round(float(hold_min_s) / float(ctrl_dt))
    hold_max_steps = round(float(hold_max_s) / float(ctrl_dt))
    if hold_min_steps < 1:
        raise ValueError(
            f"hold_min_s={hold_min_s} is {hold_min_steps} control steps; the hold "
            "must last at least one step (use hold_enabled=False for no hold)."
        )
    if hold_max_steps < hold_min_steps:
        raise ValueError(
            f"hold_max_s={hold_max_s} ({hold_max_steps} steps) is shorter than "
            f"hold_min_s={hold_min_s} ({hold_min_steps} steps)."
        )
    boundaries = tuple(int(k) for k in gate_boundaries)
    n_stem_segments = int(n_stem_segments)
    if not boundaries or any(k < 1 or k > n_stem_segments for k in boundaries):
        raise ValueError(
            f"gate_boundaries={tuple(gate_boundaries)} must be a non-empty tuple of "
            f"stem boundary indices in [1, {n_stem_segments}] (boundary k sits "
            "between stem segments k-1 and k)."
        )
    if int(gate_open_steps) < 1:
        raise ValueError(f"gate_open_steps={gate_open_steps} must be >= 1.")
    if float(gate_height) <= 0.0:
        raise ValueError(f"gate_height={gate_height} must be positive.")
    if float(park_z) > -float(gate_height):
        raise ValueError(
            f"park_z={park_z} must be <= -gate_height ({-float(gate_height)}): the "
            "open gate has to be fully below the floor."
        )
    return HoldParams(
        hold_min_steps=hold_min_steps,
        hold_max_steps=hold_max_steps,
        gate_boundaries=boundaries,
        gate_open_steps=int(gate_open_steps),
        gate_height=float(gate_height),
        park_z=float(park_z),
    )


def _draw(rng: jax.Array, p: HoldParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A fresh ``(hold_left, gate_k)``: uniform hold steps and a uniform boundary."""
    k_hold, k_gate = jax.random.split(rng)
    hold = jax.random.randint(
        k_hold, (), p.hold_min_steps, p.hold_max_steps + 1, dtype=jnp.int32
    )
    idx = jax.random.randint(k_gate, (), 0, len(p.gate_boundaries), dtype=jnp.int32)
    gate_k = jnp.asarray(p.gate_boundaries, dtype=jnp.int32)[idx]
    return hold, gate_k


def hold_init(rng: jax.Array, p: HoldParams) -> HoldState:
    """The reset state: a fresh hold, a fresh boundary, the gate closed.

    Args:
      rng: PRNG key.
      p: hold parameters.

    Returns:
      The initial `HoldState`.
    """
    hold, gate_k = _draw(rng, p)
    return HoldState(hold_left=hold, gate_k=gate_k, open_t=jnp.int32(0))


def holding(hs: HoldState) -> jnp.ndarray:
    """bool: the env is in the hold (gate closed, trial clock frozen)."""
    return hs.hold_left > 0


def hold_update(
    hs: HoldState, trial_end: jnp.ndarray, rng: jax.Array, p: HoldParams
) -> tuple[HoldState, dict[str, jnp.ndarray]]:
    """Advances the hold by one env step.

    Counts ``hold_left`` down (to 0), advances ``open_t`` after the release and,
    on ``trial_end``, re-arms: a new hold and boundary are drawn and
    ``open_t`` returns to 0, so the gate is closed on the trial-end step itself.
    ``trial_end`` is expected only on unfrozen steps (`freeze_trial` zeroes it
    during the hold).

    Args:
      hs: current state.
      trial_end: bool, the trial ended on this step.
      rng: PRNG key for the re-arm draw (consumed every step, used on a trial
        end).
      p: hold parameters.

    Returns:
      ``(new_state, events)``; ``events`` holds float32 scalars
      ``hold_active`` (the step started in the hold, i.e. the trial was frozen
      on it) and ``gate_release`` (the hold ended on this step).
    """
    trial_end = jnp.asarray(trial_end, dtype=bool)
    was_holding = holding(hs)
    left = jnp.maximum(hs.hold_left - 1, 0)
    release = was_holding & (left == 0)
    open_t = jnp.where(left > 0, 0, jnp.minimum(hs.open_t + 1, p.gate_open_steps))

    new_hold, new_k = _draw(rng, p)
    new_hs = HoldState(
        hold_left=jnp.where(trial_end, new_hold, left).astype(jnp.int32),
        gate_k=jnp.where(trial_end, new_k, hs.gate_k).astype(jnp.int32),
        open_t=jnp.where(trial_end, 0, open_t).astype(jnp.int32),
    )
    events = {
        "hold_active": was_holding.astype(jnp.float32),
        "gate_release": release.astype(jnp.float32),
    }
    return new_hs, events


def gate_z(hs: HoldState, p: HoldParams) -> jnp.ndarray:
    """float32 gate height offset: 0 holding, a linear drop, then ``park_z``.

    Args:
      hs: hold state.
      p: hold parameters.

    Returns:
      The z offset of the gate body from its closed pose (standing on the
      floor).
    """
    ramp = -p.gate_height * hs.open_t.astype(jnp.float32) / p.gate_open_steps
    opened = jnp.where(hs.open_t >= p.gate_open_steps, p.park_z, ramp)
    return jnp.where(holding(hs), 0.0, opened).astype(jnp.float32)


def freeze_trial(frozen, old_ts, new_ts, events):
    """Keeps ``old_ts`` and zeroes every event while ``frozen``; passes through otherwise.

    Args:
      frozen: bool, the step started in the hold (``holding(hs)`` before
        `hold_update`).
      old_ts: the trial state before ``trial_update`` (any pytree).
      new_ts: the trial state ``trial_update`` returned (same structure).
      events: the event dict ``trial_update`` returned.

    Returns:
      ``(trial_state, events)``.
    """
    frozen = jnp.asarray(frozen, dtype=bool)
    ts = jax.tree.map(
        lambda o, n: jnp.where(frozen, o, n).astype(o.dtype), old_ts, new_ts
    )
    events = {k: jnp.where(frozen, jnp.zeros_like(v), v) for k, v in events.items()}
    return ts, events
