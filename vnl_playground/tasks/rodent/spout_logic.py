"""Trial state machine for the vision-guided Y-maze "spout" task (pure JAX).

One *episode* holds many *trials*.  A hidden ``bait`` arm is drawn once per
episode and stays fixed; the rat has to find out (by trial and error across
trials) which arm pays.  Within a trial the rat commits to the first arm whose
end zone it enters; from that moment nothing it does can reveal or pay the other
arm, so a memoryless policy is held to exactly ``1 / n_arms`` hits.

``trial_update`` is called once per env step from jitted, batched (vmapped)
code with a per-env rng key.  It is written with ``jnp.where`` only: there is no
Python control flow on traced values.  Everything here works on a *single* env;
batch it with ``jax.vmap``.

The geometry inputs come from `spout_geometry.locate`: ``arm_id`` in
``{-1, 0, 1}`` (``-1`` in the stem and the junction triangle, where
``dist_to_end`` is finite but meaningless) and ``dist_to_end`` (negative past the
arm end, in the end alcove).  Every rule that reads ``dist_to_end`` is gated on
``arm_id >= 0``.

Rules, applied in this order every step (later rules see the earlier rules'
updates from the same step):

1. While no choice is locked, ``arm_id >= 0`` and
   ``dist_to_end <= appear_distance``: lock ``choice = arm_id``.  If the locked
   arm is the bait and ``uniform(rng) < pay_prob`` the treat appears.
2. A shown, not-yet-eaten treat within ``reach`` of the torso is eaten:
   ``reward = 1`` and the dwell starts.
3. A locked choice whose treat did not appear starts the dwell (no reward) once
   the rat reaches an arm end (``dist_to_end < spout_radius``) of *either* arm.
4. An active dwell counts down; the trial ends when it hits zero, or when
   ``trial_steps >= trial_timeout_steps`` (a timeout).  The end step reports the
   outcome events and resets every observable-relevant field, so the next trial
   starts independently of the previous one.  Only ``bait``, ``trial_idx`` and
   ``hits`` persist.
5. Otherwise ``trial_steps`` is incremented.

Timing consequences of applying the rules in order: the dwell counts down on the
step that starts it, so a dwell of ``D`` steps spans exactly ``D`` calls
(counting the call that started it) and ``trial_end`` fires on the ``D``-th one.
``trial_steps`` is read before it is incremented, so a trial times out on the
``(trial_timeout_steps + 1)``-th call.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "SpoutParams",
    "TrialState",
    "critic_privileged_vector",
    "init_trial_state",
    "trial_update",
]


class TrialState(NamedTuple):
    """Per-env trial state.  All fields are scalars (batch with ``jax.vmap``).

    A `NamedTuple` is already a pytree; for ``state.info`` use
    ``ts._asdict()`` and ``TrialState(**d)``.

    Attributes:
      bait: int32, the arm that pays (fixed for the episode).
      choice: int32, the arm locked this trial, ``-1`` while none is.
      treat_shown: bool, the treat has appeared on the chosen arm.
      eaten: bool, the treat has been eaten this trial.
      dwell: int32, steps left in the end-of-trial dwell, ``-1`` when inactive.
      trial_steps: int32, steps taken in the current trial.
      trial_idx: int32, number of completed trials this episode.
      hits: int32, completed trials whose choice was the bait arm.
    """

    bait: jnp.ndarray
    choice: jnp.ndarray
    treat_shown: jnp.ndarray
    eaten: jnp.ndarray
    dwell: jnp.ndarray
    trial_steps: jnp.ndarray
    trial_idx: jnp.ndarray
    hits: jnp.ndarray


class SpoutParams(NamedTuple):
    """Static task parameters (the defaults are the standard task values).

    Attributes:
      n_arms: number of arms the bait is drawn from.
      pay_prob: probability that the treat appears when the bait arm is chosen.
      appear_distance: the choice locks within this distance of an arm end (m).
      spout_radius: reaching an arm end within this distance (m) is a visit.
      reach: the treat is eaten when the torso is within this distance (m).
      dwell_steps: steps the rat is held at the spout before the trial ends.
      trial_timeout_steps: a trial is cut off after this many steps.
    """

    n_arms: int = 2
    pay_prob: float = 1.0
    appear_distance: float = 0.5
    spout_radius: float = 0.15
    reach: float = 0.1
    dwell_steps: int = 30
    trial_timeout_steps: int = 800


def init_trial_state(rng: jax.Array, p: SpoutParams) -> TrialState:
    """A fresh episode: no choice, trial 0, bait drawn uniformly from the arms.

    Args:
      rng: PRNG key.
      p: task parameters (only ``n_arms`` is used).

    Returns:
      The initial `TrialState`.
    """
    bait = jax.random.randint(rng, (), 0, p.n_arms, dtype=jnp.int32)
    return TrialState(
        bait=bait.astype(jnp.int32),
        choice=jnp.int32(-1),
        treat_shown=jnp.bool_(False),
        eaten=jnp.bool_(False),
        dwell=jnp.int32(-1),
        trial_steps=jnp.int32(0),
        trial_idx=jnp.int32(0),
        hits=jnp.int32(0),
    )


def trial_update(
    ts: TrialState,
    arm_id: jnp.ndarray,
    dist_to_end: jnp.ndarray,
    treat_dist: jnp.ndarray,
    rng: jax.Array,
    p: SpoutParams,
) -> tuple[TrialState, dict[str, jnp.ndarray]]:
    """Advances the trial state machine by one env step.

    Args:
      ts: current state.
      arm_id: int in ``{-1, 0, 1}`` from `spout_geometry.locate`.
      dist_to_end: distance to the end of the rat's arm (negative past the end),
        from `spout_geometry.locate`.  Meaningless when ``arm_id == -1``.
      treat_dist: torso-to-shown-treat xy distance.  Only read while the treat
        is shown.
      rng: PRNG key for this step (consumed by the ``pay_prob`` draw).
      p: task parameters.

    Returns:
      ``(new_state, events)``.  ``events`` holds float32 scalars: ``reward``
      (treat eaten this step), ``show_treat`` (treat appeared this step),
      ``trial_end``, ``timeout`` (trial ended by the step limit),
      ``choice_none`` (trial ended with no arm chosen) and, on the end step
      only, ``hit`` (choice == bait), ``trial_after_first`` (trial_idx >= 1) and
      ``hit_after_first``.
    """
    arm_id = jnp.asarray(arm_id, jnp.int32)
    dist_to_end = jnp.asarray(dist_to_end, jnp.float32)
    treat_dist = jnp.asarray(treat_dist, jnp.float32)
    in_arm = arm_id >= 0

    # 1. Lock the choice on first entering an arm's appear zone.
    lock = (ts.choice == -1) & in_arm & (dist_to_end <= p.appear_distance)
    choice = jnp.where(lock, arm_id, ts.choice)
    pay = jax.random.uniform(rng, ()) < p.pay_prob
    show = lock & (arm_id == ts.bait) & pay
    treat_shown = ts.treat_shown | show

    # 2. Eat a shown treat within reach: reward once, start the dwell.
    eat = treat_shown & ~ts.eaten & (treat_dist < p.reach)
    eaten = ts.eaten | eat
    dwell = jnp.where(eat, p.dwell_steps, ts.dwell)

    # 3. A locked choice with no treat shown: reaching any arm end starts the dwell.
    visit = (
        (dwell < 0)
        & (choice >= 0)
        & ~treat_shown
        & in_arm
        & (dist_to_end < p.spout_radius)
    )
    dwell = jnp.where(visit, p.dwell_steps, dwell)

    # 4. Count the dwell down; end the trial when it hits zero or on timeout.
    dwell = jnp.where(dwell > 0, dwell - 1, dwell)
    dwell_done = dwell == 0
    timeout = ts.trial_steps >= p.trial_timeout_steps
    end = dwell_done | timeout

    hit = end & (choice == ts.bait)
    after_first = end & (ts.trial_idx >= 1)
    events = {
        "reward": eat.astype(jnp.float32),
        "show_treat": show.astype(jnp.float32),
        "trial_end": end.astype(jnp.float32),
        "hit": hit.astype(jnp.float32),
        "trial_after_first": after_first.astype(jnp.float32),
        "hit_after_first": (hit & after_first).astype(jnp.float32),
        "timeout": (end & timeout).astype(jnp.float32),
        "choice_none": (end & (choice == -1)).astype(jnp.float32),
    }

    # On the end step reset every observable-relevant field; bait, trial_idx
    # (+1) and hits (+hit) persist.  5. Otherwise the step counter ticks on.
    new_ts = TrialState(
        bait=ts.bait,
        choice=jnp.where(end, -1, choice).astype(jnp.int32),
        treat_shown=treat_shown & ~end,
        eaten=eaten & ~end,
        dwell=jnp.where(end, -1, dwell).astype(jnp.int32),
        trial_steps=jnp.where(end, 0, ts.trial_steps + 1).astype(jnp.int32),
        trial_idx=(ts.trial_idx + end.astype(jnp.int32)).astype(jnp.int32),
        hits=(ts.hits + hit.astype(jnp.int32)).astype(jnp.int32),
    )
    return new_ts, events


def critic_privileged_vector(ts: TrialState, max_trials: int = 20) -> jnp.ndarray:
    """Privileged critic features, ``float32[..., 8]``.

    Layout: ``[onehot(bait, 2), onehot(choice + 1, 3), treat_shown,
    trial_idx / max_trials, dwell >= 0]``.  The choice one-hot reads
    ``[none, arm 0, arm 1]``.  ``trial_idx / max_trials`` is not clipped.  Works
    on a single state or on one with leading batch dimensions.

    Args:
      ts: trial state.
      max_trials: normaliser for ``trial_idx``.

    Returns:
      The 8-dim privileged vector.
    """
    f32 = jnp.float32
    return jnp.concatenate(
        [
            jax.nn.one_hot(ts.bait, 2, dtype=f32),
            jax.nn.one_hot(ts.choice + 1, 3, dtype=f32),
            ts.treat_shown.astype(f32)[..., None],
            (ts.trial_idx.astype(f32) / max_trials)[..., None],
            (ts.dwell >= 0).astype(f32)[..., None],
        ],
        axis=-1,
    )
