"""Unit tests for `vnl_playground.tasks.rodent.spout_logic` (CPU, no GPU).

The trial state machine is pure JAX and runs once per env step inside jitted,
batched code, so every test drives it through ``jax.jit``.  Arm ids follow
`spout_geometry.locate`: ``-1`` in the stem / junction, ``0`` left, ``1`` right.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vnl_playground.tasks.rodent.spout_logic import (
    SpoutParams,
    TrialState,
    critic_privileged_vector,
    init_trial_state,
    trial_update,
)

P = SpoutParams(
    n_arms=2,
    pay_prob=1.0,
    appear_distance=0.5,
    spout_radius=0.15,
    reach=0.1,
    dwell_steps=5,
    trial_timeout_steps=20,
)
FAR = 10.0  # a treat distance that is nowhere near the reach
UPDATE = jax.jit(trial_update)

EVENT_KEYS = {
    "reward",
    "show_treat",
    "trial_end",
    "hit",
    "trial_after_first",
    "hit_after_first",
    "timeout",
    "choice_none",
}


def _state(bait: int, **overrides) -> TrialState:
    """A fresh trial state with a chosen bait arm."""
    ts = init_trial_state(jax.random.PRNGKey(0), P)
    return ts._replace(bait=jnp.int32(bait), **overrides)


class Runner:
    """Steps `trial_update` with a fresh rng per step and records the events."""

    def __init__(self, ts: TrialState, p: SpoutParams = P, seed: int = 0):
        self.ts = ts
        self.p = p
        self.key = jax.random.PRNGKey(seed)
        self.events: list[dict] = []

    def step(self, arm: int, dist: float, treat_dist: float = FAR) -> dict:
        self.key, sub = jax.random.split(self.key)
        self.ts, ev = UPDATE(
            self.ts,
            jnp.int32(arm),
            jnp.float32(dist),
            jnp.float32(treat_dist),
            sub,
            self.p,
        )
        ev = {k: float(v) for k, v in ev.items()}
        self.events.append(ev)
        return ev

    def total(self, key: str) -> float:
        return sum(ev[key] for ev in self.events)

    def hold_until_end(
        self, arm: int, dist: float, treat_dist: float = FAR, max_steps: int = 100
    ):
        """Repeats one observation until the trial ends; returns (events, n_calls).

        Stops AT the end step: stepping on would start a new trial (the rat is
        still standing in the arm end, so it would lock a new choice).
        """
        for n in range(1, max_steps + 1):
            ev = self.step(arm, dist, treat_dist)
            if ev["trial_end"] == 1.0:
                return ev, n
        raise AssertionError("trial did not end")


# ---------------------------------------------------------------------------
# init / structure
# ---------------------------------------------------------------------------
def test_init_trial_state_fields_and_dtypes():
    ts = init_trial_state(jax.random.PRNGKey(3), P)
    assert ts._fields == (
        "bait",
        "choice",
        "treat_shown",
        "eaten",
        "dwell",
        "trial_steps",
        "trial_idx",
        "hits",
    )
    assert int(ts.choice) == -1
    assert not bool(ts.treat_shown) and not bool(ts.eaten)
    assert int(ts.dwell) == -1
    assert int(ts.trial_steps) == 0 and int(ts.trial_idx) == 0 and int(ts.hits) == 0
    expected = dict(
        bait=jnp.int32,
        choice=jnp.int32,
        treat_shown=jnp.bool_,
        eaten=jnp.bool_,
        dwell=jnp.int32,
        trial_steps=jnp.int32,
        trial_idx=jnp.int32,
        hits=jnp.int32,
    )
    for name, dt in expected.items():
        assert getattr(ts, name).dtype == dt, name


def test_init_bait_is_uniform_over_arms():
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    bait = np.asarray(jax.vmap(lambda k: init_trial_state(k, P).bait)(keys))
    assert set(np.unique(bait)) == {0, 1}
    assert 0.4 < bait.mean() < 0.6
    # More arms -> bait covers {0..n_arms-1}.
    p4 = P._replace(n_arms=4)
    bait4 = np.asarray(jax.vmap(lambda k: init_trial_state(k, p4).bait)(keys))
    assert set(np.unique(bait4)) == {0, 1, 2, 3}


def test_trial_state_dict_roundtrip():
    ts = _state(1, choice=jnp.int32(0), dwell=jnp.int32(3))
    d = ts._asdict()
    assert isinstance(d, dict) and set(d) == set(TrialState._fields)
    back = TrialState(**d)
    for a, b in zip(ts, back):
        assert a.dtype == b.dtype and bool(a == b)


def test_events_are_float32_scalars_with_expected_keys():
    ts = _state(0)
    _, ev = UPDATE(
        ts, jnp.int32(0), jnp.float32(0.4), jnp.float32(FAR), jax.random.PRNGKey(1), P
    )
    assert set(ev) == EVENT_KEYS
    for k, v in ev.items():
        assert v.dtype == jnp.float32 and v.shape == (), k


def test_update_preserves_state_dtypes():
    r = Runner(_state(0))
    for arm, d, td in [(0, 0.4, FAR), (0, 0.1, 0.05), (0, 0.1, 0.05)]:
        r.step(arm, d, td)
    assert r.ts.bait.dtype == jnp.int32 and r.ts.choice.dtype == jnp.int32
    assert r.ts.dwell.dtype == jnp.int32 and r.ts.trial_steps.dtype == jnp.int32
    assert r.ts.trial_idx.dtype == jnp.int32 and r.ts.hits.dtype == jnp.int32
    assert r.ts.treat_shown.dtype == jnp.bool_ and r.ts.eaten.dtype == jnp.bool_


# ---------------------------------------------------------------------------
# rule 1: choice lock + treat appearance
# ---------------------------------------------------------------------------
def test_lock_and_show_on_bait():
    r = Runner(_state(0))
    ev = r.step(0, 0.4)  # inside the appear zone of arm 0 (the bait)
    assert int(r.ts.choice) == 0
    assert bool(r.ts.treat_shown)
    assert ev["show_treat"] == 1.0
    assert ev["reward"] == 0.0 and ev["trial_end"] == 0.0
    # Show fires only on the lock step.
    ev2 = r.step(0, 0.35)
    assert ev2["show_treat"] == 0.0 and bool(r.ts.treat_shown)


def test_no_lock_outside_appear_zone_or_in_stem():
    r = Runner(_state(0))
    r.step(0, 0.6)  # in the arm but outside the last 0.5 m
    assert int(r.ts.choice) == -1 and not bool(r.ts.treat_shown)
    # arm_id == -1 (stem / junction): dist_to_end is meaningless, never locks.
    r.step(-1, 0.1)
    r.step(-1, -1.0)
    assert int(r.ts.choice) == -1 and not bool(r.ts.treat_shown)
    # The boundary is inclusive: dist == appear_distance locks.
    r.step(1, 0.5)
    assert int(r.ts.choice) == 1


def test_no_show_on_nonbait():
    r = Runner(_state(1))
    ev = r.step(0, 0.4)  # lock on arm 0, but the bait is arm 1
    assert int(r.ts.choice) == 0
    assert not bool(r.ts.treat_shown)
    assert ev["show_treat"] == 0.0


def test_pay_prob_zero_never_shows():
    p0 = P._replace(pay_prob=0.0)
    for seed in range(20):
        r = Runner(_state(0), p=p0, seed=seed)
        r.step(0, 0.4)  # lock on the bait arm
        assert int(r.ts.choice) == 0
        assert not bool(r.ts.treat_shown)
        assert r.total("show_treat") == 0.0


def test_pay_prob_is_a_per_trial_bernoulli():
    p_half = P._replace(pay_prob=0.5)
    shown = []
    for seed in range(200):
        r = Runner(_state(0), p=p_half, seed=seed)
        r.step(0, 0.4)
        shown.append(bool(r.ts.treat_shown))
    assert 0.35 < np.mean(shown) < 0.65


# ---------------------------------------------------------------------------
# rules 2 and 4: eat -> reward once -> dwell -> end
# ---------------------------------------------------------------------------
def test_reward_once_then_dwell_then_end():
    r = Runner(_state(0))
    r.step(0, 0.4)  # lock + show
    r.step(0, 0.3, treat_dist=0.3)  # still out of reach: nothing happens
    assert r.total("reward") == 0.0 and not bool(r.ts.eaten)
    eat_i = len(r.events)
    ev = r.step(0, 0.15, treat_dist=0.05)  # within reach -> eaten
    assert ev["reward"] == 1.0
    assert bool(r.ts.eaten)
    assert int(r.ts.dwell) == P.dwell_steps - 1  # rule 4 decrements on the same step
    # Keep standing on the treat: reward must not fire again before the trial ends.
    ev, _ = r.hold_until_end(0, 0.15, treat_dist=0.05)
    end_i = len(r.events) - 1
    assert r.total("reward") == 1.0
    # The dwell spans exactly dwell_steps steps, counting the eating step as the first.
    assert end_i == eat_i + P.dwell_steps - 1
    assert r.total("trial_end") == 1.0
    # Ending step reports the hit.
    assert ev["hit"] == 1.0 and ev["timeout"] == 0.0


def test_dwell_of_one_step_ends_on_the_eating_step():
    p1 = P._replace(dwell_steps=1)
    r = Runner(_state(0), p=p1)
    r.step(0, 0.4)
    ev = r.step(0, 0.15, treat_dist=0.05)
    assert ev["reward"] == 1.0 and ev["trial_end"] == 1.0 and ev["hit"] == 1.0


def test_treat_not_eaten_when_not_shown():
    # Non-bait choice: no treat, so being "at the treat" gives nothing.
    r = Runner(_state(1))
    r.step(0, 0.4)
    ev = r.step(0, 0.3, treat_dist=0.0)
    assert ev["reward"] == 0.0 and not bool(r.ts.eaten)


# ---------------------------------------------------------------------------
# rule 3: visit to a non-paying arm end, and the choice lock
# ---------------------------------------------------------------------------
def test_nonpaying_chosen_arm_visit_ends_trial_with_miss():
    r = Runner(_state(1))
    r.step(0, 0.4)  # lock on arm 0, bait is arm 1
    r.step(0, 0.2)  # not yet at the spout
    assert int(r.ts.dwell) == -1
    r.step(0, 0.1)  # within spout_radius -> dwell starts, no reward
    assert int(r.ts.dwell) == P.dwell_steps - 1
    ev, n = r.hold_until_end(0, 0.1)
    assert n == P.dwell_steps - 1  # the starting call counted as the first dwell step
    assert r.total("reward") == 0.0
    assert ev["hit"] == 0.0 and ev["timeout"] == 0.0
    assert r.total("trial_end") == 1.0 and r.total("hit") == 0.0
    assert r.total("show_treat") == 0.0


def test_choice_lock_blocks_other_arm():
    # Bait is arm 0; lock on the non-bait arm 1, then walk to arm 0's end.
    r = Runner(_state(0))
    r.step(1, 0.4)
    assert int(r.ts.choice) == 1
    for d in (0.4, 0.3, 0.2):
        r.step(0, d, treat_dist=0.0)  # even standing exactly on arm 0's treat
        assert int(r.ts.choice) == 1  # lock is never overwritten
        assert not bool(r.ts.treat_shown)  # other arm is never revealed
        assert int(r.ts.dwell) == -1
    r.step(0, 0.1, treat_dist=0.0)  # reaching arm 0's end starts the dwell ...
    assert int(r.ts.dwell) == P.dwell_steps - 1
    assert int(r.ts.choice) == 1 and not bool(r.ts.treat_shown)
    # ... which runs out with a miss: no treat, no reward.  Stop at the end step.
    end, _ = r.hold_until_end(0, -0.1, treat_dist=0.0)
    assert end["hit"] == 0.0 and end["timeout"] == 0.0 and end["choice_none"] == 0.0
    assert r.total("show_treat") == 0.0
    assert r.total("reward") == 0.0
    assert r.total("trial_end") == 1.0
    assert r.total("hit") == 0.0
    assert int(r.ts.hits) == 0 and int(r.ts.trial_idx) == 1


def test_bait_lock_unaffected_by_visiting_other_arm():
    # Locked on the bait arm with the treat shown; wander to the other arm's end.
    r = Runner(_state(0))
    r.step(0, 0.4)
    assert bool(r.ts.treat_shown)
    for _ in range(8):
        ev = r.step(1, 0.05)
        assert ev["reward"] == 0.0 and ev["trial_end"] == 0.0
    assert int(r.ts.dwell) == -1  # treat_shown suppresses the non-paying-visit dwell
    assert int(r.ts.choice) == 0
    # Going back and eating the treat still pays.
    ev = r.step(0, 0.15, treat_dist=0.05)
    assert ev["reward"] == 1.0


# ---------------------------------------------------------------------------
# rule 4: timeout
# ---------------------------------------------------------------------------
def test_timeout_ends_trial():
    r = Runner(_state(0))
    n_before_end = 0
    for _ in range(P.trial_timeout_steps + 1):
        ev = r.step(-1, 1.0)  # wanders in the stem forever, never choosing
        if ev["trial_end"] == 1.0:
            break
        n_before_end += 1
    # trial_steps counts completed steps; the trial ends on the step that sees it
    # at trial_timeout_steps, i.e. the (timeout + 1)-th call.
    assert n_before_end == P.trial_timeout_steps
    assert ev["trial_end"] == 1.0 and ev["timeout"] == 1.0
    assert ev["hit"] == 0.0 and ev["choice_none"] == 1.0
    assert int(r.ts.trial_idx) == 1 and int(r.ts.hits) == 0
    assert int(r.ts.trial_steps) == 0


def test_timeout_after_choice_is_not_choice_none():
    r = Runner(_state(1))
    r.step(0, 0.4)  # lock on the wrong arm and then stall
    ev = None
    for _ in range(P.trial_timeout_steps + 1):
        ev = r.step(-1, 1.0)
        if ev["trial_end"] == 1.0:
            break
    assert ev["trial_end"] == 1.0 and ev["timeout"] == 1.0
    assert ev["choice_none"] == 0.0 and ev["hit"] == 0.0


def test_no_events_on_ordinary_steps():
    r = Runner(_state(0))
    ev = r.step(-1, 1.0)
    assert all(v == 0.0 for v in ev.values())
    assert int(r.ts.trial_steps) == 1


# ---------------------------------------------------------------------------
# hit / after-first bookkeeping
# ---------------------------------------------------------------------------
def _run_hit_trial(r: Runner, arm: int) -> dict:
    """Walks to `arm` and eats the treat; returns the trial-end events."""
    r.step(arm, 0.4)
    r.step(arm, 0.15, treat_dist=0.05)
    for _ in range(P.dwell_steps + 1):
        ev = r.step(arm, 0.15, treat_dist=0.05)
        if ev["trial_end"] == 1.0:
            return ev
    raise AssertionError("trial did not end")


def test_hit_after_first_counts():
    r = Runner(_state(0))
    ev0 = _run_hit_trial(r, 0)  # trial 0: a hit, but it is the first trial
    assert ev0["hit"] == 1.0
    assert ev0["trial_after_first"] == 0.0 and ev0["hit_after_first"] == 0.0
    assert int(r.ts.trial_idx) == 1 and int(r.ts.hits) == 1

    ev1 = _run_hit_trial(r, 0)  # trial 1: hit, and counts as after-first
    assert ev1["hit"] == 1.0
    assert ev1["trial_after_first"] == 1.0 and ev1["hit_after_first"] == 1.0
    assert int(r.ts.trial_idx) == 2 and int(r.ts.hits) == 2

    # trial 2: chooses the wrong arm -> a miss that still counts as after-first.
    r.step(1, 0.4)
    ev2 = None
    for _ in range(P.dwell_steps + 3):
        ev2 = r.step(1, 0.1)
        if ev2["trial_end"] == 1.0:
            break
    assert ev2["trial_end"] == 1.0
    assert ev2["hit"] == 0.0
    assert ev2["trial_after_first"] == 1.0 and ev2["hit_after_first"] == 0.0
    assert int(r.ts.trial_idx) == 3 and int(r.ts.hits) == 2
    # bait persists across trials within the episode.
    assert int(r.ts.bait) == 0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------
def test_trial_reset_fields():
    r = Runner(_state(0))
    _run_hit_trial(r, 0)
    init = init_trial_state(jax.random.PRNGKey(0), P)
    for name in ("choice", "treat_shown", "eaten", "dwell", "trial_steps"):
        assert bool(getattr(r.ts, name) == getattr(init, name)), name
        assert getattr(r.ts, name).dtype == getattr(init, name).dtype, name
    # Only bait / trial_idx / hits carry over.
    assert int(r.ts.bait) == 0 and int(r.ts.trial_idx) == 1 and int(r.ts.hits) == 1


def test_trial_reset_after_miss_and_timeout():
    init = init_trial_state(jax.random.PRNGKey(0), P)
    # after a miss via rule 3
    r = Runner(_state(1))
    r.step(0, 0.4)
    r.hold_until_end(0, 0.1)
    # after a timeout
    r2 = Runner(_state(1))
    r2.step(0, 0.4)
    r2.hold_until_end(-1, 1.0)
    for rr in (r, r2):
        for name in ("choice", "treat_shown", "eaten", "dwell", "trial_steps"):
            assert bool(getattr(rr.ts, name) == getattr(init, name)), name
        assert int(rr.ts.trial_idx) == 1


def test_next_trial_start_is_history_independent():
    # After a trial ends, the next trial behaves exactly like a fresh state with
    # the same bait / trial_idx, whatever happened in the previous trial.
    a = Runner(_state(0))
    _run_hit_trial(a, 0)  # previous trial: a paid hit
    b = Runner(_state(0))
    b.step(1, 0.4)  # previous trial: wrong arm, then a non-paying visit
    b.hold_until_end(1, 0.1)
    fresh = Runner(_state(0, trial_idx=jnp.int32(1)))
    runners = (a, b, fresh)
    for rr in runners[:2]:
        assert int(rr.ts.trial_idx) == 1
    seq = [(-1, 1.0, FAR), (0, 0.4, FAR), (0, 0.3, FAR), (0, 0.15, 0.05)]
    seq += [(0, 0.15, 0.05)] * P.dwell_steps
    for arm, d, td in seq:
        evs = [rr.step(arm, d, td) for rr in runners]
        assert evs[0] == evs[1] == evs[2]
        for name in (
            "bait",
            "choice",
            "treat_shown",
            "eaten",
            "dwell",
            "trial_steps",
            "trial_idx",
        ):
            vals = [
                bool(getattr(rr.ts, name) == getattr(fresh.ts, name)) for rr in (a, b)
            ]
            assert all(vals), name
    assert a.total("trial_end") == 2.0 and fresh.total("trial_end") == 1.0


# ---------------------------------------------------------------------------
# critic-privileged vector
# ---------------------------------------------------------------------------
def test_critic_privileged_vector_layout():
    ts = _state(
        1,
        choice=jnp.int32(0),
        treat_shown=jnp.bool_(True),
        dwell=jnp.int32(7),
        trial_idx=jnp.int32(5),
    )
    v = critic_privileged_vector(ts, max_trials=20)
    assert v.shape == (8,) and v.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(v), [0, 1, 0, 1, 0, 1, 0.25, 1])

    init = init_trial_state(jax.random.PRNGKey(0), P)._replace(bait=jnp.int32(0))
    v0 = critic_privileged_vector(init)
    # bait one-hot, choice=-1 one-hot at slot 0, no treat, trial 0, no dwell.
    np.testing.assert_allclose(np.asarray(v0), [1, 0, 1, 0, 0, 0, 0, 0])

    # choice 1 -> slot 2 of the choice one-hot; dwell == 0 still counts as active.
    ts2 = _state(0, choice=jnp.int32(1), dwell=jnp.int32(0))
    v2 = critic_privileged_vector(ts2, max_trials=10)
    np.testing.assert_allclose(np.asarray(v2), [1, 0, 0, 0, 1, 0, 0, 1])


def test_critic_privileged_vector_batched():
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    ts = jax.vmap(lambda k: init_trial_state(k, P))(keys)
    v = jax.vmap(critic_privileged_vector)(ts)
    assert v.shape == (6, 8)
    vb = critic_privileged_vector(ts)  # leading batch dims work without vmap too
    np.testing.assert_allclose(np.asarray(v), np.asarray(vb))


# ---------------------------------------------------------------------------
# jit + vmap
# ---------------------------------------------------------------------------
def test_jit_vmap():
    n = 8
    keys = jax.random.split(jax.random.PRNGKey(7), n)
    ts = jax.vmap(lambda k: init_trial_state(k, P))(keys)
    assert ts.bait.shape == (n,) and ts.treat_shown.dtype == jnp.bool_

    batched = jax.jit(jax.vmap(trial_update, in_axes=(0, 0, 0, 0, 0, None)))
    # Each env walks into the arm given by (env index % 2) and stays in the appear zone.
    arm = jnp.arange(n, dtype=jnp.int32) % 2
    dist = jnp.full((n,), 0.4, jnp.float32)
    treat = jnp.full((n,), FAR, jnp.float32)
    step_keys = jax.random.split(jax.random.PRNGKey(11), n)
    ts1, ev = batched(ts, arm, dist, treat, step_keys, P)

    assert set(ev) == EVENT_KEYS
    for k, v in ev.items():
        assert v.shape == (n,) and v.dtype == jnp.float32, k
    np.testing.assert_array_equal(np.asarray(ts1.choice), np.asarray(arm))
    # pay_prob = 1: the treat shows exactly on the envs whose chosen arm is the bait.
    expected_show = np.asarray(arm == ts.bait)
    np.testing.assert_array_equal(np.asarray(ts1.treat_shown), expected_show)
    np.testing.assert_array_equal(np.asarray(ev["show_treat"]) == 1.0, expected_show)

    # Matches the unbatched function env by env.
    for i in range(n):
        ts_i = jax.tree_util.tree_map(lambda x, i=i: x[i], ts)
        out_i, ev_i = trial_update(ts_i, arm[i], dist[i], treat[i], step_keys[i], P)
        for a, b in zip(out_i, jax.tree_util.tree_map(lambda x, i=i: x[i], ts1)):
            assert bool(a == b)
        for k in ev_i:
            assert float(ev_i[k]) == float(ev[k][i])

    # Second step through the dict form (what the env stores in `state.info`).
    d = ts1._asdict()
    ts2, _ = batched(TrialState(**d), arm, dist, treat, step_keys, P)
    np.testing.assert_array_equal(np.asarray(ts2.trial_steps), np.ones(n, np.int32) * 2)


def test_scan_over_a_trial_under_jit():
    """The state carries through `lax.scan` with stable dtypes (as in an env rollout)."""

    def body(ts, inp):
        arm, dist, treat_dist, key = inp
        ts, ev = trial_update(ts, arm, dist, treat_dist, key, P)
        return ts, ev

    # lock + show at 0, eat at 3, dwell 5 -> trial_end at call 7; stop there.
    T = 3 + P.dwell_steps
    arm = jnp.zeros((T,), jnp.int32)
    dist = jnp.concatenate([jnp.full((2,), 0.4), jnp.full((T - 2,), 0.15)]).astype(
        jnp.float32
    )
    treat = jnp.concatenate([jnp.full((3,), FAR), jnp.full((T - 3,), 0.05)]).astype(
        jnp.float32
    )
    keys = jax.random.split(jax.random.PRNGKey(0), T)
    ts0 = _state(0)
    ts_end, evs = jax.jit(lambda t: jax.lax.scan(body, t, (arm, dist, treat, keys)))(
        ts0
    )
    assert float(evs["reward"].sum()) == 1.0
    assert float(evs["trial_end"].sum()) == 1.0
    assert float(evs["hit"].sum()) == 1.0
    assert int(ts_end.trial_idx) == 1 and int(ts_end.hits) == 1


@pytest.mark.parametrize("dist", [0.5, 0.4999, 0.0, -0.2])
def test_lock_inside_appear_zone_including_past_the_end(dist):
    # dist_to_end goes negative past arm_end_xy (the end alcove); still in the zone.
    r = Runner(_state(1))
    r.step(1, dist)
    assert int(r.ts.choice) == 1 and bool(r.ts.treat_shown)
