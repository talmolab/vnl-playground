"""CPU tests for the inter-trial hold / gate state machine (`spout_hold`).

Run with::

    JAX_PLATFORMS=cpu python -m pytest tests/test_spout_hold.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vnl_playground.tasks.rodent import spout_hold, spout_logic
from vnl_playground.tasks.rodent.spout_hold import HoldState

# The T-maze defaults: hold 1-3 s at ctrl_dt 0.01, gate 0.3 m tall, parked 1 m below the floor.
P = spout_hold.make_hold_params(
    hold_min_s=1.0,
    hold_max_s=3.0,
    ctrl_dt=0.01,
    gate_boundaries=(1,),
    gate_open_steps=20,
    gate_height=0.3,
    park_z=-1.3,
    n_stem_segments=5,
)


def _hs(hold_left, gate_k=1, open_t=0) -> HoldState:
    return HoldState(
        hold_left=jnp.int32(hold_left),
        gate_k=jnp.int32(gate_k),
        open_t=jnp.int32(open_t),
    )


def _run(hs, p, n, trial_end=None, seed=0):
    """``n`` eager hold_update calls; returns the per-step (state, events, gate_z) lists."""
    key = jax.random.PRNGKey(seed)
    states, events, zs = [], [], []
    for t in range(n):
        key, sub = jax.random.split(key)
        end = jnp.bool_(False) if trial_end is None else jnp.bool_(trial_end[t])
        hs, ev = spout_hold.hold_update(hs, end, sub, p)
        states.append(hs)
        events.append({k: float(v) for k, v in ev.items()})
        zs.append(float(spout_hold.gate_z(hs, p)))
    return states, events, zs


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_make_hold_params_seconds_to_steps():
    assert P.hold_min_steps == 100 and P.hold_max_steps == 300
    assert P.gate_boundaries == (1,)
    assert P.gate_open_steps == 20
    assert isinstance(P.hold_min_steps, int) and isinstance(P.hold_max_steps, int)
    p = spout_hold.make_hold_params(
        hold_min_s=0.5,
        hold_max_s=0.5,
        ctrl_dt=0.01,
        gate_boundaries=[2, 5],
        gate_open_steps=4,
        gate_height=0.3,
        park_z=-1.3,
        n_stem_segments=5,
    )
    assert (p.hold_min_steps, p.hold_max_steps) == (50, 50)
    assert p.gate_boundaries == (2, 5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(hold_min_s=0.0), "hold_min_s"),  # rounds to 0 steps
        (dict(hold_min_s=3.0, hold_max_s=1.0), "hold_max_s"),
        (dict(gate_boundaries=()), "gate_boundaries"),
        (dict(gate_boundaries=(0,)), "gate_boundaries"),  # 0 is the back wall
        (dict(gate_boundaries=(6,)), "gate_boundaries"),  # past the junction edge
        (dict(gate_open_steps=0), "gate_open_steps"),
        (dict(park_z=-0.2), "park_z"),  # not below the floor
    ],
)
def test_make_hold_params_rejects(kwargs, match):
    base = dict(
        hold_min_s=1.0,
        hold_max_s=3.0,
        ctrl_dt=0.01,
        gate_boundaries=(1,),
        gate_open_steps=20,
        gate_height=0.3,
        park_z=-1.3,
        n_stem_segments=5,
    )
    with pytest.raises(ValueError, match=match):
        spout_hold.make_hold_params(**{**base, **kwargs})


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------


def test_hold_draw_range_and_mean():
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    hs = jax.vmap(lambda k: spout_hold.hold_init(k, P))(keys)
    hold = np.asarray(hs.hold_left)
    assert hold.dtype == np.int32
    assert hold.min() >= 100 and hold.max() <= 300
    assert abs(hold.mean() - 200.0) <= 5.0, hold.mean()
    # Uniform over the integers, both ends included.
    assert hold.min() <= 102 and hold.max() >= 298
    assert np.all(np.asarray(hs.open_t) == 0)
    assert np.all(np.asarray(hs.gate_k) == 1)
    assert np.all(np.asarray(jax.vmap(spout_hold.holding)(hs)))


def test_fixed_hold_when_min_equals_max():
    p = P._replace(hold_min_steps=150, hold_max_steps=150)
    keys = jax.random.split(jax.random.PRNGKey(1), 256)
    hs = jax.vmap(lambda k: spout_hold.hold_init(k, p))(keys)
    assert np.all(np.asarray(hs.hold_left) == 150)
    # The re-arm draw too.
    done = _hs(0, open_t=p.gate_open_steps)
    re = jax.vmap(lambda k: spout_hold.hold_update(done, jnp.bool_(True), k, p)[0])(
        keys
    )
    assert np.all(np.asarray(re.hold_left) == 150)


def test_boundary_sampled_only_from_gate_boundaries():
    p = P._replace(gate_boundaries=(1, 3, 5))
    keys = jax.random.split(jax.random.PRNGKey(2), 4096)
    k = np.asarray(jax.vmap(lambda kk: spout_hold.hold_init(kk, p))(keys).gate_k)
    assert set(np.unique(k)) == {1, 3, 5}
    for b in (1, 3, 5):
        assert abs((k == b).mean() - 1 / 3) < 0.04, (b, (k == b).mean())
    done = _hs(0, gate_k=3, open_t=p.gate_open_steps)
    k2 = np.asarray(
        jax.vmap(lambda kk: spout_hold.hold_update(done, jnp.bool_(True), kk, p)[0])(
            keys
        ).gate_k
    )
    assert set(np.unique(k2)) == {1, 3, 5}
    # A single entry is fixed.
    k1 = np.asarray(jax.vmap(lambda kk: spout_hold.hold_init(kk, P))(keys).gate_k)
    assert np.all(k1 == 1)


# ---------------------------------------------------------------------------
# Countdown, release, gate animation, re-arm
# ---------------------------------------------------------------------------


def test_countdown_release_and_hold_active():
    h = 5
    states, events, _ = _run(_hs(h), P, 10)
    left = [int(s.hold_left) for s in states]
    assert left == [4, 3, 2, 1, 0, 0, 0, 0, 0, 0]
    # hold_active: the step STARTED in the hold (the trial is frozen on it): exactly H steps.
    assert [e["hold_active"] for e in events] == [1.0] * h + [0.0] * 5
    # gate_release fires once, on the step hold_left reaches 0.
    assert [e["gate_release"] for e in events] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert spout_hold.holding(_hs(1)) and not spout_hold.holding(_hs(0))


def test_gate_z_closed_then_linear_drop_then_parked():
    n = P.gate_open_steps
    h = 3
    assert float(spout_hold.gate_z(_hs(h), P)) == 0.0  # at reset: closed
    states, _, zs = _run(_hs(h), P, h + n + 10)
    open_t = [int(s.open_t) for s in states]
    # Closed (0) on every step that ends still holding.
    for s, z in zip(states, zs):
        if bool(spout_hold.holding(s)):
            assert z == 0.0
    # Release on call h (index h - 1): open_t counts 1..n, then saturates at n.
    assert open_t[: h - 1] == [0] * (h - 1)
    assert open_t[h - 1 : h - 1 + n] == list(range(1, n + 1))
    assert all(t == n for t in open_t[h - 1 + n :])
    drop = zs[h - 1 : h - 1 + n]
    # Strictly decreasing over exactly n steps, linear while the gate is still above the floor.
    assert all(b < a for a, b in zip([0.0, *drop], drop))
    np.testing.assert_allclose(
        drop[:-1], -P.gate_height * np.arange(1, n) / n, rtol=1e-6
    )
    assert (
        drop[-2] > -P.gate_height
    )  # the top is still above the floor one step before the end
    # Then parked, fully below the floor, and constant.
    assert drop[-1] == pytest.approx(P.park_z)
    assert P.park_z <= -P.gate_height
    assert all(z == pytest.approx(P.park_z) for z in zs[h - 1 + n :])


def test_rearm_on_trial_end():
    n = P.gate_open_steps
    released = _hs(0, gate_k=1, open_t=n)
    assert float(spout_hold.gate_z(released, P)) == pytest.approx(P.park_z)
    hs, ev = spout_hold.hold_update(released, jnp.bool_(True), jax.random.PRNGKey(3), P)
    assert 100 <= int(hs.hold_left) <= 300
    assert int(hs.open_t) == 0 and int(hs.gate_k) == 1
    assert bool(spout_hold.holding(hs))
    assert float(spout_hold.gate_z(hs, P)) == 0.0  # closed again on the re-arm step
    assert ev["gate_release"] == 0.0 and ev["hold_active"] == 0.0
    # No trial end: nothing is re-drawn.
    hs2, _ = spout_hold.hold_update(
        released, jnp.bool_(False), jax.random.PRNGKey(3), P
    )
    assert (int(hs2.hold_left), int(hs2.open_t), int(hs2.gate_k)) == (0, n, 1)


def test_dtypes():
    hs = spout_hold.hold_init(jax.random.PRNGKey(0), P)
    for leaf in hs:
        assert leaf.dtype == jnp.int32 and leaf.shape == ()
    hs, ev = spout_hold.hold_update(hs, jnp.bool_(False), jax.random.PRNGKey(1), P)
    for leaf in hs:
        assert leaf.dtype == jnp.int32 and leaf.shape == ()
    assert set(ev) == set(spout_hold.HOLD_EVENTS)
    for v in ev.values():
        assert v.dtype == jnp.float32
    assert spout_hold.gate_z(hs, P).dtype == jnp.float32


def test_jit_vmap_matches_eager():
    n_env, n_steps = 16, 40
    p = P._replace(
        hold_min_steps=3,
        hold_max_steps=12,
        gate_boundaries=(1, 2, 4),
        gate_open_steps=5,
    )
    keys = jax.random.split(jax.random.PRNGKey(7), n_env)
    ends = np.zeros((n_steps, n_env), dtype=bool)
    ends[25, ::3] = True  # some envs re-arm mid-run

    def rollout(key, end_col):
        k0, k1 = jax.random.split(key)
        hs0 = spout_hold.hold_init(k0, p)

        def body(carry, end):
            hs, k = carry
            k, sub = jax.random.split(k)
            hs, ev = spout_hold.hold_update(hs, end, sub, p)
            return (hs, k), (hs, ev, spout_hold.gate_z(hs, p))

        _, out = jax.lax.scan(body, (hs0, k1), end_col)
        return out

    batched = jax.jit(jax.vmap(rollout))(keys, jnp.asarray(ends.T))
    for e in (0, 5, 9):
        eager = rollout(keys[e], jnp.asarray(ends[:, e]))
        for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(batched)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b)[e])


# ---------------------------------------------------------------------------
# Combined with spout_logic: the trial clock is frozen during the hold
# ---------------------------------------------------------------------------


def _harness(p_trial, p_hold, n_steps, arm_id, dist_to_end, treat_dist=1.0, seed=0):
    """The env's per-step order: trial_update -> freeze while holding -> hold_update."""
    key = jax.random.PRNGKey(seed)
    key, k_bait, k_hold = jax.random.split(key, 3)
    ts = spout_logic.init_trial_state(k_bait, p_trial)
    hs = spout_hold.hold_init(k_hold, p_hold)
    rows = []
    for _ in range(n_steps):
        key, k_pay, k_h = jax.random.split(key, 3)
        frozen = spout_hold.holding(hs)
        new_ts, ev = spout_logic.trial_update(
            ts, arm_id, dist_to_end, treat_dist, k_pay, p_trial
        )
        ts, ev = spout_hold.freeze_trial(frozen, ts, new_ts, ev)
        hs, hev = spout_hold.hold_update(hs, ev["trial_end"] > 0.0, k_h, p_hold)
        rows.append((ts, ev, hs, hev, bool(frozen)))
    return rows


def test_trial_clock_frozen_during_hold():
    h = 6
    p_hold = P._replace(hold_min_steps=h, hold_max_steps=h)
    p_trial = spout_logic.SpoutParams()
    # The rat sits inside arm 0's appear zone: unfrozen, the choice would lock at once.
    rows = _harness(p_trial, p_hold, h + 4, arm_id=0, dist_to_end=0.3)
    for i, (ts, ev, _hs, hev, frozen) in enumerate(rows):
        if i < h:
            assert frozen and hev["hold_active"] == 1.0
            assert int(ts.trial_steps) == 0 and int(ts.choice) == -1, i
            assert not bool(ts.treat_shown)
            assert all(float(v) == 0.0 for v in ev.values()), (i, ev)
        else:
            assert not frozen and hev["hold_active"] == 0.0
    assert [r[3]["gate_release"] for r in rows] == [0.0] * (h - 1) + [1.0] + [0.0] * 4
    # Released: the choice locks on the first unfrozen step, and the clock starts.
    assert int(rows[h][0].choice) == 0
    assert [int(r[0].trial_steps) for r in rows[h:]] == [1, 2, 3, 4]


def test_no_event_fires_during_hold_even_with_zero_timeout():
    """trial_timeout_steps=0 ends every unfrozen step; during the hold nothing ends."""
    h = 5
    p_hold = P._replace(hold_min_steps=h, hold_max_steps=h)
    p_trial = spout_logic.SpoutParams(trial_timeout_steps=0)
    rows = _harness(p_trial, p_hold, h + 3, arm_id=-1, dist_to_end=10.0)
    ends = [float(r[1]["trial_end"]) for r in rows]
    assert ends[:h] == [0.0] * h
    # First unfrozen step times out, and the trial end re-arms the hold.
    assert ends[h] == 1.0
    hs_after = rows[h][2]
    assert int(hs_after.hold_left) == h and int(hs_after.open_t) == 0
    assert int(rows[h][0].trial_idx) == 1
    # ... so the next h steps are frozen again.
    assert ends[h + 1 :] == [0.0] * (len(ends) - h - 1)
    assert all(int(r[0].trial_idx) == 1 for r in rows[h:])


def test_freeze_trial_passes_through_when_not_holding():
    ts = spout_logic.init_trial_state(jax.random.PRNGKey(0), spout_logic.SpoutParams())
    new_ts, ev = spout_logic.trial_update(
        ts, 0, 0.3, 1.0, jax.random.PRNGKey(1), spout_logic.SpoutParams()
    )
    out_ts, out_ev = spout_hold.freeze_trial(jnp.bool_(False), ts, new_ts, ev)
    for a, b in zip(out_ts, new_ts):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for k in ev:
        np.testing.assert_array_equal(np.asarray(out_ev[k]), np.asarray(ev[k]))
    frozen_ts, frozen_ev = spout_hold.freeze_trial(jnp.bool_(True), ts, new_ts, ev)
    for a, b in zip(frozen_ts, ts):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert np.asarray(a).dtype == np.asarray(b).dtype
    assert all(float(v) == 0.0 for v in frozen_ev.values())
    assert set(frozen_ev) == set(ev)
