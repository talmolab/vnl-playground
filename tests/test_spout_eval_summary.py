"""Unit tests for ``spout_eval_summary`` / ``spout_eval_log_line`` (CPU, no GPU).

``spout_eval_summary`` consumes the all-env ``[T, N]`` eval arrays
(``allenv``): ``done`` plus, for every ``spout/<name>`` the eval shim mirrors,
``anchor/_spout_eval/<name>``.  Episode boundary convention (pinned here): a
``done`` at step ``t`` is the LAST step of that episode -- the env auto-resets
inside the same step and the next episode's first step is ``t + 1``; the
``spout/*`` events of the done step belong to the OLD episode.

The builders below are deliberately independent of the implementation under
test: they emit the four trial events the way ``spout_logic.trial_update``
does (``trial_after_first`` = a trial end with ``trial_idx >= 1``, where the
env's own ``trial_idx`` resets on auto-reset), so the legacy keys and the new
per-trial-index keys are checked against the same ground truth.
"""

import math

import numpy as np
import pytest

from vnl_playground.tasks.rodent.spout_eval import (
    SPOUT_ALIAS,
    SpoutEvalMetricAlias,
    spout_eval_log_line,
    spout_eval_summary,
)

NEW_KEYS = (
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
LEGACY_KEYS = (
    "episodes",
    "trials_per_ep",
    "hit_rate",
    "hit_rate_after_first",
    "treats_per_ep",
    "timeout_frac",
    "fall_frac",
)


def _allenv(done, trial_end, hit, timeout=None, treat=None, fell=None):
    """``allenv`` dict from boolean/0-1 ``[T, N]`` arrays.

    ``trial_after_first`` / ``hit_after_first`` are derived with the ENV's own
    counter (completed trials this episode, reset after a done step), not with
    the implementation's index arithmetic.
    """
    done = np.asarray(done, bool)
    te = np.asarray(trial_end, bool)
    h = np.asarray(hit, bool) & te
    t_len, n_envs = done.shape
    taf = np.zeros((t_len, n_envs), np.float32)
    haf = np.zeros((t_len, n_envs), np.float32)
    for n in range(n_envs):
        idx = 0
        for t in range(t_len):
            if te[t, n] and idx >= 1:
                taf[t, n] = 1.0
                haf[t, n] = float(h[t, n])
            idx += int(te[t, n])
            if done[t, n]:
                idx = 0
    out = {
        "done": done.astype(np.float32),
        SPOUT_ALIAS + "trial_end": te.astype(np.float32),
        SPOUT_ALIAS + "hit": h.astype(np.float32),
        SPOUT_ALIAS + "trial_after_first": taf,
        SPOUT_ALIAS + "hit_after_first": haf,
        SPOUT_ALIAS + "timeout": (
            np.zeros((t_len, n_envs), np.float32)
            if timeout is None
            else np.asarray(timeout, np.float32)
        ),
    }
    if treat is not None:
        out["rewards/treat_collected"] = np.asarray(treat, np.float32)
    if fell is not None:
        out["terminations/belly_up"] = np.asarray(fell, np.float32)
    return out


def _scripted(envs, trial_len=10, tail=0, final_done=False):
    """Hand-made ``allenv`` from per-env scripts.

    ``envs[n]`` is a list of episodes; an episode is the list of its trials'
    hit outcomes (1/0).  Trial ``k`` of an episode ends ``trial_len`` steps
    after the previous one (the first ``trial_len`` steps into the episode);
    ``tail`` idle steps follow the last trial.  A ``done`` is written on the
    last step of every episode but the env's final one (which is cut by the
    end of the window, as a real eval window cuts it), unless ``final_done``.
    Shorter envs are padded with idle steps to the common window length.
    """
    lengths = [sum(len(ep) * trial_len + tail for ep in episodes) for episodes in envs]
    t_len = max(lengths)
    n_envs = len(envs)
    done = np.zeros((t_len, n_envs), bool)
    te = np.zeros((t_len, n_envs), bool)
    hit = np.zeros((t_len, n_envs), bool)
    for n, episodes in enumerate(envs):
        t = 0
        for e, ep in enumerate(episodes):
            for k, outcome in enumerate(ep):
                end = t + (k + 1) * trial_len - 1
                te[end, n] = True
                hit[end, n] = bool(outcome)
            t += len(ep) * trial_len + tail
            last_episode = e == len(episodes) - 1
            if not last_episode or final_done:
                done[t - 1, n] = True
    return _allenv(done, te, hit)


def _fixed_input():
    """A deterministic, RNG-free ``[T, N]`` window with every event type."""
    t_len, n_envs = 240, 7
    t = np.arange(t_len)[:, None]
    n = np.arange(n_envs)[None, :]
    done = ((t + 11 * n) % 53) == 52
    te = ((t * (n + 3) + 2 * n) % 17) == 0
    hit = ((t * 5 + n * 3) % 7) < 3
    timeout = te & (((t + n) % 5) == 0)
    treat = ((t + 4 * n) % 23) == 0
    fell = done & (((t + n) % 3) == 0)
    return _allenv(done, te, hit, timeout=timeout, treat=treat, fell=fell)


# Captured from the implementation before the per-trial-index keys were added,
# on `_fixed_input()`; the legacy keys must never move.
LEGACY_GOLDEN = {
    "episodes": 37,
    "trials_per_ep": 99 / 37,
    "hit_rate": 43 / 99,
    "hit_rate_after_first": 29 / 63,
    "treats_per_ep": 73 / 37,
    "timeout_frac": 21 / 99,
    "fall_frac": 11 / 37,
}


# -- (a) memoryless deterministic policy: the pooled estimator is biased ------
def test_memoryless_policy_is_unbiased_per_index_but_not_pooled():
    # half the envs always hit (6 trials fit the window), half always miss
    # (only 4): the always-hit envs finish more trials, so they get more
    # weight in a pooled average.
    envs = [[[1] * 6]] * 4 + [[[0] * 4]] * 4
    sp = spout_eval_summary(_scripted(envs))
    assert sp["hit_t0"] == pytest.approx(0.5)
    assert sp["hit_t1"] == pytest.approx(0.5)
    assert sp["hit_t2"] == pytest.approx(0.5)
    assert (sp["n_t0"], sp["n_t1"], sp["n_t2"]) == (8, 8, 8)
    assert sp["hit_ge1_envmean"] == pytest.approx(0.5)
    assert sp["n_ge1_segments"] == 8
    # index >= 3 is the first place the groups differ in trial count: always-hit
    # envs contribute indices 3, 4, 5, always-miss only 3 -> 12 hits / 16 trials
    assert sp["n_t3p"] == 16
    assert sp["hit_t3p"] == pytest.approx(0.75)
    # the legacy pooled estimator is inflated: 20 hits / 32 trials after first
    assert sp["hit_rate_after_first"] == pytest.approx(20 / 32)
    assert sp["hit_rate_after_first"] > 0.5


# -- (b) perfect memory ---------------------------------------------------------
def test_perfect_memory_coin_flip_then_always_hit():
    envs = [[[1] + [1] * 5]] * 3 + [[[0] + [1] * 5]] * 5
    sp = spout_eval_summary(_scripted(envs))
    assert sp["hit_t0"] == pytest.approx(3 / 8)
    assert sp["n_t0"] == 8
    assert sp["hit_t1"] == 1.0
    assert sp["hit_t2"] == 1.0
    assert sp["hit_t3p"] == 1.0
    assert sp["hit_ge1_envmean"] == 1.0
    assert (sp["n_t1"], sp["n_t2"], sp["n_t3p"]) == (8, 8, 24)
    assert sp["n_ge1_segments"] == 8


# -- (c) episode boundaries -----------------------------------------------------
def test_indices_restart_at_each_episode_boundary():
    # one env, two episodes; the first episode's LAST step is its third
    # trial's end step (tail=0), so the done coincides with a trial end that
    # must be counted in the OLD episode (index 2), not as index 0 of the next.
    sp = spout_eval_summary(_scripted([[[1, 0, 1], [0, 1, 1, 1]]]))
    assert sp["episodes"] == 2
    assert (sp["n_t0"], sp["n_t1"], sp["n_t2"], sp["n_t3p"]) == (2, 2, 2, 1)
    assert sp["hit_t0"] == pytest.approx(0.5)  # ep0 trial0=1, ep1 trial0=0
    assert sp["hit_t1"] == pytest.approx(0.5)  # ep0 trial1=0, ep1 trial1=1
    assert sp["hit_t2"] == 1.0  # ep0 trial2=1, ep1 trial2=1
    assert sp["hit_t3p"] == 1.0  # ep1 trial3=1
    # per segment: ep0 index>=1 trials [0, 1] -> 0.5; ep1 [1, 1, 1] -> 1.0
    assert sp["n_ge1_segments"] == 2
    assert sp["hit_ge1_envmean"] == pytest.approx(0.75)
    # legacy (env-side counter) agrees with the index>=1 trial population
    n_ge1 = sp["n_t1"] + sp["n_t2"] + sp["n_t3p"]
    h_ge1 = (
        sp["hit_t1"] * sp["n_t1"]
        + sp["hit_t2"] * sp["n_t2"]
        + sp["hit_t3p"] * sp["n_t3p"]
    )
    assert sp["hit_rate_after_first"] == pytest.approx(h_ge1 / n_ge1)


def test_trials_are_attributed_to_the_right_segment():
    # Segment means must not be pooled across the boundary: ep0 index>=1 trials
    # are all misses, ep1 index>=1 trials all hits, and ep0 has three times the
    # trials -> envmean 0.5, pooled-by-trial 0.25.
    sp = spout_eval_summary(_scripted([[[1, 0, 0, 0, 0], [0, 1, 1]]]))
    assert sp["n_ge1_segments"] == 2
    assert sp["hit_ge1_envmean"] == pytest.approx(0.5)
    assert sp["hit_rate_after_first"] == pytest.approx(2 / 6)


def test_done_on_last_step_opens_no_extra_segment():
    # the window ends exactly on the episode's done: that env has ONE episode
    # (the next one has no steps), and its trials keep their indices.
    sp = spout_eval_summary(_scripted([[[1, 1, 0]]], final_done=True))
    assert sp["episodes"] == 1
    assert (sp["n_t0"], sp["n_t1"], sp["n_t2"], sp["n_t3p"]) == (1, 1, 1, 0)
    assert sp["n_ge1_segments"] == 1
    assert sp["hit_ge1_envmean"] == pytest.approx(0.5)


def test_idle_tail_and_trial_free_segments():
    # idle steps after the last trial (a partial trial in flight at the done)
    # do not shift indices; a segment with 0 or 1 trials has no index>=1 trial
    # and is not counted in n_ge1_segments.
    envs = [[[1], [], [0, 1], [1, 1, 0]]]
    sp = spout_eval_summary(_scripted(envs, tail=7))
    assert sp["episodes"] == 4
    assert (sp["n_t0"], sp["n_t1"], sp["n_t2"], sp["n_t3p"]) == (3, 2, 1, 0)
    assert sp["n_ge1_segments"] == 2
    assert sp["hit_t0"] == pytest.approx(2 / 3)  # first trials: 1, (none), 0, 1
    assert sp["hit_ge1_envmean"] == pytest.approx((1.0 + 0.5) / 2)


def test_per_env_segments_are_independent():
    # env 0 resets every 3 trials, env 1 never: the boundaries are per env.
    sp = spout_eval_summary(_scripted([[[1, 1, 1], [1, 1, 1]], [[0, 0, 0, 0, 0, 0]]]))
    assert sp["n_t0"] == 3 and sp["n_t1"] == 3 and sp["n_t2"] == 3
    assert sp["n_t3p"] == 3
    assert sp["hit_t0"] == pytest.approx(2 / 3)
    assert sp["hit_t3p"] == 0.0
    assert sp["n_ge1_segments"] == 3
    assert sp["hit_ge1_envmean"] == pytest.approx(2 / 3)


# -- (d) NaN for an empty index, and the log line --------------------------------
def test_empty_indices_are_nan_and_log_line_prints_nan():
    # every env completes exactly one trial: indices >= 1 have no trials.
    sp = spout_eval_summary(_scripted([[[1]], [[0]], [[1]]]))
    assert sp["n_t0"] == 3 and sp["hit_t0"] == pytest.approx(2 / 3)
    assert (sp["n_t1"], sp["n_t2"], sp["n_t3p"], sp["n_ge1_segments"]) == (0, 0, 0, 0)
    for key in ("hit_t1", "hit_t2", "hit_t3p", "hit_ge1_envmean"):
        assert math.isnan(sp[key]), key
    line = spout_eval_log_line(1234, sp)
    assert "hit_t1=nan hit_t2=nan hit_t3p=nan hit_ge1_envmean=nan" in line
    assert "n_t0=3 n_t1=0 n_t2=0 n_t3p=0 n_ge1_segments=0" in line


def test_no_trials_at_all_is_all_nan_and_logs():
    sp = spout_eval_summary(_scripted([[[]], [[]]], tail=30))
    assert sp["n_t0"] == 0 and math.isnan(sp["hit_t0"])
    line = spout_eval_log_line(7, sp)
    assert "hit_t0=nan" in line and "n_t0=0" in line


def test_log_line_field_order_and_prefix():
    sp = spout_eval_summary(_scripted([[[1, 0, 1, 1]]]))
    line = spout_eval_log_line(5_000_000, sp)
    assert line.startswith("SPOUT_EVAL env_steps=5000000 episodes=1 trials_per_ep=")
    keys = [tok.split("=")[0] for tok in line.split()[1:]]
    assert keys == [
        "env_steps",
        "episodes",
        "trials_per_ep",
        "hit_rate",
        "hit_rate_after_first",
        "treats_per_ep",
        "timeout_frac",
        "fall_frac",
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
    ]


# -- (e) the legacy keys are unchanged ------------------------------------------
def test_legacy_keys_unchanged_on_fixed_input():
    sp = spout_eval_summary(_fixed_input())
    for key, want in LEGACY_GOLDEN.items():
        assert sp[key] == pytest.approx(want, rel=1e-12, abs=1e-12), key


def test_summary_key_set_is_legacy_plus_new():
    assert set(spout_eval_summary(_fixed_input())) == set(LEGACY_KEYS) | set(NEW_KEYS)


def test_new_keys_consistent_with_env_side_counters_on_fixed_input():
    # The env's own trial_after_first / hit_after_first (counted from its
    # trial_idx) must agree with the done-based index >= 1 population.
    allenv = _fixed_input()
    sp = spout_eval_summary(allenv)
    taf = float(allenv[SPOUT_ALIAS + "trial_after_first"].sum())
    haf = float(allenv[SPOUT_ALIAS + "hit_after_first"].sum())
    n_ge1 = sp["n_t1"] + sp["n_t2"] + sp["n_t3p"]
    assert n_ge1 == taf
    h_ge1 = sum(
        sp[f"hit_{k}"] * sp[f"n_{k}"] for k in ("t1", "t2", "t3p") if sp[f"n_{k}"]
    )
    assert h_ge1 == pytest.approx(haf)
    assert sp["n_t0"] + n_ge1 == float(allenv[SPOUT_ALIAS + "trial_end"].sum())


def _brute_force(done, te, hit):
    """Per-trial python loops: the definition, written out segment by segment."""
    t_len, n_envs = done.shape
    segments = []  # one list of (index, hit) per (env, segment)
    for n in range(n_envs):
        cur = []
        for t in range(t_len):
            if te[t, n]:
                cur.append((len(cur), float(hit[t, n])))
            if done[t, n]:  # last step of the episode: the next step opens a new one
                segments.append(cur)
                cur = []
        segments.append(cur)  # a trailing (possibly empty) segment is harmless
    trials = [tr for seg in segments for tr in seg]

    def mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    want = {}
    for name, ok in (
        ("t0", lambda i: i == 0),
        ("t1", lambda i: i == 1),
        ("t2", lambda i: i == 2),
        ("t3p", lambda i: i >= 3),
    ):
        sel = [h for i, h in trials if ok(i)]
        want[f"hit_{name}"], want[f"n_{name}"] = mean(sel), len(sel)
    fracs = [
        mean([h for i, h in seg if i >= 1])
        for seg in segments
        if any(i >= 1 for i, _ in seg)
    ]
    want["hit_ge1_envmean"], want["n_ge1_segments"] = mean(fracs), len(fracs)
    return want


@pytest.mark.parametrize("seed", range(8))
def test_matches_brute_force_on_random_windows(seed):
    rng = np.random.default_rng(seed)
    t_len, n_envs = int(rng.integers(30, 120)), int(rng.integers(1, 9))
    te = rng.random((t_len, n_envs)) < rng.uniform(0.05, 0.4)
    hit = rng.random((t_len, n_envs)) < 0.5
    done = rng.random((t_len, n_envs)) < rng.uniform(0.0, 0.15)
    done[-1, : n_envs // 2] = True  # some envs end on a done at the last step
    sp = spout_eval_summary(_allenv(done, te, hit))
    want = _brute_force(done, te, hit & te)
    for key, value in want.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(sp[key]), (seed, key)
        else:
            assert sp[key] == pytest.approx(value), (seed, key)


# -- the real state machine ------------------------------------------------------
@pytest.mark.parametrize("seed", range(3))
def test_boundaries_agree_with_the_real_trial_state_machine(seed):
    """Drive ``spout_logic.trial_update`` for many envs with random terminations
    and the env's reset convention (the done step's events belong to the old
    episode; the next step starts from a fresh ``trial_idx = 0`` state), then
    require the done-based index >= 1 population to equal the env's own
    ``trial_after_first`` / ``hit_after_first`` counters."""
    import jax
    import jax.numpy as jnp

    from vnl_playground.tasks.rodent.spout_logic import (
        SpoutParams,
        init_trial_state,
        trial_update,
    )

    p = SpoutParams(
        n_arms=2,
        pay_prob=0.7,
        appear_distance=0.5,
        spout_radius=0.15,
        reach=0.1,
        dwell_steps=3,
        trial_timeout_steps=12,
    )
    t_len, n_envs = 300, 16
    key = jax.random.PRNGKey(seed)
    k_init, k_scan = jax.random.split(key)
    ts0 = jax.vmap(lambda k: init_trial_state(k, p))(jax.random.split(k_init, n_envs))

    def body(carry, k):
        (ts,) = carry
        k_arm, k_dist, k_tr, k_pay, k_done, k_reset = jax.random.split(k, 6)
        arm = jax.random.randint(k_arm, (n_envs,), -1, 2)
        dist = jax.random.uniform(k_dist, (n_envs,), minval=-0.1, maxval=0.8)
        treat = jax.random.uniform(k_tr, (n_envs,), minval=0.0, maxval=0.3)
        new_ts, ev = jax.vmap(lambda t, a, d, td, r: trial_update(t, a, d, td, r, p))(
            ts, arm, dist, treat, jax.random.split(k_pay, n_envs)
        )
        done = jax.random.uniform(k_done, (n_envs,)) < 0.03
        fresh = jax.vmap(lambda r: init_trial_state(r, p))(
            jax.random.split(k_reset, n_envs)
        )
        nxt = jax.tree.map(
            lambda f, s: jnp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), f, s),
            fresh,
            new_ts,
        )
        return (nxt,), (ev, done)

    _, (ev, done) = jax.lax.scan(body, (ts0,), jax.random.split(k_scan, t_len))
    allenv = {"done": np.asarray(done, np.float32)}
    for name in ("trial_end", "hit", "trial_after_first", "hit_after_first", "timeout"):
        allenv[SPOUT_ALIAS + name] = np.asarray(ev[name])
    sp = spout_eval_summary(allenv)
    assert sp["n_t0"] > 20 and sp["n_t3p"] > 0 and sp["n_ge1_segments"] > 10
    assert sp["n_t1"] + sp["n_t2"] + sp["n_t3p"] == int(
        allenv[SPOUT_ALIAS + "trial_after_first"].sum()
    )
    h_ge1 = sum(
        sp[f"hit_{k}"] * sp[f"n_{k}"] for k in ("t1", "t2", "t3p") if sp[f"n_{k}"]
    )
    assert h_ge1 == pytest.approx(float(allenv[SPOUT_ALIAS + "hit_after_first"].sum()))


def test_warns_when_boundaries_disagree_with_the_env_counter(caplog):
    # an env-side counter that does not line up with the `done` boundaries
    # (here: trial_after_first zeroed) must be flagged, not silently trusted.
    allenv = _scripted([[[1, 0, 1, 1]]])
    allenv[SPOUT_ALIAS + "trial_after_first"] = np.zeros_like(
        allenv[SPOUT_ALIAS + "trial_after_first"]
    )
    with caplog.at_level("WARNING"):
        spout_eval_summary(allenv)
    assert any("do not line up" in r.getMessage() for r in caplog.records)


def test_no_warning_when_consistent(caplog):
    with caplog.at_level("WARNING"):
        spout_eval_summary(_fixed_input())
    assert not [r for r in caplog.records if "do not line up" in r.getMessage()]


class _StubEnv:
    """Minimal env: records the metrics it is stepped with, emits ``spout/*``."""

    def __init__(self):
        self.stepped_with = None

    def _state(self, metrics):
        from mujoco_playground._src import mjx_env

        return mjx_env.State(
            data=None, obs={}, reward=0.0, done=0.0, metrics=metrics, info={}
        )

    def reset(self, rng):
        return self._state({"spout/hit": 1.0, "spout/trial_end": 0.0, "other": 5.0})

    def step(self, state, action):
        self.stepped_with = dict(state.metrics)
        return self._state({"spout/hit": 0.0, "spout/trial_end": 1.0, "other": 6.0})


def test_alias_wrapper_mirrors_spout_metrics_and_strips_them_on_the_way_down():
    inner = _StubEnv()
    env = SpoutEvalMetricAlias(inner, ("hit", "trial_end"))

    st = env.reset(None)
    assert st.metrics[SPOUT_ALIAS + "hit"] == 1.0
    assert st.metrics[SPOUT_ALIAS + "trial_end"] == 0.0
    assert st.metrics["spout/hit"] == 1.0 and st.metrics["other"] == 5.0

    st = env.step(st, None)
    # The wrapped env never sees the mirrored keys ...
    assert not [k for k in inner.stepped_with if k.startswith(SPOUT_ALIAS)]
    assert set(inner.stepped_with) == {"spout/hit", "spout/trial_end", "other"}
    # ... and the step output carries the fresh mirror.
    assert st.metrics[SPOUT_ALIAS + "hit"] == 0.0
    assert st.metrics[SPOUT_ALIAS + "trial_end"] == 1.0
