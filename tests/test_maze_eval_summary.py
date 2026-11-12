"""Unit tests for ``maze_eval_summary`` / ``maze_eval_log_line`` (CPU, no GPU).

``maze_eval_summary`` consumes the all-env ``[T, N]`` eval arrays (``allenv``)
that ``run_eval_rollout_envzero`` returns: ``done``, the carried
``rewards/*`` / ``terminations/*`` env metrics and, when ``maze_critic_priv``
is on, the visited-cell count under ``anchor/_maze_eval/visited_cells`` (the
eval shim's mirror of ``maze/visited_cells``; the plain ``maze/visited_cells``
key is also read, for when the rollout carries ``maze/`` itself).

Episode-1 convention (pinned here, the same one ``spout_eval_summary`` uses):
a ``done`` at step ``t`` is the LAST step of that episode -- the env auto-resets
inside the same step, so the ``rewards/*`` / ``terminations/*`` /
``maze/visited_cells`` values of the done step belong to the OLD episode.
"Episode 1" of an env is steps ``0 .. t1`` for its first done ``t1``, or the
whole window when it never ends.

The builder below is independent of the implementation under test.
"""

import re

import numpy as np
import pytest

from vnl_playground.tasks.rodent.maze_eval import (
    FALL_TERMINATIONS,
    MAZE_ALIAS,
    MazeEvalMetricAlias,
    maze_eval_log_line,
    maze_eval_summary,
)

DT = 0.01
FIELDS = (
    "envs",
    "treats_ep1",
    "treats_ep1_sd",
    "len_ep1_s",
    "fall_frac",
    "alltreats_frac",
    "timeout_frac",
    "visited_ep1",
)


def _allenv(t_len, scripts, visited=True, terms=True):
    """``allenv`` from per-env scripts.

    Each script is a dict with
      ``treats``: steps whose ``rewards/treat_collected`` is paid (weight 1);
      ``ends``: ``[(step, cause), ...]`` -- a ``done`` is written at ``step``
        with cause ``"belly_up"`` / ``"torso_too_low"`` / ``"fallen"`` /
        ``"all_treats"`` / ``"truncation"`` (done with no termination flag,
        EpisodeWrapper's); the ``terminations/fallen`` key exists only when a
        script uses that cause (the default config does not carry it);
      ``visited``: callable ``t -> visited-cell count`` of the step.
    """
    n = len(scripts)
    out = {
        "done": np.zeros((t_len, n), np.float32),
        "reward": np.zeros((t_len, n), np.float32),
        "rewards/treat_collected": np.zeros((t_len, n), np.float32),
    }
    if terms:
        for key in ("belly_up", "torso_too_low", "all_treats_collected", "any"):
            out[f"terminations/{key}"] = np.zeros((t_len, n), np.float32)
    if visited:
        out[MAZE_ALIAS + "visited_cells"] = np.zeros((t_len, n), np.float32)
    if terms and any(c == "fallen" for sc in scripts for _, c in sc.get("ends", ())):
        out["terminations/fallen"] = np.zeros((t_len, n), np.float32)
    for e, sc in enumerate(scripts):
        for t in sc.get("treats", ()):
            out["rewards/treat_collected"][t, e] = 1.0
        for step, cause in sc.get("ends", ()):
            out["done"][step, e] = 1.0
            if terms and cause != "truncation":
                flag = {
                    "belly_up": "belly_up",
                    "torso_too_low": "torso_too_low",
                    "fallen": "fallen",
                    "all_treats": "all_treats_collected",
                }[cause]
                out[f"terminations/{flag}"][step, e] = 1.0
                out["terminations/any"][step, e] = 1.0
        if visited:
            vis = sc.get("visited", lambda t: 0.0)
            out[MAZE_ALIAS + "visited_cells"][:, e] = [vis(t) for t in range(t_len)]
    return out


def _five_envs():
    """Known treat counts and termination causes, T = 100.

    env 0  3 treats,  belly_up at step 49      -> len 50, fall
    env 1  20 treats, all_treats at step 79    -> len 80, all-treats
    env 2  3 treats,  never ends               -> len 100 (window end), timeout
    env 3  2 treats,  truncation at step 59    -> len 60, timeout (time limit)
    env 4  0 treats,  torso_too_low at step 29 -> len 30, fall
    The visited count of env ``e`` at step ``t`` is ``t + 1 + 1000 * e``, so a
    read at the wrong step or the wrong env is visible in the mean.
    """

    def vis(e):
        return lambda t: float(t + 1 + 1000 * e)

    return [
        dict(treats=[10, 20, 30], ends=[(49, "belly_up")], visited=vis(0)),
        dict(treats=list(range(0, 80, 4)), ends=[(79, "all_treats")], visited=vis(1)),
        dict(treats=[5, 50, 95], ends=[], visited=vis(2)),
        dict(treats=[1, 2], ends=[(59, "truncation")], visited=vis(3)),
        dict(treats=[], ends=[(29, "torso_too_low")], visited=vis(4)),
    ]


def test_known_counts_and_causes_give_exact_means_and_fractions():
    allenv = _allenv(100, _five_envs())
    sm = maze_eval_summary(allenv, DT)
    treats = np.array([3, 20, 3, 2, 0], np.float64)
    lens = np.array([50, 80, 100, 60, 30], np.float64)
    assert set(sm) == set(FIELDS)
    assert sm["envs"] == 5
    assert sm["treats_ep1"] == pytest.approx(treats.mean())
    assert sm["treats_ep1_sd"] == pytest.approx(treats.std())  # population sd over envs
    assert sm["len_ep1_s"] == pytest.approx(lens.mean() * DT)
    assert sm["fall_frac"] == pytest.approx(2 / 5)  # envs 0 and 4
    assert sm["alltreats_frac"] == pytest.approx(1 / 5)  # env 1
    assert sm["timeout_frac"] == pytest.approx(
        2 / 5
    )  # envs 2 (window end) and 3 (truncation)
    # visited cells at the LAST step of episode 1 (step last, value last + 1 + 1000 e).
    expected = np.mean([lens[e] + 1000 * e for e in range(5)])
    assert sm["visited_ep1"] == pytest.approx(expected)


def test_only_episode_one_counts():
    """An env with two episodes in the window contributes its first one only."""

    def vis(t):  # episode 1 climbs to 40 by its last step (39); episode 2 restarts at 1
        return float(t + 1) if t < 40 else float(t - 39)

    sc = [
        dict(
            treats=[3, 9, 45, 46, 47, 60, 70],  # 2 before the first done, 5 after
            ends=[(39, "belly_up"), (79, "all_treats")],
            visited=vis,
        ),
        dict(treats=[1], ends=[], visited=lambda t: 7.0),
    ]
    sm = maze_eval_summary(_allenv(120, sc), DT)
    assert sm["treats_ep1"] == pytest.approx((2 + 1) / 2)
    assert sm["len_ep1_s"] == pytest.approx((40 + 120) / 2 * DT)
    assert sm["fall_frac"] == pytest.approx(1 / 2)  # not the second done's cause
    assert sm["alltreats_frac"] == pytest.approx(0.0)  # episode 2's cause is ignored
    assert sm["timeout_frac"] == pytest.approx(1 / 2)  # env 1 never ends
    assert sm["visited_ep1"] == pytest.approx((40 + 7) / 2)


def test_fallen_termination_counts_as_a_fall_when_the_env_emits_it():
    """A config using the ``fallen`` criterion: its falls are in ``fall_frac`` too, so the
    three fractions still sum to 1."""
    sc = [
        dict(treats=[1], ends=[(20, "fallen")]),
        dict(treats=[], ends=[(30, "belly_up")]),
        dict(treats=[2], ends=[(40, "all_treats")]),
        dict(treats=[], ends=[]),
    ]
    sm = maze_eval_summary(_allenv(60, sc), DT)
    assert sm["fall_frac"] == pytest.approx(2 / 4)
    assert sm["alltreats_frac"] == pytest.approx(1 / 4)
    assert sm["timeout_frac"] == pytest.approx(1 / 4)
    assert sm["fall_frac"] + sm["alltreats_frac"] + sm["timeout_frac"] == pytest.approx(
        1.0
    )


def test_done_on_the_last_step_of_the_window_is_a_real_ending():
    sc = [dict(treats=[4], ends=[(19, "all_treats")], visited=lambda t: 3.0)]
    sm = maze_eval_summary(_allenv(20, sc), DT)
    assert sm["alltreats_frac"] == 1.0 and sm["timeout_frac"] == 0.0
    assert sm["len_ep1_s"] == pytest.approx(20 * DT)


def test_absent_visited_metric_gives_nan_and_the_formatter_prints_nan():
    allenv = _allenv(100, _five_envs(), visited=False)
    sm = maze_eval_summary(allenv, DT)
    assert np.isnan(sm["visited_ep1"])
    # The other fields are unaffected by the absence.
    assert sm["treats_ep1"] == pytest.approx(np.mean([3, 20, 3, 2, 0]))
    line = maze_eval_log_line(1234, sm)
    assert "visited_ep1=nan" in line
    assert "nan" not in line.replace("visited_ep1=nan", "")


def test_plain_maze_visited_key_is_read_when_present():
    allenv = _allenv(100, _five_envs(), visited=False)
    allenv["maze/visited_cells"] = np.full((100, 5), 9.0, np.float32)
    assert maze_eval_summary(allenv, DT)["visited_ep1"] == pytest.approx(9.0)


def test_absent_termination_signals_give_nan_not_a_guess():
    allenv = _allenv(100, _five_envs(), terms=False)
    sm = maze_eval_summary(allenv, DT)
    for key in ("fall_frac", "alltreats_frac", "timeout_frac"):
        assert np.isnan(sm[key]), key
    assert sm["treats_ep1"] == pytest.approx(np.mean([3, 20, 3, 2, 0]))
    line = maze_eval_log_line(5, sm)
    assert (
        "fall_frac=nan" in line
        and "alltreats_frac=nan" in line
        and "timeout_frac=nan" in line
    )


def test_absent_treat_reward_gives_nan_treats():
    allenv = _allenv(100, _five_envs())
    del allenv["rewards/treat_collected"]
    sm = maze_eval_summary(allenv, DT)
    assert np.isnan(sm["treats_ep1"]) and np.isnan(sm["treats_ep1_sd"])
    assert sm["len_ep1_s"] == pytest.approx(np.mean([50, 80, 100, 60, 30]) * DT)


def test_treat_count_is_weight_independent():
    allenv = _allenv(100, _five_envs())
    allenv["rewards/treat_collected"] *= (
        2.5  # a reward weight of 2.5 pays 2.5 per treat
    )
    assert maze_eval_summary(allenv, DT)["treats_ep1"] == pytest.approx(5.6)


def test_len_scales_with_the_step_duration():
    allenv = _allenv(100, _five_envs())
    a = maze_eval_summary(allenv, 0.01)["len_ep1_s"]
    b = maze_eval_summary(allenv, 0.02)["len_ep1_s"]
    assert b == pytest.approx(2 * a)


def test_log_line_format_is_exact():
    sm = maze_eval_summary(_allenv(100, _five_envs()), DT)
    line = maze_eval_log_line(1_228_800, sm)
    assert line.startswith("MAZE_EVAL env_steps=1228800 envs=5 ")
    pattern = (
        r"^MAZE_EVAL env_steps=\d+ envs=\d+ treats_ep1=\S+ treats_ep1_sd=\S+ "
        r"len_ep1_s=\S+ fall_frac=\S+ alltreats_frac=\S+ timeout_frac=\S+ "
        r"visited_ep1=\S+$"
    )
    assert re.match(pattern, line), line
    assert "treats_ep1=5.6 " in line and "len_ep1_s=0.64 " in line
    assert "fall_frac=0.4 " in line and "alltreats_frac=0.2 " in line


def test_accepts_jax_arrays():
    import jax.numpy as jnp

    allenv = {k: jnp.asarray(v) for k, v in _allenv(100, _five_envs()).items()}
    sm = maze_eval_summary(allenv, DT)
    assert sm["treats_ep1"] == pytest.approx(5.6)


def test_fall_terminations_are_the_posture_criteria():
    assert FALL_TERMINATIONS == (
        "terminations/belly_up",
        "terminations/torso_too_low",
        "terminations/fallen",
    )


class _StubEnv:
    """Minimal env: records the metrics it is stepped with, emits ``maze/*``."""

    def __init__(self):
        self.stepped_with = None

    def _state(self, metrics):
        from mujoco_playground._src import mjx_env

        return mjx_env.State(
            data=None, obs={}, reward=0.0, done=0.0, metrics=metrics, info={}
        )

    def reset(self, rng):
        return self._state({"maze/visited_cells": 1.0, "other": 5.0})

    def step(self, state, action):
        self.stepped_with = dict(state.metrics)
        return self._state({"maze/visited_cells": 2.0, "other": 6.0})


def test_alias_wrapper_mirrors_maze_metrics_and_strips_them_on_the_way_down():
    inner = _StubEnv()
    env = MazeEvalMetricAlias(inner, ("visited_cells",))

    st = env.reset(None)
    assert st.metrics[MAZE_ALIAS + "visited_cells"] == 1.0
    assert st.metrics["maze/visited_cells"] == 1.0 and st.metrics["other"] == 5.0

    st = env.step(st, None)
    # The wrapped env never sees the mirrored keys ...
    assert not [k for k in inner.stepped_with if k.startswith(MAZE_ALIAS)]
    assert set(inner.stepped_with) == {"maze/visited_cells", "other"}
    # ... and the step output carries the fresh mirror.
    assert st.metrics[MAZE_ALIAS + "visited_cells"] == 2.0
