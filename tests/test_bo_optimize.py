"""Unit tests for scripts/bo_optimize.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# Load bo_optimize as a module even though scripts/ is not a package.
_SPEC = importlib.util.spec_from_file_location(
    "bo_optimize",
    Path(__file__).resolve().parents[1] / "scripts" / "bo_optimize.py",
)
bo = importlib.util.module_from_spec(_SPEC)
sys.modules["bo_optimize"] = bo
_SPEC.loader.exec_module(bo)


def test_load_warmstart_filters_and_maps():
    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    trials = bo.load_warmstart(fixture)

    names = {t.user_attrs["source_name"] for t in trials}
    assert names == {"keep-ms-a", "keep-ms-b", "keep-out-of-bounds"}


def test_load_warmstart_maps_axis_values():
    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    trials = bo.load_warmstart(fixture)
    by_name = {t.user_attrs["source_name"]: t for t in trials}

    a = by_name["keep-ms-a"]
    assert a.params["fs"] == pytest.approx(1.0)
    assert a.params["damp"] == pytest.approx(5e-7)
    assert a.params["cc"] == pytest.approx(0.025)
    assert a.params["cdc"] == pytest.approx(0.025)
    assert a.params["qvel_init"] == "zeros"
    assert a.values == [pytest.approx(0.57), pytest.approx(0.57),
                        pytest.approx(0.13), pytest.approx(0.14),
                        pytest.approx(0.20), pytest.approx(0.19)]
    assert a.user_attrs["R"] == pytest.approx(396.0)
    assert a.user_attrs["constraint"] == [pytest.approx(380.0 - 396.0)]


def test_load_warmstart_includes_out_of_bounds_fs():
    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    trials = bo.load_warmstart(fixture)
    by_name = {t.user_attrs["source_name"]: t for t in trials}
    oob = by_name["keep-out-of-bounds"]
    assert oob.params["fs"] == pytest.approx(0.3)


def test_make_study_creates_multi_objective_with_constraints(tmp_path):
    journal = tmp_path / "bo_study.log"
    study = bo.make_study("test-study", journal)
    assert len(study.directions) == 6
    assert study.directions[0].name == "MAXIMIZE"
    assert study.directions[1].name == "MAXIMIZE"
    assert study.directions[2].name == "MINIMIZE"
    assert study.directions[3].name == "MINIMIZE"
    assert study.directions[4].name == "MINIMIZE"
    assert study.directions[5].name == "MINIMIZE"
    assert isinstance(study.sampler, bo.NSGAIISampler)


def test_make_study_resumes_from_journal(tmp_path):
    journal = tmp_path / "bo_study.log"
    s1 = bo.make_study("resume-test", journal)
    trial = s1.ask(bo.SEARCH_SPACE_DISTRIBUTIONS)
    trial.set_user_attr("constraint", [-5.0])
    s1.tell(trial, [0.5, 0.5, 0.1, 0.1, 0.2, 0.2])

    s2 = bo.make_study("resume-test", journal)
    assert len(s2.trials) == 1


def test_launch_training_builds_correct_cli(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(bo.subprocess, "run", fake_run)

    params = {
        "fs": 0.95,
        "damp": 7e-7,
        "cc": 0.03,
        "cdc": 0.02,
        "qvel_init": "zeros",
    }
    rc = bo.launch_training(params, "trial-0003", tmp_path)

    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[0].endswith("python") or cmd[0] == sys.executable
    assert bo.TRAINING_SCRIPT in cmd
    assert "--force-scale" in cmd and cmd[cmd.index("--force-scale") + 1] == "0.95"
    assert "--joint-damping" in cmd and cmd[cmd.index("--joint-damping") + 1] == "7e-07"
    assert "--control-cost" in cmd and cmd[cmd.index("--control-cost") + 1] == "0.03"
    assert "--control-diff-cost" in cmd and cmd[cmd.index("--control-diff-cost") + 1] == "0.02"
    assert "--qvel-init" in cmd and cmd[cmd.index("--qvel-init") + 1] == "zeros"
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "1"
    # tags are nargs="*" -- bo-s13 first, then trial tag
    tag_idx = cmd.index("--wandb-tags")
    assert cmd[tag_idx + 1] == "bo-s13"
    assert cmd[tag_idx + 2] == "trial-0003"
    # log file exists
    assert (tmp_path / "trial-0003.log").exists()


def test_launch_training_returns_nonzero_on_failure(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        class R:
            returncode = 7
        return R()
    monkeypatch.setattr(bo.subprocess, "run", fake_run)
    rc = bo.launch_training(
        {"fs": 1.0, "damp": 5e-7, "cc": 0.025, "cdc": 0.025, "qvel_init": "zeros"},
        "trial-0099", tmp_path,
    )
    assert rc == 7


class _FakeRun:
    def __init__(self, summary):
        self.summary = summary


class _FakeApi:
    def __init__(self, runs_by_tag):
        self._runs_by_tag = runs_by_tag
        self.call_count = 0

    def runs(self, path, filters):
        self.call_count += 1
        tag = filters["tags"]["$in"][0]
        return list(self._runs_by_tag.get(tag, []))


def test_read_metrics_returns_dict_on_success(monkeypatch):
    summary = {
        "eval/episode_reward": 405.0,
        "eval/emg_biceps_corr": 0.71,
        "eval/emg_triceps_corr": 0.62,
        "eval/emg_biceps_mae": 0.12,
        "eval/emg_triceps_mae": 0.14,
        "eval/emg_biceps_trial_mae": 0.19,
        "eval/emg_triceps_trial_mae": 0.20,
    }
    api = _FakeApi({"trial-0001": [_FakeRun(summary)]})
    monkeypatch.setattr(bo, "_wandb_api", lambda: api)

    out = bo.read_metrics("trial-0001", retries=1, backoff_s=0.0)
    assert out == {
        "R": 405.0,
        "bcorr": 0.71,
        "tcorr": 0.62,
        "bmae": 0.12,
        "tmae": 0.14,
        "btrial": 0.19,
        "ttrial": 0.20,
    }


def test_read_metrics_retries_then_returns_none(monkeypatch):
    api = _FakeApi({})  # no runs at all
    monkeypatch.setattr(bo, "_wandb_api", lambda: api)

    out = bo.read_metrics("trial-9999", retries=3, backoff_s=0.0)
    assert out is None
    assert api.call_count == 3


def test_read_metrics_nan_returns_none(monkeypatch):
    summary = {
        "eval/episode_reward": float("nan"),
        "eval/emg_biceps_corr": 0.5,
        "eval/emg_triceps_corr": 0.5,
        "eval/emg_biceps_mae": 0.2,
        "eval/emg_triceps_mae": 0.2,
        "eval/emg_biceps_trial_mae": 0.25,
        "eval/emg_triceps_trial_mae": 0.25,
    }
    api = _FakeApi({"trial-0002": [_FakeRun(summary)]})
    monkeypatch.setattr(bo, "_wandb_api", lambda: api)

    out = bo.read_metrics("trial-0002", retries=1, backoff_s=0.0)
    assert out is None


def test_report_frontier_filters_below_400(tmp_path):
    journal = tmp_path / "j.log"
    study = bo.make_study("frontier-test", journal)

    # Trial A: feasible, R=410 -> should appear
    tA = study.ask(bo.SEARCH_SPACE_DISTRIBUTIONS)
    tA.set_user_attr("constraint", [380.0 - 410.0])
    tA.set_user_attr("R", 410.0)
    study.tell(tA, [0.7, 0.6, 0.12, 0.13, 0.18, 0.19])

    # Trial B: feasible under BO but R<400 post-hoc -> dropped from winner CSV
    tB = study.ask(bo.SEARCH_SPACE_DISTRIBUTIONS)
    tB.set_user_attr("constraint", [380.0 - 390.0])
    tB.set_user_attr("R", 390.0)
    study.tell(tB, [0.8, 0.7, 0.10, 0.11, 0.16, 0.17])

    # Trial C: infeasible -> dropped
    tC = study.ask(bo.SEARCH_SPACE_DISTRIBUTIONS)
    tC.set_user_attr("constraint", [380.0 - 350.0])
    tC.set_user_attr("R", 350.0)
    study.tell(tC, [0.9, 0.8, 0.08, 0.09, 0.14, 0.15])

    out = tmp_path / "frontier.csv"
    bo.report_frontier(study, out)

    import csv
    with open(out) as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 1
    assert float(rows[0]["R"]) == pytest.approx(410.0)
