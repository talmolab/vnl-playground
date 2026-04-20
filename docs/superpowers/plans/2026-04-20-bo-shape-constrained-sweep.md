# BO-driven shape-constrained sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/bo_optimize.py`, a single-process Optuna NSGA-II driver that warm-starts from the s10+s11 CSV, runs 40 serial training trials on one GPU over 6 objectives (shape correlations + timestep MAE + trial MAE for both muscles) under a reward floor constraint, and reports a Pareto frontier filtered to `eval/episode_reward ≥ 400`.

**Architecture:** One Python file with five pure-ish functions (`load_warmstart`, `make_study`, `launch_training`, `read_metrics`, `report_frontier`) plus a `main()` loop. Unit tests mock wandb and subprocess; integration tests run against the real CSV and a 1-trial smoke run.

**Tech Stack:** Python 3, Optuna (NSGAIISampler + JournalFileStorage), pandas, wandb Python API, `subprocess`. Venv: `/root/vast/eric/track-mjx/.venv`.

**Spec:** `docs/superpowers/specs/2026-04-20-bo-shape-constrained-sweep-design.md`

---

## Global commands

All tests and runs use the shared venv:

```bash
source /root/vast/eric/track-mjx/.venv/bin/activate
```

All steps below assume `cwd = /root/vast/eric/vnl-playground` and the venv is already activated.

Run tests with:

```bash
pytest tests/test_bo_optimize.py -v
```

---

## Task 1: Setup — install Optuna, create skeleton

**Files:**
- Create: `scripts/bo_optimize.py`
- Create: `tests/__init__.py`
- Create: `tests/test_bo_optimize.py`
- Create: `tests/fixtures/bo_warmstart_sample.csv`

- [ ] **Step 1: Install Optuna into the shared venv**

Run:
```bash
source /root/vast/eric/track-mjx/.venv/bin/activate && pip install optuna
```

Expected: installs Optuna ≥ 3.x. Confirm with:
```bash
python -c "import optuna; print(optuna.__version__)"
```

- [ ] **Step 2: Create skeleton `scripts/bo_optimize.py`**

Write:
```python
"""Bayesian optimization driver for moving-shoulder training.

See docs/superpowers/specs/2026-04-20-bo-shape-constrained-sweep-design.md.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import optuna
import pandas as pd
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
)
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.samplers import NSGAIISampler
from optuna.trial import FrozenTrial, TrialState


OBJECTIVE_DIRECTIONS = ["maximize", "maximize", "minimize", "minimize", "minimize", "minimize"]
OBJECTIVE_KEYS = (
    "eval/emg_biceps_corr",
    "eval/emg_triceps_corr",
    "eval/emg_biceps_mae",
    "eval/emg_triceps_mae",
    "eval/emg_biceps_trial_mae",
    "eval/emg_triceps_trial_mae",
)
REWARD_KEY = "eval/episode_reward"
CONSTRAINT_REWARD_FLOOR = 380.0
WINNER_REWARD_FLOOR = 400.0
WANDB_PROJECT = "vnl-mjx-rl"
TRAINING_SCRIPT = "train_mouse_janelia_sigmoid_moving_shoulder.py"


def load_warmstart(csv_path: Path) -> list[FrozenTrial]:
    raise NotImplementedError


def make_study(study_name: str, journal_path: Path) -> optuna.Study:
    raise NotImplementedError


def launch_training(params: dict, tag: str, log_dir: Path) -> int:
    raise NotImplementedError


def read_metrics(tag: str, retries: int = 3, backoff_s: float = 30.0) -> Optional[dict]:
    raise NotImplementedError


def report_frontier(study: optuna.Study, out_csv: Path) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `tests/__init__.py` (empty file)**

Write:
```python
```
(empty)

- [ ] **Step 4: Create `tests/test_bo_optimize.py` skeleton**

Write:
```python
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
```

- [ ] **Step 5: Create `tests/fixtures/bo_warmstart_sample.csv` fixture**

A hand-crafted 7-row CSV mirroring the relevant columns from `s11_ms_s10_ms_final.csv`. Covers: 2 keep-rows (moving-shoulder, finished, default tau, non-NaN evals), 1 tau-extra row to drop, 1 NaN-eval row to drop, 1 non-moving-shoulder row to drop, 1 crashed row to drop, 1 out-of-bounds fs value to keep (warm-start loader does NOT filter on bounds).

Write:
```csv
Name,State,Tags,_fields.reference_data_path,_fields.force_scale,_fields.joint_damping,_fields.control_cost,_fields.control_diff_cost,_fields.qvel_init,eval/episode_reward,eval/emg_biceps_corr,eval/emg_triceps_corr,eval/emg_biceps_mae,eval/emg_triceps_mae,eval/emg_biceps_trial_mae,eval/emg_triceps_trial_mae
keep-ms-a,finished,"s11-ms, moving-shoulder","/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder",1.0,5e-7,0.025,0.025,zeros,396,0.57,0.57,0.13,0.14,0.20,0.19
keep-ms-b,finished,"s10-ms, moving-shoulder","/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder",0.7,5e-7,0.05,0.1,zeros,344,0.90,0.79,0.18,0.21,0.22,0.23
drop-tau-extra,finished,"s11-ms, moving-shoulder, tau-extra","/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder",1.0,3e-7,0.025,0.025,zeros,350,0.40,0.30,0.18,0.18,0.22,0.22
drop-nan-eval,finished,"s11-ms, moving-shoulder","/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder",1.0,5e-7,0.025,0.025,zeros,,,,,,,
drop-non-ms,finished,"s11, arm-only","/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data",1.0,5e-7,0.025,0.025,zeros,380,0.50,0.50,0.15,0.15,0.20,0.20
drop-crashed,crashed,"s11-ms, moving-shoulder","/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder",1.0,5e-7,0.025,0.025,zeros,,,,,,,
keep-out-of-bounds,finished,"s10-ms, moving-shoulder","/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder",0.3,2e-7,0.05,0.1,zeros,200,0.30,0.30,0.25,0.25,0.28,0.28
```

- [ ] **Step 6: Verify the skeleton imports cleanly**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: 0 tests collected, no import errors. ("no tests ran" is the desired output.)

- [ ] **Step 7: Commit**

```bash
git add scripts/bo_optimize.py tests/__init__.py tests/test_bo_optimize.py tests/fixtures/bo_warmstart_sample.csv
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "add scaffold for BO driver + tests"
```

---

## Task 2: `load_warmstart` — filter CSV and emit FrozenTrials

**Files:**
- Modify: `scripts/bo_optimize.py` (implement `load_warmstart`)
- Modify: `tests/test_bo_optimize.py` (add tests)

Filter rules (from spec):
- `State == "finished"`
- `Tags` contains `"moving-shoulder"`
- `Tags` does NOT contain `"tau-extra"`
- no NaN in any of the five eval metrics
- `_fields.reference_data_path` ends in `reference_data_moving_shoulder`
- Maps to FrozenTrial with `state=COMPLETE`, `values=[bcorr, tcorr, bmae, tmae, btrial_mae, ttrial_mae]`, `user_attrs={"R": reward, "constraint": [380.0 - reward]}`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_bo_optimize.py`:
```python
def test_load_warmstart_filters_and_maps(tmp_path):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: 3 FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `load_warmstart`**

Replace the `load_warmstart` stub in `scripts/bo_optimize.py` with:
```python
SEARCH_SPACE_DISTRIBUTIONS = {
    "fs": FloatDistribution(0.5, 1.0),
    "damp": FloatDistribution(1e-7, 1.5e-6, log=True),
    "cc": FloatDistribution(0.0, 0.1),
    "cdc": FloatDistribution(0.0, 0.1),
    "qvel_init": CategoricalDistribution(["zeros", "reference"]),
}

_MS_REF_SUFFIX = "reference_data_moving_shoulder"


def _row_is_moving_shoulder_and_standard(row: pd.Series) -> bool:
    tags = str(row.get("Tags", ""))
    if "moving-shoulder" not in tags:
        return False
    if "tau-extra" in tags:
        return False
    if row.get("State") != "finished":
        return False
    ref = str(row.get("_fields.reference_data_path", ""))
    if not ref.endswith(_MS_REF_SUFFIX):
        return False
    return True


def load_warmstart(csv_path: Path) -> list[FrozenTrial]:
    df = pd.read_csv(csv_path)
    trials: list[FrozenTrial] = []
    for i, row in df.iterrows():
        if not _row_is_moving_shoulder_and_standard(row):
            continue
        try:
            fs = float(row["_fields.force_scale"])
            damp = float(row["_fields.joint_damping"])
            cc = float(row["_fields.control_cost"])
            cdc = float(row["_fields.control_diff_cost"])
            qvel = str(row["_fields.qvel_init"])
            R = float(row["eval/episode_reward"])
            bcorr = float(row["eval/emg_biceps_corr"])
            tcorr = float(row["eval/emg_triceps_corr"])
            bmae = float(row["eval/emg_biceps_mae"])
            tmae = float(row["eval/emg_triceps_mae"])
            btrial = float(row["eval/emg_biceps_trial_mae"])
            ttrial = float(row["eval/emg_triceps_trial_mae"])
        except (ValueError, TypeError):
            continue
        if any(math.isnan(v) for v in (fs, damp, cc, cdc, R, bcorr, tcorr, bmae, tmae, btrial, ttrial)):
            continue
        if qvel not in ("zeros", "reference"):
            continue
        params = {"fs": fs, "damp": damp, "cc": cc, "cdc": cdc, "qvel_init": qvel}
        trials.append(
            optuna.trial.create_trial(
                params=params,
                distributions=SEARCH_SPACE_DISTRIBUTIONS,
                values=[bcorr, tcorr, bmae, tmae, btrial, ttrial],
                user_attrs={
                    "R": R,
                    "constraint": [CONSTRAINT_REWARD_FLOOR - R],
                    "source_name": str(row.get("Name", f"row-{i}")),
                },
            )
        )
    return trials
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bo_optimize.py tests/test_bo_optimize.py
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "implement load_warmstart with CSV filtering"
```

---

## Task 3: `make_study` — Optuna NSGA-II + journal storage + constraints

**Files:**
- Modify: `scripts/bo_optimize.py` (implement `make_study`)
- Modify: `tests/test_bo_optimize.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_bo_optimize.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_bo_optimize.py::test_make_study_creates_multi_objective_with_constraints tests/test_bo_optimize.py::test_make_study_resumes_from_journal -v
```

Expected: 2 FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `make_study`**

Replace the `make_study` stub:
```python
def _constraints_func(trial: FrozenTrial) -> list[float]:
    c = trial.user_attrs.get("constraint")
    if c is None:
        # Unknown feasibility -> treat as feasible; NSGA-II will still use objectives.
        return [-1.0]
    return list(c)


def make_study(study_name: str, journal_path: Path) -> optuna.Study:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    sampler = NSGAIISampler(constraints_func=_constraints_func)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=OBJECTIVE_DIRECTIONS,
        sampler=sampler,
        load_if_exists=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bo_optimize.py tests/test_bo_optimize.py
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "implement make_study with NSGA-II and journal storage"
```

---

## Task 4: `launch_training` — subprocess runner

**Files:**
- Modify: `scripts/bo_optimize.py` (implement `launch_training`)
- Modify: `tests/test_bo_optimize.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_bo_optimize.py`:
```python
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
    # tags are nargs="*" — bo-s13 first, then trial tag
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: 2 new FAILs (NotImplementedError).

- [ ] **Step 3: Implement `launch_training`**

Replace the `launch_training` stub:
```python
FIXED_CLI_ARGS = (
    "--seed", "1",
    "--num-timesteps", "800000000",
    "--num-evals", "8",
    "--episode-length", "100",
    "--joint-armature", "4e-10",
    "--ctrl-dt", "0.0025",
    "--sim-dt", "0.00125",
    "--joints-weight", "5.0",
    "--joints-vel-weight", "0.5",
    "--wrist-pos-weight", "0.1",
    "--bodies-pos-weight", "0.1",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fmt_float(x: float) -> str:
    # Use a consistent format so tests can assert exact strings.
    if abs(x) < 1e-3 and x != 0:
        return f"{x:.0e}"
    return str(x)


def launch_training(params: dict, tag: str, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / TRAINING_SCRIPT),
        "--force-scale", _fmt_float(params["fs"]),
        "--joint-damping", _fmt_float(params["damp"]),
        "--control-cost", _fmt_float(params["cc"]),
        "--control-diff-cost", _fmt_float(params["cdc"]),
        "--qvel-init", str(params["qvel_init"]),
        *FIXED_CLI_ARGS,
        "--wandb-tags", "bo-s13", tag,
    ]
    with open(log_path, "w") as fp:
        result = subprocess.run(
            cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    return result.returncode
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: all PASS. (Note: `_fmt_float(7e-7)` produces `"7e-07"` to match the test.)

- [ ] **Step 5: Commit**

```bash
git add scripts/bo_optimize.py tests/test_bo_optimize.py
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "implement launch_training subprocess runner"
```

---

## Task 5: `read_metrics` — wandb API with retry

**Files:**
- Modify: `scripts/bo_optimize.py` (implement `read_metrics`)
- Modify: `tests/test_bo_optimize.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_bo_optimize.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: 3 new FAILs.

- [ ] **Step 3: Implement `read_metrics`**

Replace the `read_metrics` stub and also add `_wandb_api`:
```python
def _wandb_api():
    import wandb  # imported lazily so tests can patch without wandb login
    return wandb.Api()


def read_metrics(tag: str, retries: int = 3, backoff_s: float = 30.0) -> Optional[dict]:
    for attempt in range(retries):
        api = _wandb_api()
        runs = list(api.runs(
            path=WANDB_PROJECT,
            filters={"tags": {"$in": [tag]}},
        ))
        if runs:
            summary = runs[0].summary
            try:
                R = float(summary[REWARD_KEY])
                bcorr = float(summary["eval/emg_biceps_corr"])
                tcorr = float(summary["eval/emg_triceps_corr"])
                bmae = float(summary["eval/emg_biceps_mae"])
                tmae = float(summary["eval/emg_triceps_mae"])
                btrial = float(summary["eval/emg_biceps_trial_mae"])
                ttrial = float(summary["eval/emg_triceps_trial_mae"])
            except (KeyError, TypeError, ValueError):
                R = bcorr = tcorr = bmae = tmae = btrial = ttrial = float("nan")
            if any(math.isnan(v) for v in (R, bcorr, tcorr, bmae, tmae, btrial, ttrial)):
                return None
            return {
                "R": R,
                "bcorr": bcorr,
                "tcorr": tcorr,
                "bmae": bmae,
                "tmae": tmae,
                "btrial": btrial,
                "ttrial": ttrial,
            }
        if attempt < retries - 1:
            time.sleep(backoff_s)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bo_optimize.py tests/test_bo_optimize.py
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "implement read_metrics with wandb API + retry"
```

---

## Task 6: `report_frontier` — Pareto filter + CSV output

**Files:**
- Modify: `scripts/bo_optimize.py` (implement `report_frontier`)
- Modify: `tests/test_bo_optimize.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_bo_optimize.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: 1 new FAIL.

- [ ] **Step 3: Implement `report_frontier`**

Replace the `report_frontier` stub:
```python
import csv as _csv


def report_frontier(study: optuna.Study, out_csv: Path) -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    winners = []
    for t in completed:
        R = t.user_attrs.get("R")
        if R is None or R < WINNER_REWARD_FLOOR:
            continue
        # Require values to be present (completed multi-objective trials have values).
        if t.values is None:
            continue
        bcorr, tcorr, bmae, tmae, btrial, ttrial = t.values
        winners.append({
            "trial": t.number,
            "R": R,
            "bcorr": bcorr,
            "tcorr": tcorr,
            "bmae": bmae,
            "tmae": tmae,
            "btrial": btrial,
            "ttrial": ttrial,
            "fs": t.params.get("fs"),
            "damp": t.params.get("damp"),
            "cc": t.params.get("cc"),
            "cdc": t.params.get("cdc"),
            "qvel_init": t.params.get("qvel_init"),
            "source_name": t.user_attrs.get("source_name", ""),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial", "R", "bcorr", "tcorr", "bmae", "tmae", "btrial", "ttrial",
                  "fs", "damp", "cc", "cdc", "qvel_init", "source_name"]
    with open(out_csv, "w", newline="") as fp:
        w = _csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for row in winners:
            w.writerow(row)

    print(f"Wrote {len(winners)} winner(s) to {out_csv}")
    for row in sorted(winners, key=lambda r: -r["bcorr"])[:5]:
        print(f"  trial={row['trial']} R={row['R']:.0f} "
              f"bcorr={row['bcorr']:.2f} tcorr={row['tcorr']:.2f} "
              f"bmae={row['bmae']:.2f} tmae={row['tmae']:.2f} "
              f"btrial={row['btrial']:.2f} ttrial={row['ttrial']:.2f} "
              f"fs={row['fs']:.2f} damp={row['damp']:.1e} cc={row['cc']:.3f} cdc={row['cdc']:.3f}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bo_optimize.py tests/test_bo_optimize.py
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "implement report_frontier with R>=400 filter"
```

---

## Task 7: `main()` loop — ask/tell integration + guardrails

**Files:**
- Modify: `scripts/bo_optimize.py` (implement `main`)
- Modify: `tests/test_bo_optimize.py` (add test)

Responsibilities of `main`:
- CLI: `--warmstart-csv`, `--study-name`, `--journal`, `--n-trials`, `--jsonl-log`, `--frontier-csv`, `--log-dir`
- Load warm-start, create/resume study, then loop N times: `ask` → `launch_training` → `read_metrics` → `tell` (or FAIL).
- Append to JSONL after each successful trial.
- Abort + loud log after 5 consecutive FAIL trials.

- [ ] **Step 1: Write failing test**

Append to `tests/test_bo_optimize.py`:
```python
def test_main_loop_completes_with_mocked_training(monkeypatch, tmp_path):
    # Fake successful training: subprocess returns 0 and wandb returns metrics.
    def fake_launch(params, tag, log_dir):
        (log_dir / f"{tag}.log").write_text("ok")
        return 0

    call_log = []

    def fake_read(tag, retries=3, backoff_s=30.0):
        call_log.append(tag)
        return {
            "R": 405.0 + len(call_log),
            "bcorr": 0.7,
            "tcorr": 0.6,
            "bmae": 0.12,
            "tmae": 0.13,
            "btrial": 0.18,
            "ttrial": 0.19,
        }

    monkeypatch.setattr(bo, "launch_training", fake_launch)
    monkeypatch.setattr(bo, "read_metrics", fake_read)

    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    argv = [
        "bo_optimize.py",
        "--warmstart-csv", str(fixture),
        "--study-name", "main-loop-test",
        "--journal", str(tmp_path / "j.log"),
        "--n-trials", "3",
        "--jsonl-log", str(tmp_path / "trials.jsonl"),
        "--frontier-csv", str(tmp_path / "frontier.csv"),
        "--log-dir", str(tmp_path / "runs"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    bo.main()

    assert len(call_log) == 3
    jsonl = (tmp_path / "trials.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 3
    # Frontier contains at least 1 row (all 3 trials are feasible with R>=400)
    frontier_rows = (tmp_path / "frontier.csv").read_text().strip().splitlines()
    assert len(frontier_rows) >= 2  # header + >=1 data


def test_main_loop_aborts_after_5_consecutive_failures(monkeypatch, tmp_path):
    def fake_launch(params, tag, log_dir):
        (log_dir / f"{tag}.log").write_text("fail")
        return 1  # non-zero exit

    monkeypatch.setattr(bo, "launch_training", fake_launch)
    monkeypatch.setattr(bo, "read_metrics", lambda *a, **kw: None)

    fixture = Path(__file__).parent / "fixtures" / "bo_warmstart_sample.csv"
    argv = [
        "bo_optimize.py",
        "--warmstart-csv", str(fixture),
        "--study-name", "abort-test",
        "--journal", str(tmp_path / "j.log"),
        "--n-trials", "20",
        "--jsonl-log", str(tmp_path / "trials.jsonl"),
        "--frontier-csv", str(tmp_path / "frontier.csv"),
        "--log-dir", str(tmp_path / "runs"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        bo.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: 2 new FAILs (NotImplementedError from `main`).

- [ ] **Step 3: Implement `main`**

Replace the `main` stub:
```python
CONSECUTIVE_FAIL_LIMIT = 5


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fp:
        fp.write(json.dumps(row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warmstart-csv", type=Path, required=True)
    p.add_argument("--study-name", type=str, required=True)
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--jsonl-log", type=Path, required=True)
    p.add_argument("--frontier-csv", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    args = p.parse_args()

    study = make_study(args.study_name, args.journal)

    # Warm-start only if the study is empty (avoid re-adding on resume).
    if len(study.trials) == 0:
        warmstarts = load_warmstart(args.warmstart_csv)
        print(f"Adding {len(warmstarts)} warm-start trials...")
        study.add_trials(warmstarts)
    else:
        print(f"Resuming study with {len(study.trials)} existing trials; skipping warm-start.")

    consecutive_fail = 0
    for i in range(args.n_trials):
        trial = study.ask(SEARCH_SPACE_DISTRIBUTIONS)
        tag = f"trial-{trial.number:04d}"
        params = {
            "fs": trial.params["fs"],
            "damp": trial.params["damp"],
            "cc": trial.params["cc"],
            "cdc": trial.params["cdc"],
            "qvel_init": trial.params["qvel_init"],
        }
        print(f"[{i+1}/{args.n_trials}] {tag}  params={params}")

        rc = launch_training(params, tag, args.log_dir)
        metrics = read_metrics(tag) if rc == 0 else None

        if metrics is None:
            print(f"  {tag} FAILED (rc={rc} or no metrics)")
            study.tell(trial, state=TrialState.FAIL)
            consecutive_fail += 1
            if consecutive_fail >= CONSECUTIVE_FAIL_LIMIT:
                print(f"ABORT: {CONSECUTIVE_FAIL_LIMIT} consecutive failures", file=sys.stderr)
                _append_jsonl(args.jsonl_log, {"event": "abort",
                                               "reason": "consecutive_fail_limit",
                                               "trial": trial.number})
                sys.exit(2)
            continue

        consecutive_fail = 0
        R = metrics["R"]
        trial.set_user_attr("constraint", [CONSTRAINT_REWARD_FLOOR - R])
        trial.set_user_attr("R", R)
        study.tell(trial, [
            metrics["bcorr"], metrics["tcorr"],
            metrics["bmae"], metrics["tmae"],
            metrics["btrial"], metrics["ttrial"],
        ])
        _append_jsonl(args.jsonl_log, {
            "trial": trial.number,
            "tag": tag,
            **params,
            **metrics,
        })
        print(f"  {tag} OK  R={R:.0f} bcorr={metrics['bcorr']:.2f} "
              f"tcorr={metrics['tcorr']:.2f}")

    report_frontier(study, args.frontier_csv)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_bo_optimize.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bo_optimize.py tests/test_bo_optimize.py
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "implement main() loop with consecutive-fail guardrail"
```

---

## Task 8: Warm-start dry-run against real CSV

No production code changes; only an assertion-style script invocation to verify the filter produces a reasonable count against the real 237-row pool.

**Files:**
- No writes; read-only check.

- [ ] **Step 1: Run dry-run loader**

Run:
```bash
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
import importlib.util
spec = importlib.util.spec_from_file_location('bo', 'scripts/bo_optimize.py')
bo = importlib.util.module_from_spec(spec); spec.loader.exec_module(bo)
trials = bo.load_warmstart(Path('s11_ms_s10_ms_final.csv'))
print(f'retained: {len(trials)}')
feasible = sum(1 for t in trials if t.user_attrs['constraint'][0] <= 0)
print(f'feasible (R>=380): {feasible}')
if trials:
    fss = [t.params['fs'] for t in trials]
    damps = [t.params['damp'] for t in trials]
    print(f'fs range:   {min(fss):.3f} .. {max(fss):.3f}')
    print(f'damp range: {min(damps):.2e} .. {max(damps):.2e}')
"
```

Expected: retained count between 150 and 230 (spec estimate: ~200). Feasible count between 30 and 60. fs range inside [0.3, 1.0]. damp range inside [1e-7, 1.5e-6].

- [ ] **Step 2: If count is <100 or 0, investigate — likely a column-name mismatch**

Check CSV column headers with:
```bash
head -1 s11_ms_s10_ms_final.csv | tr ',' '\n' | grep -E "(reference_data_path|force_scale|joint_damping|control_cost|control_diff|qvel_init|biceps_corr|triceps_corr|biceps_mae|triceps_mae|episode_reward)"
```

If any expected column is missing, update `load_warmstart` to match actual column names and re-run Task 8 Step 1.

- [ ] **Step 3: No commit — this is a verification step only**

---

## Task 9: Single-trial end-to-end smoke test

Launch ONE real training run with a short `--num-timesteps` so the full ask/tell/wandb loop is exercised in ~5–10 minutes instead of ~1 hour.

**Files:**
- No writes; runtime verification.

- [ ] **Step 1: Override the fixed num-timesteps for smoke**

Temporarily modify `scripts/bo_optimize.py` `FIXED_CLI_ARGS`:
```python
FIXED_CLI_ARGS = (
    "--seed", "1",
    "--num-timesteps", "10000000",  # SMOKE ONLY — revert before production
    "--num-evals", "2",              # SMOKE ONLY — revert before production
    "--episode-length", "100",
    "--joint-armature", "4e-10",
    "--ctrl-dt", "0.0025",
    "--sim-dt", "0.00125",
    "--joints-weight", "5.0",
    "--joints-vel-weight", "0.5",
    "--wrist-pos-weight", "0.1",
    "--bodies-pos-weight", "0.1",
)
```

- [ ] **Step 2: Run a 1-trial smoke run**

Run:
```bash
python scripts/bo_optimize.py \
  --warmstart-csv s11_ms_s10_ms_final.csv \
  --study-name bo-smoke \
  --journal /tmp/bo_smoke.log \
  --n-trials 1 \
  --jsonl-log /tmp/bo_smoke.jsonl \
  --frontier-csv /tmp/bo_smoke_frontier.csv \
  --log-dir /tmp/bo_smoke_runs
```

Expected: ~5–10 min wallclock. Exits with 0. One wandb run appears with tags `bo-s13` and `trial-0XXX` (trial number is >= warm-start count, since warm-starts fill the low numbers). `/tmp/bo_smoke.jsonl` has exactly 1 line. `/tmp/bo_smoke_frontier.csv` has header + 0 or 1 rows (smoke R may be below 400).

- [ ] **Step 3: Verify the wandb run has all 5 metrics**

Find the smoke run via its tag on wandb web UI (or via API) and confirm it reports `eval/episode_reward`, `eval/emg_biceps_corr`, `eval/emg_triceps_corr`, `eval/emg_biceps_mae`, `eval/emg_triceps_mae` in its summary. If any is missing, the short smoke training may not have reached the eval step — increase `--num-timesteps` for smoke to `50000000` and re-run step 2.

- [ ] **Step 4: Resume check**

Run the same command again (same `--study-name bo-smoke`, same `--journal`). Expected output line: `Resuming study with <N+1> existing trials; skipping warm-start.` where N was the warm-start count. Then it runs one more trial. Kill with ctrl-C after you see the subprocess start — confirm journal file is intact and `optuna` can re-read it.

- [ ] **Step 5: Revert smoke overrides**

Restore `FIXED_CLI_ARGS` to the production values (`--num-timesteps 800000000`, `--num-evals 8`).

- [ ] **Step 6: Commit the reverted file**

```bash
git add scripts/bo_optimize.py
git -c user.email=eric@talmolab.org -c user.name=eric commit -m "revert smoke-run overrides after e2e verification"
```

(If Task 9 Step 3 required increasing the smoke `--num-timesteps`, that change is only in the intermediate commits of this task — the final commit here restores production settings.)

---

## Task 10: Launch the production 40-trial BO run

**Files:**
- No code changes; runtime-only.

- [ ] **Step 1: Start the run under nohup so it survives disconnect**

Run:
```bash
nohup python scripts/bo_optimize.py \
  --warmstart-csv s11_ms_s10_ms_final.csv \
  --study-name bo-s13-prod \
  --journal bo_study.log \
  --n-trials 40 \
  --jsonl-log bo_trials.jsonl \
  --frontier-csv bo_frontier.csv \
  --log-dir bo_runs \
  > bo_driver.log 2>&1 &
echo $!
```

Expected: returns a PID. `bo_driver.log` starts filling.

- [ ] **Step 2: Confirm first trial is launching**

Wait ~30s then:
```bash
tail -20 bo_driver.log
```

Expected: you see `Adding ~200 warm-start trials...` then `[1/40] trial-NNNN  params=...`.

- [ ] **Step 3: Mid-run spot check (~4h in)**

```bash
wc -l bo_trials.jsonl
tail -5 bo_driver.log
```

Expected: `bo_trials.jsonl` has 3–5 lines; driver log shows progression.

- [ ] **Step 4: On completion**

`bo_driver.log` ends with a `Wrote N winner(s) to bo_frontier.csv` line and a summary of top-5 by bcorr.

Inspect:
```bash
cat bo_frontier.csv
```

Report winners back to stakeholder.

---

## Self-review checklist (post-plan)

- [ ] **Spec coverage:** Every section of the spec mapped to a task
  - Search space (spec §"Problem formulation") → Task 2 `SEARCH_SPACE_DISTRIBUTIONS`, Task 7 `main`
  - Objectives + constraint → Task 3 `_constraints_func`, Task 7 `main`
  - Warm-start + filter → Task 2
  - Architecture → Tasks 1–7
  - Components → Tasks 2–7 (one task each)
  - Data flow per trial → Task 7
  - Error handling (fail states + 5-consecutive guardrail) → Task 7
  - Testing (3 pre-flight checks) → Tasks 8, 9 (9 covers both smoke and resume)
  - Outputs → Task 7 `main` (journal, jsonl, runs/*.log, frontier.csv)
- [ ] **No placeholders:** every code block is complete, no "TBD"
- [ ] **Type consistency:** function signatures match across tasks (`load_warmstart(Path) -> list[FrozenTrial]`, etc.). Names `fs/damp/cc/cdc/qvel_init` are consistent throughout.

---

## Notes on execution

- The 40-trial run is ~40h. Launch it so the finish lands during working hours the day after next; come back to the frontier CSV then.
- If the NSGA-II surrogate pushes all early trials into the infeasible R<380 corner, that's expected for the first 2–3 trials — it means the constraint is new information the sampler needs to learn. Don't panic-kill before trial 5.
- If you need to widen the search space mid-run, don't — start a new study. The journal file is tied to this study's distributions.
