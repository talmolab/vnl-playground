# s15-ms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three EMG-metric upgrades (percentile normalization, per-trial Pearson, lagged cross-correlation with ±50 ms window), log them during every eval cycle in wandb, re-evaluate 8 frontier checkpoints, then run a branch-gated Stage 3 sweep (0–22 runs, ≤12 h wall-clock).

**Architecture:** Extract pure-numpy EMG metric helpers into a new `vnl_playground/eval_metrics/emg.py` module shared by both the trainer (`train_mouse_janelia_sigmoid_moving_shoulder.py`) and the eval-replay script (`scripts/emg_comparison.py`). TDD each metric addition with a unit test in `tests/test_emg_metrics.py`. All changes are eval-side; no training-reward changes.

**Tech Stack:** Python 3, numpy, scipy.signal (existing), pytest (existing), wandb, orbax checkpoints, brax PPO (unchanged).

**Spec:** `docs/superpowers/specs/2026-04-23-s15-ms-design.md`. Thinking log: `docs/superpowers/specs/2026-04-23-s15-ms-thinking.md`.

**Execution environment:** `cd /root/vast/eric/vnl-playground`. For tests and local experimentation: `.venv/bin/python`. For runs that load mujoco/brax/jax (trainer, emg_comparison.py, Stage 3 sweeps): `source /root/vast/eric/track-mjx/.venv/bin/activate`.

**Branches are determined by Stage 2 results.** Task 12 is a conditional switch that executes exactly one of 12A–12D based on the Stage 2 eval_matrix.csv.

---

## Task 1: Create shared EMG metrics module (empty scaffold)

**Files:**
- Create: `vnl_playground/eval_metrics/__init__.py`
- Create: `vnl_playground/eval_metrics/emg.py`
- Create: `tests/test_emg_metrics.py`

- [ ] **Step 1.1: Create the package directory and empty `__init__.py`**

```bash
mkdir -p /root/vast/eric/vnl-playground/vnl_playground/eval_metrics
touch /root/vast/eric/vnl-playground/vnl_playground/eval_metrics/__init__.py
```

- [ ] **Step 1.2: Create `emg.py` with module docstring and imports only**

Write `/root/vast/eric/vnl-playground/vnl_playground/eval_metrics/emg.py`:
```python
"""EMG evaluation metrics shared by the trainer and the eval-replay script.

Pure-numpy implementations; no jax, no mujoco, no brax. All functions accept
(n_trials, T) arrays for sim and bio traces and return plain Python floats /
ints suitable for wandb logging.
"""
from __future__ import annotations

import numpy as np

LAG_RANGE_STEPS_DEFAULT = 20  # ±20 steps × ctrl-dt = ±50 ms at ctrl_dt=2.5 ms
```

- [ ] **Step 1.3: Create `tests/test_emg_metrics.py` with a sanity test**

Write `/root/vast/eric/vnl-playground/tests/test_emg_metrics.py`:
```python
"""Unit tests for vnl_playground.eval_metrics.emg."""
from __future__ import annotations

import numpy as np
import pytest


def test_module_imports():
    from vnl_playground.eval_metrics import emg
    assert emg.LAG_RANGE_STEPS_DEFAULT == 20
```

- [ ] **Step 1.4: Run the sanity test**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: `test_module_imports PASSED`

- [ ] **Step 1.5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add vnl_playground/eval_metrics/__init__.py vnl_playground/eval_metrics/emg.py tests/test_emg_metrics.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "scaffold shared EMG metrics module for s15-ms"
```

---

## Task 2: Implement `lagged_corr` helper (TDD)

**Files:**
- Modify: `vnl_playground/eval_metrics/emg.py`
- Modify: `tests/test_emg_metrics.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/test_emg_metrics.py`:
```python
def test_lagged_corr_identical_sines_returns_one_and_zero_lag():
    from vnl_playground.eval_metrics import emg
    t = np.linspace(0, 2 * np.pi, 60)
    sig = np.sin(t)
    m = emg.compute_lag_metrics(sig, sig, ctrl_dt_ms=2.5)
    assert m["lagged_corr_max"] == pytest.approx(1.0, abs=1e-6)
    assert m["phase_lag_steps"] == 0
    assert m["phase_lag_ms"] == pytest.approx(0.0)


def test_lagged_corr_shifted_sines_finds_the_lag():
    from vnl_playground.eval_metrics import emg
    t = np.linspace(0, 2 * np.pi, 60)
    bio = np.sin(t)
    # Sim is 5 samples ahead of bio. Slicing bio[5:]/sim[:-5] gives r=1 at lag=+5.
    sim = np.sin(t + 5 * (t[1] - t[0]))
    m = emg.compute_lag_metrics(sim, bio, ctrl_dt_ms=2.5)
    assert m["lagged_corr_max"] > 0.99
    assert abs(m["phase_lag_steps"]) == 5
    assert m["phase_lag_ms"] == pytest.approx(m["phase_lag_steps"] * 2.5)


def test_lagged_corr_uncorrelated_noise_returns_low_corr():
    from vnl_playground.eval_metrics import emg
    rng = np.random.default_rng(0)
    a = rng.normal(size=60)
    b = rng.normal(size=60)
    m = emg.compute_lag_metrics(a, b, ctrl_dt_ms=2.5)
    # Best-over-41-lags corr of iid noise has some positive expectation,
    # but well under 0.6 with high probability.
    assert m["lagged_corr_max"] < 0.6


def test_lagged_corr_returns_expected_keys():
    from vnl_playground.eval_metrics import emg
    sig = np.sin(np.linspace(0, 2 * np.pi, 60))
    m = emg.compute_lag_metrics(sig, sig, ctrl_dt_ms=2.5)
    assert set(m.keys()) == {
        "lagged_corr_max", "phase_lag_steps", "phase_lag_ms",
        "lagged_corr_at_0", "lagged_corr_at_neg5", "lagged_corr_at_pos5",
        "lagged_corr_fwhm_steps",
    }
```

- [ ] **Step 2.2: Run tests — verify they fail**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: 4 new tests FAIL with `AttributeError: module 'vnl_playground.eval_metrics.emg' has no attribute 'compute_lag_metrics'`.

- [ ] **Step 2.3: Implement `compute_lag_metrics`**

Append to `vnl_playground/eval_metrics/emg.py`:
```python
def compute_lag_metrics(sim_mean, bio_mean, ctrl_dt_ms: float = 2.5,
                        lag_range_steps: int = LAG_RANGE_STEPS_DEFAULT) -> dict:
    """Cross-correlation with lag over ±lag_range_steps steps.

    Takes 1-D sim_mean and bio_mean (trial-averaged traces). Returns dict with:
      lagged_corr_max, phase_lag_steps, phase_lag_ms,
      lagged_corr_at_0, lagged_corr_at_neg5, lagged_corr_at_pos5,
      lagged_corr_fwhm_steps.

    Positive phase_lag_steps means sim leads bio by that many steps (i.e., the
    best match is at bio[lag:] vs sim[:-lag]).
    """
    sim_mean = np.asarray(sim_mean, dtype=np.float64)
    bio_mean = np.asarray(bio_mean, dtype=np.float64)
    L = min(len(sim_mean), len(bio_mean))
    sim_mean = sim_mean[:L]
    bio_mean = bio_mean[:L]

    lags = np.arange(-lag_range_steps, lag_range_steps + 1)
    corrs = np.full(lags.shape, np.nan, dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag < 0:
            s, b = sim_mean[-lag:], bio_mean[:L + lag]
        elif lag > 0:
            s, b = sim_mean[:-lag], bio_mean[lag:]
        else:
            s, b = sim_mean, bio_mean
        if len(s) >= 3 and s.std() > 0 and b.std() > 0:
            corrs[i] = np.corrcoef(s, b)[0, 1]

    if np.all(np.isnan(corrs)):
        return {
            "lagged_corr_max": float("nan"),
            "phase_lag_steps": 0,
            "phase_lag_ms": 0.0,
            "lagged_corr_at_0": float("nan"),
            "lagged_corr_at_neg5": float("nan"),
            "lagged_corr_at_pos5": float("nan"),
            "lagged_corr_fwhm_steps": 0,
        }

    argmax = int(np.nanargmax(corrs))
    best_r = float(corrs[argmax])
    best_lag = int(lags[argmax])
    half_max = max(best_r / 2.0, 0.0)
    above = np.nan_to_num(corrs, nan=-np.inf) >= half_max
    fwhm_steps = int(np.sum(above))

    def _at(target_lag: int) -> float:
        idx = int(np.where(lags == target_lag)[0][0])
        v = corrs[idx]
        return float(v) if np.isfinite(v) else float("nan")

    return {
        "lagged_corr_max": best_r,
        "phase_lag_steps": best_lag,
        "phase_lag_ms": best_lag * float(ctrl_dt_ms),
        "lagged_corr_at_0": _at(0),
        "lagged_corr_at_neg5": _at(-5),
        "lagged_corr_at_pos5": _at(+5),
        "lagged_corr_fwhm_steps": fwhm_steps,
    }
```

- [ ] **Step 2.4: Run tests — verify they pass**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: 5 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add vnl_playground/eval_metrics/emg.py tests/test_emg_metrics.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "implement compute_lag_metrics helper for EMG phase analysis"
```

---

## Task 3: Implement `compute_per_trial_metrics` helper (TDD)

**Files:**
- Modify: `vnl_playground/eval_metrics/emg.py`
- Modify: `tests/test_emg_metrics.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/test_emg_metrics.py`:
```python
def test_per_trial_metrics_identical_traces():
    from vnl_playground.eval_metrics import emg
    rng = np.random.default_rng(42)
    trials = rng.normal(size=(5, 60))
    m = emg.compute_per_trial_metrics(trials, trials, ctrl_dt_ms=2.5)
    assert m["trial_corr_mean"] == pytest.approx(1.0)
    assert m["trial_corr_median"] == pytest.approx(1.0)
    assert m["per_trial_lagged_corr_mean"] == pytest.approx(1.0)
    assert m["per_trial_lagged_corr_median"] == pytest.approx(1.0)
    assert m["per_trial_phase_lag_mean_ms"] == pytest.approx(0.0)
    assert m["per_trial_phase_lag_std_ms"] == pytest.approx(0.0)


def test_per_trial_metrics_handles_mismatched_trial_counts():
    from vnl_playground.eval_metrics import emg
    rng = np.random.default_rng(0)
    sim = rng.normal(size=(3, 30))
    bio = rng.normal(size=(7, 30))
    m = emg.compute_per_trial_metrics(sim, bio, ctrl_dt_ms=2.5)
    # Must not raise and must return finite trial_corr_mean (first 3 trials).
    assert np.isfinite(m["trial_corr_mean"])


def test_per_trial_metrics_handles_zero_variance_trial():
    from vnl_playground.eval_metrics import emg
    sim = np.zeros((2, 30))
    sim[1] = np.sin(np.linspace(0, 2 * np.pi, 30))
    bio = np.sin(np.linspace(0, 2 * np.pi, 30))[np.newaxis].repeat(2, axis=0)
    m = emg.compute_per_trial_metrics(sim, bio, ctrl_dt_ms=2.5)
    # Trial 0 is zero-variance; should be skipped, not crash.
    assert np.isfinite(m["trial_corr_mean"])
```

- [ ] **Step 3.2: Run tests — verify they fail**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: 3 new tests FAIL with `AttributeError: compute_per_trial_metrics`.

- [ ] **Step 3.3: Implement `compute_per_trial_metrics`**

Append to `vnl_playground/eval_metrics/emg.py`:
```python
def compute_per_trial_metrics(sim_muscle, bio_traces, ctrl_dt_ms: float = 2.5,
                              lag_range_steps: int = LAG_RANGE_STEPS_DEFAULT) -> dict:
    """Per-trial Pearson r, per-trial MAE, and per-trial lag summary.

    sim_muscle: (n_sim, T). bio_traces: (n_bio, T). Aligns first min(n_sim, n_bio)
    pairs. Trials with zero variance in either trace are skipped in corr / lag
    statistics but still contribute to MAE.
    """
    sim_muscle = np.asarray(sim_muscle, dtype=np.float64)
    bio_traces = np.asarray(bio_traces, dtype=np.float64)
    n = min(sim_muscle.shape[0], bio_traces.shape[0])
    T = min(sim_muscle.shape[1], bio_traces.shape[1])
    sim = sim_muscle[:n, :T]
    bio = bio_traces[:n, :T]

    trial_corrs = []
    trial_maes = []
    per_trial_lagged_corrs = []
    per_trial_lags_steps = []
    for i in range(n):
        s, b = sim[i], bio[i]
        trial_maes.append(float(np.mean(np.abs(s - b))))
        if s.std() > 0 and b.std() > 0:
            r = np.corrcoef(s, b)[0, 1]
            if np.isfinite(r):
                trial_corrs.append(float(r))
            lag_m = compute_lag_metrics(s, b, ctrl_dt_ms=ctrl_dt_ms,
                                        lag_range_steps=lag_range_steps)
            if np.isfinite(lag_m["lagged_corr_max"]):
                per_trial_lagged_corrs.append(lag_m["lagged_corr_max"])
                per_trial_lags_steps.append(lag_m["phase_lag_steps"])

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def _median(xs):
        return float(np.median(xs)) if xs else float("nan")

    def _std(xs):
        return float(np.std(xs)) if xs else float("nan")

    return {
        "trial_corr_mean": _mean(trial_corrs),
        "trial_corr_median": _median(trial_corrs),
        "trial_mae": _mean(trial_maes),
        "per_trial_lagged_corr_mean": _mean(per_trial_lagged_corrs),
        "per_trial_lagged_corr_median": _median(per_trial_lagged_corrs),
        "per_trial_phase_lag_mean_ms": float(np.mean(per_trial_lags_steps) * ctrl_dt_ms) if per_trial_lags_steps else float("nan"),
        "per_trial_phase_lag_std_ms": float(np.std(per_trial_lags_steps) * ctrl_dt_ms) if per_trial_lags_steps else float("nan"),
    }
```

- [ ] **Step 3.4: Run tests — verify they pass**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: 8 tests PASS total (1 module + 4 lag + 3 per-trial).

- [ ] **Step 3.5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add vnl_playground/eval_metrics/emg.py tests/test_emg_metrics.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "implement compute_per_trial_metrics with per-trial lag summary"
```

---

## Task 4: Implement unified `compute_all_emg_metrics` entry point (TDD)

**Files:**
- Modify: `vnl_playground/eval_metrics/emg.py`
- Modify: `tests/test_emg_metrics.py`

This is the single function both the trainer and the eval-replay script will call.

- [ ] **Step 4.1: Write the failing test**

Append to `tests/test_emg_metrics.py`:
```python
def test_compute_all_emg_metrics_has_union_of_keys():
    from vnl_playground.eval_metrics import emg
    rng = np.random.default_rng(3)
    sim = rng.uniform(size=(4, 60))
    bio = rng.uniform(size=(4, 60))
    m = emg.compute_all_emg_metrics(sim, bio, ctrl_dt_ms=2.5)
    expected = {
        "mean_corr", "mean_mae",
        "trial_corr_mean", "trial_corr_median", "trial_mae",
        "lagged_corr_max", "phase_lag_steps", "phase_lag_ms",
        "lagged_corr_at_0", "lagged_corr_at_neg5", "lagged_corr_at_pos5",
        "lagged_corr_fwhm_steps",
        "per_trial_lagged_corr_mean", "per_trial_lagged_corr_median",
        "per_trial_phase_lag_mean_ms", "per_trial_phase_lag_std_ms",
    }
    assert expected.issubset(m.keys())


def test_compute_all_emg_metrics_bio_traces_none_still_returns_mean_keys():
    from vnl_playground.eval_metrics import emg
    sim = np.random.default_rng(1).uniform(size=(4, 60))
    bio_mean = sim.mean(axis=0)
    m = emg.compute_all_emg_metrics(sim, bio_mean_only=bio_mean, ctrl_dt_ms=2.5)
    # No bio_traces -> per-trial and per-trial-lag keys are NaN, but mean_corr
    # and lagged_corr_max are present and finite.
    assert np.isfinite(m["mean_corr"])
    assert np.isfinite(m["lagged_corr_max"])
    assert np.isnan(m["trial_corr_mean"])
```

- [ ] **Step 4.2: Run tests — verify failure**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: 2 new tests FAIL.

- [ ] **Step 4.3: Implement `compute_all_emg_metrics`**

Append to `vnl_playground/eval_metrics/emg.py`:
```python
def compute_all_emg_metrics(sim_muscle, bio_traces=None, bio_mean_only=None,
                            ctrl_dt_ms: float = 2.5,
                            lag_range_steps: int = LAG_RANGE_STEPS_DEFAULT) -> dict:
    """One-call entry point used by both the trainer and the eval-replay script.

    sim_muscle: (n_sim, T) simulated muscle activations for one muscle.
    bio_traces: (n_bio, T) per-trial reference EMG (preferred input).
    bio_mean_only: (T,) pre-averaged reference trace (used only if bio_traces is None).

    Returns a flat dict of floats/ints. Keys missing an input source are NaN.
    """
    sim_muscle = np.asarray(sim_muscle, dtype=np.float64)
    if bio_traces is not None:
        bio_traces = np.asarray(bio_traces, dtype=np.float64)
        bio_mean = bio_traces.mean(axis=0)
    elif bio_mean_only is not None:
        bio_mean = np.asarray(bio_mean_only, dtype=np.float64)
    else:
        raise ValueError("Must provide either bio_traces or bio_mean_only.")

    sim_mean = sim_muscle.mean(axis=0)
    L = min(len(sim_mean), len(bio_mean))
    sim_mean = sim_mean[:L]
    bio_mean = bio_mean[:L]

    out = {
        "mean_corr": float(np.corrcoef(sim_mean, bio_mean)[0, 1])
                     if sim_mean.std() > 0 and bio_mean.std() > 0 else float("nan"),
        "mean_mae": float(np.mean(np.abs(sim_mean - bio_mean))),
    }
    out.update(compute_lag_metrics(sim_mean, bio_mean, ctrl_dt_ms=ctrl_dt_ms,
                                   lag_range_steps=lag_range_steps))

    if bio_traces is not None:
        out.update(compute_per_trial_metrics(sim_muscle, bio_traces,
                                             ctrl_dt_ms=ctrl_dt_ms,
                                             lag_range_steps=lag_range_steps))
    else:
        out.update({
            "trial_corr_mean": float("nan"),
            "trial_corr_median": float("nan"),
            "trial_mae": float("nan"),
            "per_trial_lagged_corr_mean": float("nan"),
            "per_trial_lagged_corr_median": float("nan"),
            "per_trial_phase_lag_mean_ms": float("nan"),
            "per_trial_phase_lag_std_ms": float("nan"),
        })

    return out
```

- [ ] **Step 4.4: Run tests — verify pass**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: 10 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add vnl_playground/eval_metrics/emg.py tests/test_emg_metrics.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "add compute_all_emg_metrics unified entry point for trainer and eval-replay"
```

---

## Task 5: Add `--emg-norm-percentile` CLI flag and thread through trainer

**Files:**
- Modify: `train_mouse_janelia_sigmoid_moving_shoulder.py` (argparse, `load_emg_reference`, call site)

- [ ] **Step 5.1: Add the argparse flag**

In `train_mouse_janelia_sigmoid_moving_shoulder.py`, right after the `--muscle-tau-deact` argument (around line 414), add:
```python
    p.add_argument("--emg-norm-percentile", type=float, default=100.0,
                   help="Percentile used to normalize reference EMG envelopes (arr / np.percentile(arr, P)). "
                        "Default 100 (true max) ensures no reference sample exceeds 1.0 pre-clip. "
                        "Pre-s15 default was 98 — use 98.0 to reproduce old metrics.")
```

- [ ] **Step 5.2: Update `load_emg_reference` signature**

In `train_mouse_janelia_sigmoid_moving_shoulder.py:91`, change:
```python
def load_emg_reference(n_clips, target_timesteps, clip_start_frame=0):
```
to:
```python
def load_emg_reference(n_clips, target_timesteps, clip_start_frame=0,
                       norm_percentile: float = 100.0):
```

- [ ] **Step 5.3: Use the new parameter at the normalization line**

In the same function at line 143, change:
```python
            emg_by_muscle[muscle_name] = arr / np.percentile(arr, 98)
```
to:
```python
            emg_by_muscle[muscle_name] = arr / np.percentile(arr, norm_percentile)
```

- [ ] **Step 5.4: Update the call site**

In `train_mouse_janelia_sigmoid_moving_shoulder.py` around line 1505, change:
```python
    emg_reference = load_emg_reference(
        n_emg_clips,
        emg_target_timesteps,
    )
```
to (use the exact existing argument list; the key additions are `clip_start_frame` threading if present, and the new `norm_percentile`):
```python
    emg_reference = load_emg_reference(
        n_emg_clips,
        emg_target_timesteps,
        norm_percentile=args.emg_norm_percentile,
    )
```
If `clip_start_frame=...` is already passed, preserve it; only add `norm_percentile=args.emg_norm_percentile`.

- [ ] **Step 5.5: Smoke-test the flag is present in the source**

Run: `cd /root/vast/eric/vnl-playground && grep -n "emg-norm-percentile\|emg_norm_percentile" train_mouse_janelia_sigmoid_moving_shoulder.py`
Expected: 2 matches — one in argparse (`p.add_argument(...)`), one at the `load_emg_reference(...)` call site (`norm_percentile=args.emg_norm_percentile`).

- [ ] **Step 5.6: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add train_mouse_janelia_sigmoid_moving_shoulder.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "add --emg-norm-percentile CLI flag (default 100) to fix EMG peak clipping"
```

---

## Task 6: Replace in-trainer `compute_emg_metrics` with shared module and log new metrics

**Files:**
- Modify: `train_mouse_janelia_sigmoid_moving_shoulder.py` (lines 159–182, 2020–2034)

- [ ] **Step 6.1: Import shared module**

Near the other `from vnl_playground...` imports at the top of the trainer (do a `grep -n "from vnl_playground" train_mouse_janelia_sigmoid_moving_shoulder.py` to locate), add:
```python
from vnl_playground.eval_metrics import emg as emg_metrics
```

- [ ] **Step 6.2: Delete the in-file `compute_emg_metrics` definition**

In `train_mouse_janelia_sigmoid_moving_shoulder.py`, delete lines 159–182 (the local `def compute_emg_metrics(...)` block). It will be replaced by direct calls to the shared module.

- [ ] **Step 6.3: Update the eval-time logging block**

In `train_mouse_janelia_sigmoid_moving_shoulder.py` around lines 2020–2034, replace:
```python
                # Compute metrics per muscle
                emg_metrics = {}
                for sim_idx, sim_name, _, muscle_name in EMG_MUSCLE_CONFIGS:
                    emg_mean = emg_reference["means"].get(muscle_name)
                    if emg_mean is None:
                        continue
                    bio_traces = emg_reference["traces"].get(muscle_name)
                    m = compute_emg_metrics(
                        sim_actions[:, :, sim_idx], emg_mean,
                        bio_traces=bio_traces[:, :emg_target_timesteps] if bio_traces is not None else None,
                    )
                    emg_metrics[muscle_name] = m
                    wandb_log[f"eval/emg_{muscle_name.lower()}_corr"] = m["mean_corr"]
                    wandb_log[f"eval/emg_{muscle_name.lower()}_mae"] = m["mean_mae"]
                    if "trial_mae" in m:
                        wandb_log[f"eval/emg_{muscle_name.lower()}_trial_mae"] = m["trial_mae"]
```
with (note the local variable renamed to `emg_per_muscle` to avoid shadowing the imported module):
```python
                # Compute metrics per muscle using shared module.
                emg_per_muscle = {}
                ctrl_dt_ms = float(env_cfg.ctrl_dt) * 1000.0
                for sim_idx, sim_name, _, muscle_name in EMG_MUSCLE_CONFIGS:
                    emg_mean = emg_reference["means"].get(muscle_name)
                    if emg_mean is None:
                        continue
                    bio_traces = emg_reference["traces"].get(muscle_name)
                    bio_traces_slice = (bio_traces[:, :emg_target_timesteps]
                                        if bio_traces is not None else None)
                    m = emg_metrics.compute_all_emg_metrics(
                        sim_actions[:, :, sim_idx],
                        bio_traces=bio_traces_slice,
                        bio_mean_only=emg_mean if bio_traces_slice is None else None,
                        ctrl_dt_ms=ctrl_dt_ms,
                    )
                    emg_per_muscle[muscle_name] = m
                    prefix = f"eval/emg_{muscle_name.lower()}"
                    # Back-compat aliases (so existing wandb panels keep working).
                    wandb_log[f"{prefix}_corr"] = m["mean_corr"]
                    wandb_log[f"{prefix}_mae"] = m["mean_mae"]
                    wandb_log[f"{prefix}_trial_mae"] = m["trial_mae"]
                    # Full new metric set.
                    for key in (
                        "mean_corr", "mean_mae",
                        "trial_corr_mean", "trial_corr_median", "trial_mae",
                        "lagged_corr_max", "phase_lag_steps", "phase_lag_ms",
                        "lagged_corr_at_0", "lagged_corr_at_neg5", "lagged_corr_at_pos5",
                        "lagged_corr_fwhm_steps",
                        "per_trial_lagged_corr_mean", "per_trial_lagged_corr_median",
                        "per_trial_phase_lag_mean_ms", "per_trial_phase_lag_std_ms",
                    ):
                        if key in m:
                            wandb_log[f"{prefix}_{key}"] = m[key]
```

- [ ] **Step 6.4: Fix any remaining reference to the deleted local name**

Run: `cd /root/vast/eric/vnl-playground && grep -n "compute_emg_metrics\b" train_mouse_janelia_sigmoid_moving_shoulder.py`
Expected: zero results. If any appear, replace `compute_emg_metrics(...)` with `emg_metrics.compute_all_emg_metrics(...)`.

- [ ] **Step 6.5: Syntax-check the trainer**

Run: `cd /root/vast/eric/vnl-playground && /root/vast/eric/track-mjx/.venv/bin/python -c "import ast; ast.parse(open('train_mouse_janelia_sigmoid_moving_shoulder.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 6.6: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add train_mouse_janelia_sigmoid_moving_shoulder.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "wire trainer eval loop to shared EMG metrics module with lag + per-trial logging"
```

---

## Task 7: Mirror changes into `scripts/emg_comparison.py`

**Files:**
- Modify: `scripts/emg_comparison.py` (add CLI flag, use shared module, expand result columns)

- [ ] **Step 7.1: Add CLI flag and import shared module**

Near the top of `scripts/emg_comparison.py` (after the existing imports, around line 40–44 after the `from vnl_playground.tasks.mouse.imitation import ...` line), add:
```python
from vnl_playground.eval_metrics import emg as emg_metrics
```
In the argparse section of `scripts/emg_comparison.py` (search: `grep -n "add_argument" scripts/emg_comparison.py`), add a new argument next to `--checkpoint`:
```python
    parser.add_argument("--emg-norm-percentile", type=float, default=100.0,
                        help="Percentile for reference EMG normalization. 100 matches s15 trainer default; "
                             "use 98 to reproduce pre-s15 metrics.")
```

- [ ] **Step 7.2: Locate EMG loading in this script and parameterize**

Search: `grep -n "np.percentile\|percentile.*98" scripts/emg_comparison.py`
For every `np.percentile(arr, 98)` found, replace `98` with `args.emg_norm_percentile`. (Typically one location mirroring the trainer.)

- [ ] **Step 7.3: Replace the in-file `compute_emg_metrics` (lines 187–222)**

Delete lines 187–222 of `scripts/emg_comparison.py` (the local `def compute_emg_metrics` block).

At the call site around line 657, replace the call. Search: `grep -n "compute_emg_metrics\b" scripts/emg_comparison.py` — confirm one remaining call. Replace:
```python
            metrics_by_muscle[muscle_name] = compute_emg_metrics(sim_muscle, emg_traces)
```
with (derive `ctrl_dt_ms` from the checkpoint's config; search the script for where `ctrl_dt` is loaded and use that value):
```python
            metrics_by_muscle[muscle_name] = emg_metrics.compute_all_emg_metrics(
                sim_muscle, bio_traces=emg_traces, ctrl_dt_ms=ctrl_dt_ms,
            )
```
If `ctrl_dt_ms` is not already in scope, derive it near the top of the eval loop:
```python
ctrl_dt_ms = float(config.get("ctrl_dt", 0.0025)) * 1000.0  # or the equivalent lookup the script uses
```

- [ ] **Step 7.4: Ensure downstream result consumers still work**

The old return dict used keys `mean_corr`, `mean_mae`, `trial_corrs` (array), `mean_trial_corr`, `trial_maes` (array), `mean_trial_mae`. The shared module returns `mean_corr`, `mean_mae`, `trial_corr_mean`, `trial_corr_median`, `trial_mae`, plus the lag metrics. There are **no per-trial arrays** in the new dict.

Search: `grep -n "trial_corrs\|trial_maes\|mean_trial_corr\|mean_trial_mae" scripts/emg_comparison.py` — for each reference:
- `mean_trial_corr` → `trial_corr_mean`
- `mean_trial_mae` → `trial_mae`
- `trial_corrs` (array) → if used only for plotting/reporting, either recompute inline (loop over trials once) OR drop if the plot is optional. If the script writes a CSV/JSON with per-trial data, add a small helper that reconstructs per-trial arrays from the same inputs, using the same logic as `compute_per_trial_metrics` but returning the full lists.

Minimal patch: for any per-trial array consumer, add an extra `compute_per_trial_metrics` call and keep the array by extracting it inline. Do not block on this if the script runs without errors in Step 7.6.

- [ ] **Step 7.5: Syntax-check the eval-replay script**

Run: `cd /root/vast/eric/vnl-playground && /root/vast/eric/track-mjx/.venv/bin/python -c "import ast; ast.parse(open('scripts/emg_comparison.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 7.6: Dry-run on one checkpoint**

Run: `cd /root/vast/eric/vnl-playground && source /root/vast/eric/track-mjx/.venv/bin/activate && python scripts/emg_comparison.py --checkpoint checkpoints/s13-ms-armM-anchorA-fs1p1-s2-20260421-043700 --emg-norm-percentile 100 2>&1 | tail -40`
Expected: the script runs, prints per-muscle metrics including `lagged_corr_max` and `phase_lag_ms`, no exceptions.

- [ ] **Step 7.7: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/emg_comparison.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "wire emg_comparison.py to shared metrics module and expose --emg-norm-percentile"
```

---

## Task 8: Parity test — trainer and eval-replay return the same metrics on matched inputs

**Files:**
- Modify: `tests/test_emg_metrics.py`

- [ ] **Step 8.1: Add a parity-through-module test**

Append to `tests/test_emg_metrics.py`:
```python
def test_back_compat_keys_present_for_old_wandb_panels():
    """Existing dashboards read eval/emg_{m}_corr and eval/emg_{m}_mae. The
    unified entry point must still produce mean_corr and mean_mae so the
    trainer's back-compat wandb aliases can be populated."""
    from vnl_playground.eval_metrics import emg
    sim = np.random.default_rng(7).uniform(size=(3, 50))
    bio = np.random.default_rng(8).uniform(size=(3, 50))
    m = emg.compute_all_emg_metrics(sim, bio_traces=bio, ctrl_dt_ms=2.5)
    assert "mean_corr" in m and "mean_mae" in m
    assert "trial_mae" in m
```

- [ ] **Step 8.2: Run all tests**

Run: `cd /root/vast/eric/vnl-playground && .venv/bin/python -m pytest tests/test_emg_metrics.py -v`
Expected: 11 tests PASS.

- [ ] **Step 8.3: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add tests/test_emg_metrics.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "add back-compat key assertions for shared EMG metrics module"
```

---

## Task 9: Stage 2 — Pin the 8 evaluation targets and build the driver script

**Files:**
- Create: `scripts/s15_stage2_eval.sh`
- Create: `plots/2026-04-23-s15-stage2/` (output dir)

- [ ] **Step 9.1: Resolve the 8 checkpoint paths on disk**

Run (discover actual directory names):
```bash
cd /root/vast/eric/vnl-playground
for pattern in \
    "s13-ms-armM-anchorA-fs1p1-s2" \
    "s13-ms-armM-anchorA-fs1p4$" \
    "s13-ms-armM-anchorC-fs1p3$" \
    "s14-ms-anchorA-C7-t1p4b1p4-s1" \
    "s14-ms-anchorA-C4-" \
    "s12-hybrid.*fs1p0\|s12.*fs1p0" \
    "s11.*d5em7.*fs1p0\|s11.*fs1p0" \
    "s10-bridge-fs03-C-s1"
do
  ls checkpoints/ | grep -E "$pattern" | head -1
done
```
Expected: 8 directory names print, one per pattern. Copy them verbatim into the next step.

If any line is empty, search with a broader pattern (e.g. `ls checkpoints/ | grep s12 | grep fs1p0 | head -3`) and choose the best match by wandb Name or by the user's preference.

- [ ] **Step 9.2: Write the driver script**

Write `/root/vast/eric/vnl-playground/scripts/s15_stage2_eval.sh`:
```bash
#!/bin/bash
# Stage 2 of s15-ms: re-evaluate 8 frontier checkpoints with new EMG metrics at
# --emg-norm-percentile 98 (legacy) and 100 (new default). Outputs per-checkpoint
# CSV rows to plots/2026-04-23-s15-stage2/eval_matrix.csv.
set -o pipefail
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

OUT_DIR="plots/2026-04-23-s15-stage2"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/eval_matrix.csv"

# --- CHECKPOINTS (edit with the paths from Step 9.1) ---
CKPTS=(
    "checkpoints/<PASTE s13-ms-armM-anchorA-fs1p1-s2 DIRECTORY>"
    "checkpoints/<PASTE s13-ms-armM-anchorA-fs1p4 DIRECTORY>"
    "checkpoints/<PASTE s13-ms-armM-anchorC-fs1p3 DIRECTORY>"
    "checkpoints/<PASTE s14-ms-anchorA-C7 s1 DIRECTORY>"
    "checkpoints/<PASTE s14-ms-anchorA-C4 s1 DIRECTORY>"
    "checkpoints/<PASTE s12 fs1.0 DIRECTORY>"
    "checkpoints/<PASTE s11 d5em7 fs1.0 DIRECTORY>"
    "checkpoints/<PASTE s10 shape-king DIRECTORY>"
)

# CSV header
echo "checkpoint,norm_pct,muscle,mean_corr,mean_mae,trial_corr_mean,lagged_corr_max,phase_lag_ms,phase_lag_fwhm_steps,per_trial_lagged_corr_mean,per_trial_phase_lag_std_ms,trial_mae" > "${CSV}"

for CKPT in "${CKPTS[@]}"; do
  for PCT in 98 100; do
    LOG="${OUT_DIR}/eval_$(basename "${CKPT}")_p${PCT}.log"
    JSON="${OUT_DIR}/eval_$(basename "${CKPT}")_p${PCT}.json"
    echo "---- ${CKPT} @ p${PCT} ----"
    python scripts/emg_comparison.py \
        --checkpoint "${CKPT}" \
        --emg-norm-percentile "${PCT}" \
        --output-json "${JSON}" \
        2>&1 | tee "${LOG}"
    # Parse JSON → CSV rows (one row per muscle)
    python - <<PY >> "${CSV}"
import json, os
J = json.load(open("${JSON}"))
for muscle, m in J.get("metrics_by_muscle", {}).items():
    row = [
        os.path.basename("${CKPT}"), "${PCT}", muscle,
        m.get("mean_corr"), m.get("mean_mae"), m.get("trial_corr_mean"),
        m.get("lagged_corr_max"), m.get("phase_lag_ms"),
        m.get("lagged_corr_fwhm_steps"),
        m.get("per_trial_lagged_corr_mean"),
        m.get("per_trial_phase_lag_std_ms"), m.get("trial_mae"),
    ]
    print(",".join("" if v is None else f"{v}" for v in row))
PY
  done
done

echo "=== Stage 2 complete. CSV at ${CSV} ==="
```

- [ ] **Step 9.3: Add `--output-json` to emg_comparison.py if not present**

Run: `grep -n "output.json\|output-json" /root/vast/eric/vnl-playground/scripts/emg_comparison.py`
If missing, add to argparse:
```python
    parser.add_argument("--output-json", type=str, default=None,
                        help="If set, write metrics_by_muscle as JSON to this path.")
```
And near the end of the script (right before print statements or `return`), add:
```python
if args.output_json:
    import json
    with open(args.output_json, "w") as f:
        json.dump({"metrics_by_muscle": {m: v for m, v in metrics_by_muscle.items()}}, f,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
```
If `metrics_by_muscle` is named differently in the script, substitute the actual variable name (it was `metrics_by_muscle` at the call site around line 657).

- [ ] **Step 9.4: Make the driver script executable**

```bash
chmod +x /root/vast/eric/vnl-playground/scripts/s15_stage2_eval.sh
```

- [ ] **Step 9.5: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add scripts/s15_stage2_eval.sh scripts/emg_comparison.py
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "add Stage 2 driver script for s15-ms checkpoint re-evaluation"
```

---

## Task 10: Stage 2 — Execute the re-evaluation

**Files:**
- Output: `plots/2026-04-23-s15-stage2/eval_matrix.csv`, 16 log files (+ 16 JSON files)

- [ ] **Step 10.1: Run the driver**

```bash
cd /root/vast/eric/vnl-playground
bash scripts/s15_stage2_eval.sh 2>&1 | tee /tmp/s15_stage2.log
```
Expected: script completes in ~3 h. Final line `=== Stage 2 complete. CSV at plots/2026-04-23-s15-stage2/eval_matrix.csv ===`.

- [ ] **Step 10.2: Inspect the CSV**

```bash
cd /root/vast/eric/vnl-playground
cat plots/2026-04-23-s15-stage2/eval_matrix.csv | column -t -s','
```
Expected: 2 rows × 2 muscles × 8 checkpoints = 32 rows. Every row has finite numeric values; no empty fields.

- [ ] **Step 10.3: Commit the results**

```bash
cd /root/vast/eric/vnl-playground
git add plots/2026-04-23-s15-stage2/eval_matrix.csv
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "add Stage 2 eval_matrix.csv for 8 frontier checkpoints"
```

---

## Task 11: Stage 2 — Determine the Stage 3 branch

**Files:**
- Create: `docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md`

- [ ] **Step 11.1: Compute per-checkpoint summary rows**

Run this one-shot analysis:
```bash
cd /root/vast/eric/vnl-playground
.venv/bin/python <<'PY'
import pandas as pd
df = pd.read_csv("plots/2026-04-23-s15-stage2/eval_matrix.csv")
# Pivot to per-checkpoint rows (wide on muscle × norm_pct)
df["min_lagged"] = df.groupby(["checkpoint","norm_pct"])["lagged_corr_max"].transform("min")
df["max_mae"]    = df.groupby(["checkpoint","norm_pct"])["mean_mae"].transform("max")
df["max_lag_abs_ms"] = df.groupby(["checkpoint","norm_pct"])["phase_lag_ms"].transform(lambda s: s.abs().max())
wide = (df.drop_duplicates(["checkpoint","norm_pct"])
          [["checkpoint","norm_pct","min_lagged","max_mae","max_lag_abs_ms"]]
          .sort_values(["checkpoint","norm_pct"]))
print(wide.to_string(index=False))
wide.to_csv("plots/2026-04-23-s15-stage2/per_checkpoint_summary.csv", index=False)
PY
```

- [ ] **Step 11.2: Write the Stage 2 report**

Create `/root/vast/eric/vnl-playground/docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md`. Use this exact structure, filling in results from Step 11.1:

```markdown
# s15-ms Stage 2 Report

**Date:** 2026-04-23

## Per-checkpoint summary

| checkpoint | norm_pct | min(lagged_corr) | max(mean_mae) | max(|phase_lag_ms|) |
|---|---|---|---|---|
| ... (copy from per_checkpoint_summary.csv) | | | | |

## Branch decision

Walk the tree in order. First match wins.

- **Branch 1** — any row with `norm_pct=100`, `min(lagged_corr) ≥ 0.80`, `max(mean_mae) ≤ 0.15`, `max(|phase_lag_ms|) ≤ 20`: pick winning cell, go to Task 12A.
- **Branch 2** — any row with `norm_pct=100`, `min(lagged_corr) ≥ 0.80` but `max(|phase_lag_ms|) > 20`: go to Task 12B (no retraining).
- **Branch 4** — if per-trial-norm adjunct at any row shows `min(lagged_corr) ≥ norm_pct=100 value + 0.10`: go to Task 12D. (Run the per-trial probe in Step 11.3 before declaring.)
- **Branch 3** — otherwise: go to Task 12C.

## Decision: Branch <N>
## Target cell(s): <fill in>
```

- [ ] **Step 11.3: Run per-trial normalization adjunct probe (for Branch 4)**

```bash
cd /root/vast/eric/vnl-playground
# Re-run emg_comparison on the Stage-2 leader with a per-trial normalization
# monkey-patch (no trainer-side changes yet; this is the probe that gates Branch 4).
.venv/bin/python <<'PY'
import numpy as np, json, subprocess, os, sys
# Minimal adjunct: load the leader's checkpoint result JSON, re-normalize the
# reference per-trial inside emg_comparison by monkey-patching load_emg_reference.
# If implementing the probe via a script flag is more reliable, defer Branch 4
# probe to a real --emg-norm-mode per_trial flag in Task 12D (preferred if
# monkey-patch is hairy).
print("Per-trial probe is only needed if Branches 1-3 are all ambiguous; skip unless report demands it.")
PY
```
If Branches 1–3 give a clean verdict, skip this step and do not declare Branch 4.

- [ ] **Step 11.4: Commit the report**

```bash
cd /root/vast/eric/vnl-playground
git add docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md plots/2026-04-23-s15-stage2/per_checkpoint_summary.csv
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "Stage 2 report: branch decision from 8-checkpoint re-eval"
```

---

## Task 12A — Branch 1: Replicate the winner with 5 seeds

**Only execute if Stage 2 report declared Branch 1.**

**Files:**
- Create: `sweep_s15_ms_branch1.sh`
- Create: `S15_MS_LAUNCH.md`

- [ ] **Step 12A.1: Identify the winning cell from Task 11 report**

Read `docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md` "Target cell(s)". It names the winning checkpoint and its hyperparameters (force_scale, joint_damping, control_cost, control_diff_cost, joint_armature).

- [ ] **Step 12A.2: Write the sweep script**

Write `/root/vast/eric/vnl-playground/sweep_s15_ms_branch1.sh`:
```bash
#!/bin/bash
# s15-ms Branch 1: 5-seed replication of the Stage-2 winning cell under
# --emg-norm-percentile 100. 5 runs, one per seed.
set -o pipefail
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-branch1"
# --- FILL IN FROM TASK 11 REPORT ---
FS=<winner_force_scale>
DAMP=<winner_joint_damping>
CC=<winner_control_cost>
CDC=<winner_control_diff_cost>
ARM=<winner_joint_armature>
ANCHOR_TAG=<winner_anchor_tag e.g. anchorA>

BASE_ARGS=(
    --ctrl-dt 0.0025 --sim-dt 0.00125 --episode-length 100 --qvel-init zeros
    --joints-weight 5.0 --joints-vel-weight 0.5 --wrist-pos-weight 0.1
    --bodies-pos-weight 0.1 --num-timesteps 800000000 --num-evals 8
    --joint-armature "${ARM}" --joint-damping "${DAMP}"
    --control-cost "${CC}" --control-diff-cost "${CDC}"
    --force-scale "${FS}" --emg-norm-percentile 100
    --wandb-group "${WANDB_GROUP}"
)
CRASHED=(); OK=()
for SEED in 1 2 3 4 5; do
    TAG="s15-ms-${ANCHOR_TAG}-fs${FS}-s${SEED}"
    RUN_NAME="${TAG}-$(date +%Y%m%d-%H%M%S)"
    LOG="/tmp/sweep_s15_ms_branch1_s${SEED}.log"
    echo "---- [${SEED}/5] ${RUN_NAME} ----"
    if python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" --seed "${SEED}" \
        --wandb-tags s15-ms branch1 "${ANCHOR_TAG}" "fs${FS}" "seed${SEED}" p100 2>&1 | tee "${LOG}"; then
      OK+=("${RUN_NAME}")
    else
      CRASHED+=("${RUN_NAME}")
    fi
done
echo "OK: ${#OK[@]}; CRASHED: ${#CRASHED[@]}"
```

- [ ] **Step 12A.3: Write S15_MS_LAUNCH.md**

Write `/root/vast/eric/vnl-playground/S15_MS_LAUNCH.md`:
```markdown
# S15-MS Branch 1 Launch Commands

5-seed replication of Stage 2 winner under new EMG metrics (`--emg-norm-percentile 100`).
Spec: `docs/superpowers/specs/2026-04-23-s15-ms-design.md`.
Stage 2 report: `docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md`.

## Launch

Single GPU, serial (one run at a time). ~5 × ~2 h = ~10 h wall-clock.

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_master.log 2>&1 &
```

To parallelize across 3 GPUs, split seeds {1,2}, {3,4}, {5} into 3 files and launch each on a separate GPU.

## Success gates (median across 5 seeds)

See s15-ms design success criteria — `lagged_corr_max ≥ 0.80` both muscles, `R ≥ 400`, `|phase_lag_ms| ≤ 20`, `mae ≤ 0.15` both muscles.
```

- [ ] **Step 12A.4: Commit and launch**

```bash
cd /root/vast/eric/vnl-playground
chmod +x sweep_s15_ms_branch1.sh
git add sweep_s15_ms_branch1.sh S15_MS_LAUNCH.md
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "add Branch 1 sweep script and launch doc for s15-ms winner replication"
# Launch in background:
CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_master.log 2>&1 &
echo $!
```
Expected: returns a PID; `tail -f /tmp/s15_ms_branch1_master.log` shows training starting.

---

## Task 12B — Branch 2: Declare shape success, document the systematic lag

**Only execute if Stage 2 report declared Branch 2.**

**Files:**
- Modify: `docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md`

- [ ] **Step 12B.1: Verify lag is consistent across checkpoints**

Run:
```bash
cd /root/vast/eric/vnl-playground
.venv/bin/python <<'PY'
import pandas as pd
df = pd.read_csv("plots/2026-04-23-s15-stage2/eval_matrix.csv")
p100 = df[df.norm_pct == 100]
for muscle in p100.muscle.unique():
    s = p100[p100.muscle == muscle]
    print(f"[{muscle}] phase_lag_ms: mean={s.phase_lag_ms.mean():.1f} std={s.phase_lag_ms.std():.1f} "
          f"range=[{s.phase_lag_ms.min():.1f}, {s.phase_lag_ms.max():.1f}]")
PY
```
Expected: if Branch 2 is real, `std` is small (< 5 ms) and `mean` is far from zero — means the lag is a constant reference offset, not a policy flaw.

- [ ] **Step 12B.2: Shift-clip-start verification**

Re-run emg_comparison on one checkpoint with `--clip-start-frame <measured_mean_lag_in_frames>`. The new `lagged_corr_max` should be very close to the new `mean_corr` (lag ≈ 0 after shift).

```bash
# Find a reasonable shift in frames (EMG is 30 kHz, ctrl_dt is 2.5 ms):
# frames = phase_lag_ms * 30000 / 1000.  round to nearest int.
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate
python scripts/emg_comparison.py \
  --checkpoint <leader_ckpt_from_stage2> \
  --emg-norm-percentile 100 \
  --clip-start-frame <measured_frame_shift> \
  --output-json plots/2026-04-23-s15-stage2/branch2_shift_test.json
```
If `--clip-start-frame` is not already a CLI flag of `emg_comparison.py`, skip this step and note in the report.

- [ ] **Step 12B.3: Append conclusion to Stage 2 report**

Append to `docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md`:
```markdown
## Branch 2 conclusion

The s13 frontier already achieves `lagged_corr_max ≥ 0.80` on both muscles at p100.
The residual `phase_lag_ms = <FILL>` is consistent across <N> checkpoints (std <FILL>),
indicating a fixed reference-alignment offset rather than a policy failure. The
policy's shape is correct; the measured lag is the metric's way of reporting an
offset between the bio reference window and the sim episode window.

**s15 declared successful on shape.** Recommended lookup: audit `clip_start_frame`
handling in `load_emg_reference`. No retraining necessary.
```

- [ ] **Step 12B.4: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "Branch 2 conclusion: s15-ms shape success, lag is fixed reference offset"
```

---

## Task 12C — Branch 3: Shape-cap confirmed, replicate + shoulder-damping mini-scan

**Only execute if Stage 2 report declared Branch 3.**

**Files:**
- Create: `sweep_s15_ms_branch3a_replicate.sh`
- Create: `sweep_s15_ms_branch3b_shoulder.sh`
- Create: `S15_MS_LAUNCH.md`

- [ ] **Step 12C.1: Write the replicate script (10 runs)**

Write `/root/vast/eric/vnl-playground/sweep_s15_ms_branch3a_replicate.sh`:
```bash
#!/bin/bash
# s15-ms Branch 3a: 5 seeds × top 2 s13 cells under p100. 10 runs.
set -o pipefail
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-branch3a"
BASE_ARGS=(
    --ctrl-dt 0.0025 --sim-dt 0.00125 --episode-length 100 --qvel-init zeros
    --joints-weight 5.0 --joints-vel-weight 0.5 --wrist-pos-weight 0.1
    --bodies-pos-weight 0.1 --num-timesteps 800000000 --num-evals 8
    --joint-armature 4e-10 --emg-norm-percentile 100
    --wandb-group "${WANDB_GROUP}"
)

run() {
    local TAG="$1" FS="$2" DAMP="$3" CC="$4" CDC="$5" SEED="$6"
    local RUN_NAME="${TAG}-s${SEED}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/${RUN_NAME}.log"
    echo "---- ${RUN_NAME} ----"
    python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --joint-damping "${DAMP}" --control-cost "${CC}" --control-diff-cost "${CDC}" \
        --force-scale "${FS}" --seed "${SEED}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" \
        --wandb-tags s15-ms branch3a p100 "seed${SEED}" 2>&1 | tee "${LOG}"
}

# Top s13 cell #1: anchor-A fs=1.1
for SEED in 1 2 3 4 5; do
    run "s15-ms-branch3a-anchorA-fs1p1" 1.1 9e-7 0.025 0.025 "${SEED}"
done
# Top s13 cell #2: anchor-C fs=1.3
for SEED in 1 2 3 4 5; do
    run "s15-ms-branch3a-anchorC-fs1p3" 1.3 1e-6 0.035 0.0 "${SEED}"
done
```

- [ ] **Step 12C.2: Write the shoulder mini-scan script (12 runs)**

Write `/root/vast/eric/vnl-playground/sweep_s15_ms_branch3b_shoulder.sh`:
```bash
#!/bin/bash
# s15-ms Branch 3b: fix fs=1.1 at anchor A; vary --shoulder-damping across 4
# levels, 3 seeds each. 12 runs.
set -o pipefail
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-branch3b"
BASE_ARGS=(
    --ctrl-dt 0.0025 --sim-dt 0.00125 --episode-length 100 --qvel-init zeros
    --joints-weight 5.0 --joints-vel-weight 0.5 --wrist-pos-weight 0.1
    --bodies-pos-weight 0.1 --num-timesteps 800000000 --num-evals 8
    --joint-armature 4e-10 --joint-damping 9e-7
    --control-cost 0.025 --control-diff-cost 0.025
    --force-scale 1.1 --emg-norm-percentile 100
    --wandb-group "${WANDB_GROUP}"
)

for SHOULDER in 3e-7 6e-7 9e-7 1.2e-6; do
    for SEED in 1 2 3; do
        TAG="s15-ms-branch3b-shd${SHOULDER}-s${SEED}"
        RUN_NAME="${TAG}-$(date +%Y%m%d-%H%M%S)"
        LOG="/tmp/${RUN_NAME}.log"
        echo "---- ${RUN_NAME} ----"
        python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
            --shoulder-damping "${SHOULDER}" --seed "${SEED}" \
            --tag "${TAG}" --run-name "${RUN_NAME}" \
            --wandb-tags s15-ms branch3b p100 "shd${SHOULDER}" "seed${SEED}" 2>&1 | tee "${LOG}"
    done
done
```

- [ ] **Step 12C.3: Write S15_MS_LAUNCH.md (Branch 3)**

Write `/root/vast/eric/vnl-playground/S15_MS_LAUNCH.md`:
```markdown
# S15-MS Branch 3 Launch (replicate + shoulder-damping scan)

22 runs total. On 4 GPUs in parallel, plan ~11 h wall-clock.

## Recommended partition

| GPU | Script | Runs |
|---|---|---|
| 0 | sweep_s15_ms_branch3a_replicate.sh (first 5 seeds of anchor-A) | 5 |
| 1 | sweep_s15_ms_branch3a_replicate.sh (last 5 seeds of anchor-C) | 5 |
| 2 | sweep_s15_ms_branch3b_shoulder.sh (first 6 cells: shd=3e-7, 6e-7) | 6 |
| 3 | sweep_s15_ms_branch3b_shoulder.sh (last 6 cells: shd=9e-7, 1.2e-6) | 6 |

(Split via editing each script or via `SGES=1,2,3` env-var filter — keep it simple: duplicate each script and strip unwanted cells.)

## Single-GPU serial fallback

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_branch3a_replicate.sh > /tmp/s15_branch3a.log 2>&1 &
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s15_ms_branch3b_shoulder.sh > /tmp/s15_branch3b.log 2>&1 &
```
```

- [ ] **Step 12C.4: Commit and launch**

```bash
cd /root/vast/eric/vnl-playground
chmod +x sweep_s15_ms_branch3a_replicate.sh sweep_s15_ms_branch3b_shoulder.sh
git add sweep_s15_ms_branch3a_replicate.sh sweep_s15_ms_branch3b_shoulder.sh S15_MS_LAUNCH.md
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "add Branch 3 sweep scripts and launch doc for s15-ms (replicate + shoulder scan)"
```

Launch per the partition above. Expected: 22 runs complete within 12 h on ≥4 GPUs.

---

## Task 12D — Branch 4: Per-trial normalization fix

**Only execute if Stage 2 report declared Branch 4.**

**Files:**
- Modify: `train_mouse_janelia_sigmoid_moving_shoulder.py` (add `--emg-norm-mode`)
- Modify: `scripts/emg_comparison.py` (add same flag)
- Modify: `vnl_playground/eval_metrics/emg.py` (add per-trial normalization helper)
- Create: `sweep_s15_ms_branch4.sh`

- [ ] **Step 12D.1: Add `--emg-norm-mode` flag to the trainer**

In the trainer argparse near the existing `--emg-norm-percentile`, add:
```python
    p.add_argument("--emg-norm-mode", type=str, default="dataset",
                   choices=["dataset", "per_trial"],
                   help="Reference EMG normalization scope. 'dataset' = divide by "
                        "percentile of all samples (current). 'per_trial' = divide "
                        "each trial by its own percentile.")
```

- [ ] **Step 12D.2: Implement per-trial path in `load_emg_reference`**

In `load_emg_reference`, after line 142 (`arr = np.array(envelopes)`), replace the existing normalization with:
```python
            if norm_mode == "per_trial":
                denom = np.percentile(arr, norm_percentile, axis=1, keepdims=True)
                denom = np.where(denom > 0, denom, 1.0)
                emg_by_muscle[muscle_name] = arr / denom
            else:  # "dataset" (current behavior)
                emg_by_muscle[muscle_name] = arr / np.percentile(arr, norm_percentile)
```

Add `norm_mode: str = "dataset"` to `load_emg_reference`'s signature and thread it from the trainer call site (line ~1505).

- [ ] **Step 12D.3: Mirror the change in `scripts/emg_comparison.py`**

Add the same `--emg-norm-mode` flag and normalization logic in the eval-replay script.

- [ ] **Step 12D.4: Write the sweep script (6 runs)**

Write `/root/vast/eric/vnl-playground/sweep_s15_ms_branch4.sh`:
```bash
#!/bin/bash
# s15-ms Branch 4: top 2 s13 cells under --emg-norm-mode per_trial, 3 seeds.
set -o pipefail
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-branch4"
BASE_ARGS=(
    --ctrl-dt 0.0025 --sim-dt 0.00125 --episode-length 100 --qvel-init zeros
    --joints-weight 5.0 --joints-vel-weight 0.5 --wrist-pos-weight 0.1
    --bodies-pos-weight 0.1 --num-timesteps 800000000 --num-evals 8
    --joint-armature 4e-10 --emg-norm-percentile 100 --emg-norm-mode per_trial
    --wandb-group "${WANDB_GROUP}"
)

run() {
    local TAG="$1" FS="$2" DAMP="$3" CC="$4" CDC="$5" SEED="$6"
    local RUN_NAME="${TAG}-s${SEED}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/${RUN_NAME}.log"
    python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --joint-damping "${DAMP}" --control-cost "${CC}" --control-diff-cost "${CDC}" \
        --force-scale "${FS}" --seed "${SEED}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" \
        --wandb-tags s15-ms branch4 per_trial "seed${SEED}" 2>&1 | tee "${LOG}"
}

for SEED in 1 2 3; do
    run "s15-ms-branch4-anchorA-fs1p1" 1.1 9e-7 0.025 0.025 "${SEED}"
    run "s15-ms-branch4-anchorC-fs1p3" 1.3 1e-6 0.035 0.0 "${SEED}"
done
```

- [ ] **Step 12D.5: Commit and launch**

```bash
cd /root/vast/eric/vnl-playground
chmod +x sweep_s15_ms_branch4.sh
git add train_mouse_janelia_sigmoid_moving_shoulder.py scripts/emg_comparison.py \
        vnl_playground/eval_metrics/emg.py sweep_s15_ms_branch4.sh
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "Branch 4: add --emg-norm-mode per_trial and 6-run retrain script"
# Launch:
CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_branch4.sh > /tmp/s15_branch4.log 2>&1 &
```

---

## Task 13: Final analysis and s15 writeup

**Files:**
- Create: `plots/2026-04-23-s15-final/`
- Create: `docs/superpowers/specs/2026-04-23-s15-ms-results.md`

- [ ] **Step 13.1: Pull the s15 runs from wandb**

```bash
cd /root/vast/eric/vnl-playground
.venv/bin/python <<'PY'
import os, wandb
os.makedirs("plots/2026-04-23-s15-final", exist_ok=True)
api = wandb.Api()
# Adjust entity/project if different — check recent runs for the correct path.
runs = api.runs("eric-leonardis/vnl-playground", {"tags": "s15-ms"})
rows = []
for r in runs:
    summary = {k: v for k, v in r.summary.items() if not k.startswith("_")}
    row = {"name": r.name, "state": r.state, **summary}
    rows.append(row)
import pandas as pd
pd.DataFrame(rows).to_csv("plots/2026-04-23-s15-final/s15_runs.csv", index=False)
print(f"Wrote {len(rows)} runs")
PY
```

- [ ] **Step 13.2: Apply the s15 success gates**

```bash
cd /root/vast/eric/vnl-playground
.venv/bin/python <<'PY'
import pandas as pd
df = pd.read_csv("plots/2026-04-23-s15-final/s15_runs.csv")
g = df.query(
    "`eval/episode_reward` >= 400 "
    "and `eval/emg_biceps_lagged_corr_max` >= 0.80 "
    "and `eval/emg_triceps_lagged_corr_max` >= 0.80 "
    "and `eval/emg_biceps_trial_corr_mean` >= 0.5 "
    "and `eval/emg_triceps_trial_corr_mean` >= 0.5 "
    "and abs(`eval/emg_biceps_phase_lag_ms`) <= 20 "
    "and abs(`eval/emg_triceps_phase_lag_ms`) <= 20 "
    "and `eval/emg_biceps_mae` <= 0.15 "
    "and `eval/emg_triceps_mae` <= 0.15"
)
print("Winners (on per-run metrics):")
print(g[["name","eval/episode_reward",
        "eval/emg_biceps_lagged_corr_max","eval/emg_triceps_lagged_corr_max",
        "eval/emg_biceps_phase_lag_ms","eval/emg_triceps_phase_lag_ms"]].to_string(index=False))
# For the per-cell median, group by tag prefix (strip seed + timestamp suffix).
df["cell"] = df["name"].str.replace(r"-s\d+-\d{8}-\d{6}$", "", regex=True)
med = df.groupby("cell")[["eval/episode_reward",
                          "eval/emg_biceps_lagged_corr_max",
                          "eval/emg_triceps_lagged_corr_max"]].median()
med.to_csv("plots/2026-04-23-s15-final/s15_medians_by_cell.csv")
print(med)
PY
```

- [ ] **Step 13.3: Write the results doc**

Write `docs/superpowers/specs/2026-04-23-s15-ms-results.md` with sections: Outcome (shipped / not), Winning cell, Median metrics, Comparison vs s13 top cell, Known residual issues. Reference `plots/2026-04-23-s15-final/s15_medians_by_cell.csv`.

- [ ] **Step 13.4: Commit**

```bash
cd /root/vast/eric/vnl-playground
git add plots/2026-04-23-s15-final docs/superpowers/specs/2026-04-23-s15-ms-results.md
git -c user.email="eric@talmolab.org" -c user.name="Eric Leonardis" commit -m "s15-ms final results and winners analysis"
```

---

## Self-review checklist (for the engineer running the plan)

Before declaring s15 complete:

- [ ] `pytest tests/test_emg_metrics.py -v` passes (11 tests).
- [ ] `grep -n "np.percentile(arr, 98)" train_mouse_janelia_sigmoid_moving_shoulder.py scripts/emg_comparison.py` returns no literal 98 (was replaced by `args.emg_norm_percentile`).
- [ ] `grep -n "compute_emg_metrics\b" train_mouse_janelia_sigmoid_moving_shoulder.py` returns no local definition (only imports from shared module).
- [ ] Stage 2 `eval_matrix.csv` has 32 non-empty rows.
- [ ] `docs/superpowers/specs/2026-04-23-s15-ms-stage2-report.md` declares exactly one branch.
- [ ] Exactly one of `sweep_s15_ms_branch1.sh`, `sweep_s15_ms_branch3a_replicate.sh`+`sweep_s15_ms_branch3b_shoulder.sh`, `sweep_s15_ms_branch4.sh` exists in the repo (Branch 2 produces none).
- [ ] Final results doc `docs/superpowers/specs/2026-04-23-s15-ms-results.md` cites specific run names for the winner.

## Constraints and invariants (do not violate)

- **No changes to the PPO loss, task reward, or cost terms.** All new metrics are eval-side. If a branch tempts you to add a reward term, stop and escalate — the spec forbids it.
- **No new XMLs added to git.** User memory: mouse model XMLs stay untracked. If a branch requires a new XML, copy it to `vnl_playground/tasks/mouse/xmls/` but do not `git add` it.
- **Checkpoint config.json is unreliable.** When resolving checkpoint hyperparameters, prefer the wandb run's Command / `_fields.*` summary over `checkpoints/<name>/config.json`.
- **All commits use the existing author-override pattern** (`-c user.email="eric@talmolab.org" -c user.name="Eric Leonardis"`). Do not run `git config` to change global identity.
- **Commit messages: plain imperatives.** Do not lead with `s15:` or `s15-ms:` prefixes.
