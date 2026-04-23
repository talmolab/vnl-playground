"""Unit tests for vnl_playground.eval_metrics.emg."""
from __future__ import annotations

import numpy as np
import pytest


def test_module_imports():
    from vnl_playground.eval_metrics import emg
    assert emg.LAG_RANGE_STEPS_DEFAULT == 20


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
    assert np.isfinite(m["trial_corr_mean"])


def test_per_trial_metrics_handles_zero_variance_trial():
    from vnl_playground.eval_metrics import emg
    sim = np.zeros((2, 30))
    sim[1] = np.sin(np.linspace(0, 2 * np.pi, 30))
    bio = np.sin(np.linspace(0, 2 * np.pi, 30))[np.newaxis].repeat(2, axis=0)
    m = emg.compute_per_trial_metrics(sim, bio, ctrl_dt_ms=2.5)
    assert np.isfinite(m["trial_corr_mean"])
