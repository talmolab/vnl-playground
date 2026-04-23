"""EMG evaluation metrics shared by the trainer and the eval-replay script.

Pure-numpy implementations; no jax, no mujoco, no brax. All functions accept
(n_trials, T) arrays for sim and bio traces and return plain Python floats /
ints suitable for wandb logging.
"""
from __future__ import annotations

import numpy as np

LAG_RANGE_STEPS_DEFAULT = 20  # ±20 steps × ctrl-dt = ±50 ms at ctrl_dt=2.5 ms


def _lag_scan(sim_mean, bio_mean, lag_range_steps):
    """Return (best_r, best_lag, corrs, lags). Shared by compute_lag_metrics and
    compute_per_trial_metrics. No ms conversion, no dict building — pure math.

    Returns (nan, 0, corrs_all_nan, lags) if every slice was zero-variance.
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
        return float("nan"), 0, corrs, lags

    argmax = int(np.nanargmax(corrs))
    best_r = float(corrs[argmax])
    best_lag = int(lags[argmax])
    return best_r, best_lag, corrs, lags


def compute_lag_metrics(sim_mean, bio_mean, ctrl_dt_ms: float = 2.5,
                        lag_range_steps: int = LAG_RANGE_STEPS_DEFAULT) -> dict:
    """Cross-correlation with lag over ±lag_range_steps steps.

    Takes 1-D sim_mean and bio_mean (trial-averaged traces). Returns dict with:
      lagged_corr_max, phase_lag_steps, phase_lag_ms,
      lagged_corr_at_0, lagged_corr_at_neg5, lagged_corr_at_pos5,
      lagged_corr_fwhm_steps, lagged_corr_edge_saturated.

    Positive phase_lag_steps means sim leads bio by that many steps (i.e., the
    best match is at bio[lag:] vs sim[:-lag]).

    `lagged_corr_edge_saturated` is 1 if argmax lag is pinned to ±lag_range_steps
    (signal that true lag likely lies outside window), else 0.
    """
    best_r, best_lag, corrs, lags = _lag_scan(sim_mean, bio_mean, lag_range_steps)

    if np.all(np.isnan(corrs)):
        return {
            "lagged_corr_max": float("nan"),
            "phase_lag_steps": 0,
            "phase_lag_ms": 0.0,
            "lagged_corr_at_0": float("nan"),
            "lagged_corr_at_neg5": float("nan"),
            "lagged_corr_at_pos5": float("nan"),
            "lagged_corr_fwhm_steps": 0,
            "lagged_corr_edge_saturated": 0,
        }

    half_max = max(best_r / 2.0, 0.0)
    above = np.nan_to_num(corrs, nan=-np.inf) >= half_max
    fwhm_steps = int(np.sum(above))
    edge_saturated = int(abs(best_lag) == lag_range_steps)

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
        "lagged_corr_edge_saturated": edge_saturated,
    }


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
            best_r, best_lag, _, _ = _lag_scan(s, b, lag_range_steps)
            if np.isfinite(best_r):
                per_trial_lagged_corrs.append(best_r)
                per_trial_lags_steps.append(best_lag)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def _median(xs):
        return float(np.median(xs)) if xs else float("nan")

    return {
        "trial_corr_mean": _mean(trial_corrs),
        "trial_corr_median": _median(trial_corrs),
        "trial_mae": _mean(trial_maes),
        "per_trial_lagged_corr_mean": _mean(per_trial_lagged_corrs),
        "per_trial_lagged_corr_median": _median(per_trial_lagged_corrs),
        "per_trial_phase_lag_mean_ms": float(np.mean(per_trial_lags_steps) * ctrl_dt_ms) if per_trial_lags_steps else float("nan"),
        "per_trial_phase_lag_std_ms": float(np.std(per_trial_lags_steps) * ctrl_dt_ms) if per_trial_lags_steps else float("nan"),
    }


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
