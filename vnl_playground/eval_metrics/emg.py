"""EMG evaluation metrics shared by the trainer and the eval-replay script.

Pure-numpy implementations; no jax, no mujoco, no brax. All functions accept
(n_trials, T) arrays for sim and bio traces and return plain Python floats /
ints suitable for wandb logging.
"""
from __future__ import annotations

import numpy as np

LAG_RANGE_STEPS_DEFAULT = 20  # ±20 steps × ctrl-dt = ±50 ms at ctrl_dt=2.5 ms


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
