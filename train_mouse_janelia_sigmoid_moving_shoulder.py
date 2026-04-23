"""Janelia mouse forelimb imitation — Sigmoid-Normal with IK-driven moving shoulder.

Identical to train_mouse_janelia_sigmoid_normal.py in policy, distribution, and
PPO hyperparams. Differences:
  - Uses mouse_forelimb_right_moving_shoulder_ik.xml (adds sh_tx/ty/tz slide
    joints to the clavicle; muscles unchanged; no shoulder-translation actuators).
  - Uses reference_data_moving_shoulder/ (STAC v16 IK clips with 7-dim qpos).
  - Env is MouseImitationMovingShoulder: snaps qpos[:3] and qvel[:3] to the IK
    reference after every step, masks those dims out of joints/joints_vel
    rewards and pose_error termination.

Rationale: freezing the shoulder at the origin shifts triceps burst timing
relative to biology. Kinematically driving the shoulder from IK removes that
confound so the muscle policy learns the correct onset.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from datetime import datetime
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jp
import mujoco
import optax
import wandb
from brax.training import distribution
from brax.training.distribution import ParametricDistribution
from brax.training import networks
from brax.training.acme import running_statistics
from etils import epath
from flax.training import orbax_utils
from flax import linen
from orbax import checkpoint as ocp
from ml_collections import config_dict
from pprint import pprint

from mujoco_playground import wrapper

from vnl_playground.tasks.mouse.imitation_moving_shoulder import (
    MouseImitationMovingShoulder,
    default_config,
)
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.mouse.consts import (
    JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH,
    MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# =============================================================================
# EMG comparison constants and helpers
# =============================================================================

EMG_DIR = "/root/vast/eric/mouse-reach-mjx-neurips/emg"
TRIAL_CSV = "/root/vast/eric/mouse-reach-mjx-neurips/trial_info/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv"

EMG_MUSCLE_CONFIGS = [
    (5, "Triceps_Lateral", f"{EMG_DIR}/emg_triceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv", "Triceps"),
    (8, "Biceps_Long", f"{EMG_DIR}/emg_biceps_fixed_A36-1_2023-07-18_16-54-01_lightOff_tone_on.csv", "Biceps"),
]

EMG_DURATION_MS = 250
EMG_CTRL_DT = 0.0025


def load_emg_reference(n_clips, target_timesteps, clip_start_frame=0,
                       norm_percentile: float = 100.0):
    """Pre-process biological EMG data. Called once at startup.

    clip_start_frame: the in-clip mocap-frame index (200 Hz) where the sim
    rollout begins. Shifts the bio EMG window by the same amount so bio and
    sim are time-aligned. Default 0 preserves the legacy behavior.
    """
    try:
        trial_info = pd.read_csv(TRIAL_CSV)
        valid_mask = ~((trial_info["start"] == 0) & (trial_info["end"] == 0))
        valid_trials = trial_info[valid_mask]
    except FileNotFoundError:
        print("  EMG: trial CSV not found, skipping EMG comparison")
        return None

    emg_by_muscle = {}
    for sim_idx, sim_name, emg_file, muscle_name in EMG_MUSCLE_CONFIGS:
        try:
            emg_data = pd.read_csv(emg_file, header=None)
        except FileNotFoundError:
            print(f"  EMG: {emg_file} not found, skipping {muscle_name}")
            continue

        fs = 30000
        emg_duration_samples = int(EMG_DURATION_MS / 1000 * fs)
        envelopes = []

        for i, (idx, row) in enumerate(valid_trials.iterrows()):
            if i >= n_clips:
                break
            emg_start = int(1 / 200 * (row["start"] + clip_start_frame) * 30000)
            emg_end = emg_start + emg_duration_samples
            if idx >= len(emg_data) or emg_start >= 90000 or emg_end > 90000:
                continue

            trial_emg = emg_data.iloc[idx, :].values.astype(float)
            b, a = signal.butter(4, [20, 1000], btype="bandpass", fs=fs)
            filtered = signal.filtfilt(b, a, trial_emg)
            b_env, a_env = signal.butter(4, 50, btype="lowpass", fs=fs)
            envelope = signal.filtfilt(b_env, a_env, np.abs(filtered))

            reach_env = envelope[emg_start:emg_end]
            if len(reach_env) > 0:
                resampled = np.interp(
                    np.linspace(0, 1, target_timesteps),
                    np.linspace(0, 1, len(reach_env)),
                    reach_env,
                )
                envelopes.append(resampled)

        if envelopes:
            arr = np.array(envelopes)
            emg_by_muscle[muscle_name] = arr / np.percentile(arr, norm_percentile)

    if not emg_by_muscle:
        return None

    # Precompute mean + SEM (these never change)
    emg_means = {}
    emg_sems = {}
    for muscle_name, traces in emg_by_muscle.items():
        emg_means[muscle_name] = traces.mean(axis=0)
        emg_sems[muscle_name] = traces.std(axis=0) / np.sqrt(traces.shape[0])

    print(f"  EMG: loaded {', '.join(f'{k}({v.shape[0]} trials)' for k, v in emg_by_muscle.items())}")
    return {"traces": emg_by_muscle, "means": emg_means, "sems": emg_sems}


def compute_emg_metrics(sim_muscle, emg_mean_trace, bio_traces=None):
    """Correlation and MAE metrics for simulated vs biological EMG.

    Returns:
        mean_corr: correlation between sim mean and bio mean
        mean_mae: MAE between sim mean and bio mean (mean-of-means)
        trial_mae: MAE computed trial-by-trial (sim_i - bio_i), averaged
                   across all paired trials and timesteps. None if no
                   bio_traces provided.
    """
    sim_mean = sim_muscle.mean(axis=0)
    mean_corr = float(np.corrcoef(sim_mean, emg_mean_trace)[0, 1])
    mean_mae = float(np.mean(np.abs(sim_mean - emg_mean_trace)))
    result = {
        "mean_corr": mean_corr,
        "mean_mae": mean_mae,
    }
    if bio_traces is not None:
        n_pairs = min(sim_muscle.shape[0], bio_traces.shape[0])
        T = min(sim_muscle.shape[1], bio_traces.shape[1])
        # Per-trial absolute error, then average across trials and time
        trial_errors = np.abs(sim_muscle[:n_pairs, :T] - bio_traces[:n_pairs, :T])
        result["trial_mae"] = float(np.mean(trial_errors))
    return result


def plot_emg_error_fig(sim_actions, emg_ref, target_timesteps, ctrl_dt):
    """Per-timestep EMG error (sim - bio) with SEM bands."""
    time_axis = np.linspace(0, target_timesteps * ctrl_dt, target_timesteps)
    n_muscles = len(EMG_MUSCLE_CONFIGS)
    fig, axes = plt.subplots(1, n_muscles, figsize=(6 * n_muscles, 4))
    if n_muscles == 1:
        axes = [axes]
    for ax, (sim_idx, sim_name, _, muscle_name) in zip(axes, EMG_MUSCLE_CONFIGS):
        emg_mean = emg_ref["means"].get(muscle_name)
        if emg_mean is None:
            ax.set_title(f"{muscle_name} - no EMG data")
            continue
        sim_muscle = sim_actions[:, :, sim_idx]  # (n_clips, T)
        # Per-trial error, then stats across trials
        errors = sim_muscle - emg_mean[np.newaxis, :]  # (n_clips, T)
        err_mean = errors.mean(axis=0)
        err_sem = errors.std(axis=0) / np.sqrt(errors.shape[0])
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.plot(time_axis, err_mean, color="#d62728", linewidth=2)
        ax.fill_between(time_axis, err_mean - err_sem, err_mean + err_sem,
                        color="#d62728", alpha=0.25)
        ax.set_title(f"{muscle_name} error (sim − bio)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error")
        ax.set_ylim(-1, 1)
    plt.tight_layout()
    return fig


def plot_action_spectrum_fig(all_actions, ctrl_dt):
    """Log-log power spectrum of actions averaged over trials, per actuator.

    all_actions: (n_clips, T, n_actuators)
    """
    n_clips, T, n_act = all_actions.shape
    if T < 4:
        return None
    freqs = np.fft.rfftfreq(T, d=ctrl_dt)[1:]  # skip DC
    # Power spectrum per trial per actuator, then mean across trials
    spectra = np.abs(np.fft.rfft(all_actions, axis=1)) ** 2  # (n_clips, T//2+1, n_act)
    mean_spectra = spectra.mean(axis=0)[1:, :]  # drop DC, (n_freqs, n_act)
    # Mean across all actuators for a summary line
    mean_all = mean_spectra.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-actuator spectra (light lines) + mean (bold)
    ax = axes[0]
    for j in range(n_act):
        ax.loglog(freqs, mean_spectra[:, j], alpha=0.15, linewidth=0.5, color="#1f77b4")
    ax.loglog(freqs, mean_all, linewidth=2.5, color="#d62728", label="mean across actuators")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Action power spectrum (log-log)")
    nyquist = 1 / (2 * ctrl_dt)
    ax.axvline(nyquist, color="gray", linewidth=0.5, linestyle="--", label=f"Nyquist ({nyquist:.0f} Hz)")
    ax.legend(fontsize=8)

    # Right: cumulative power fraction (what % of power is above freq f?)
    ax2 = axes[1]
    total_power = mean_all.sum()
    if total_power > 0:
        cumulative_high = np.cumsum(mean_all[::-1])[::-1] / total_power
        ax2.semilogx(freqs, cumulative_high, linewidth=2, color="#2ca02c")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Fraction of power above f")
    ax2.set_title("Cumulative high-frequency power")
    ax2.set_ylim(0, 1)
    ax2.axvline(nyquist, color="gray", linewidth=0.5, linestyle="--")
    # Mark some reference frequencies
    for ref_f, label in [(10, "10 Hz"), (50, "50 Hz"), (100, "100 Hz")]:
        if ref_f < nyquist:
            ax2.axvline(ref_f, color="orange", linewidth=0.5, linestyle=":", alpha=0.5)
            ax2.text(ref_f * 1.05, 0.95, label, fontsize=7, color="orange")

    plt.tight_layout()
    return fig


def compute_spectral_metrics(all_actions, ctrl_dt):
    """Scalar metrics for high-frequency action content."""
    n_clips, T, n_act = all_actions.shape
    if T < 4:
        return {}
    freqs = np.fft.rfftfreq(T, d=ctrl_dt)[1:]
    spectra = np.abs(np.fft.rfft(all_actions, axis=1)) ** 2
    mean_spectra = spectra.mean(axis=0)[1:, :]  # (n_freqs, n_act)
    mean_all = mean_spectra.mean(axis=1)
    total = mean_all.sum()
    if total == 0:
        return {}
    nyquist = 1 / (2 * ctrl_dt)
    metrics = {}
    # Fraction of power above various frequency thresholds
    for thresh_hz in [10, 25, 50]:
        if thresh_hz < nyquist:
            mask = freqs >= thresh_hz
            metrics[f"eval/action_power_above_{thresh_hz}hz"] = float(mean_all[mask].sum() / total)
    # Spectral centroid (mean frequency weighted by power)
    metrics["eval/action_spectral_centroid_hz"] = float(np.sum(freqs * mean_all) / total)
    return metrics


def plot_emg_comparison_fig(sim_actions, emg_ref, metrics_by_muscle, target_timesteps,
                           ctrl_dt=0.0025):
    """EMG comparison plot for wandb logging. Uses precomputed EMG means/sems."""
    time_axis = np.linspace(0, target_timesteps * ctrl_dt, target_timesteps)
    colors = ["#1f77b4", "#ef7307"]
    n_muscles = len(EMG_MUSCLE_CONFIGS)
    fig, axes = plt.subplots(1, n_muscles, figsize=(6 * n_muscles, 5))
    if n_muscles == 1:
        axes = [axes]
    for ax, (sim_idx, sim_name, _, muscle_name) in zip(axes, EMG_MUSCLE_CONFIGS):
        emg_mean = emg_ref["means"].get(muscle_name)
        if emg_mean is None:
            ax.set_title(f"{muscle_name} - no EMG data")
            continue
        emg_sem = emg_ref["sems"][muscle_name]
        sim_muscle = sim_actions[:, :, sim_idx]
        n_trials = sim_muscle.shape[0]
        # Individual sim trials
        for i in range(min(n_trials, 46)):
            ax.plot(time_axis, sim_muscle[i], color=colors[0], alpha=0.1, linewidth=0.5)
        sim_mean = sim_muscle.mean(axis=0)
        sim_sem = sim_muscle.std(axis=0) / np.sqrt(n_trials)
        ax.plot(time_axis, sim_mean, color=colors[0], linewidth=2.5, label="Simulated")
        ax.fill_between(time_axis, sim_mean - sim_sem, sim_mean + sim_sem, color=colors[0], alpha=0.25)
        ax.plot(time_axis, emg_mean, color=colors[1], linewidth=2.5, label="Biological EMG")
        ax.fill_between(time_axis, emg_mean - emg_sem, emg_mean + emg_sem, color=colors[1], alpha=0.25)
        m = metrics_by_muscle[muscle_name]
        ax.set_title(f"{muscle_name} (r={m['mean_corr']:.3f}, MAE={m['mean_mae']:.4f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized activation")
        ax.set_ylim(0, 1.2)
        ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_emg_single_trials_fig(sim_actions, emg_ref, target_timesteps,
                               ctrl_dt=0.0025, n_trials=4):
    """Single-trial EMG vs sim overlay — one sim trial and one bio trial per panel.

    Creates n_trials rows x n_muscles cols.  Each panel shows one randomly
    selected biological EMG trace and the matching simulated action trace
    (same trial index) so you can see individual waveform shapes, not just
    the mean.
    """
    time_axis = np.linspace(0, target_timesteps * ctrl_dt, target_timesteps)
    colors_sim = "#1f77b4"
    colors_bio = "#ef7307"
    n_muscles = len(EMG_MUSCLE_CONFIGS)
    n_trials = min(n_trials, sim_actions.shape[0])

    fig, axes = plt.subplots(n_trials, n_muscles,
                             figsize=(6 * n_muscles, 3 * n_trials),
                             squeeze=False)
    for col, (sim_idx, sim_name, _, muscle_name) in enumerate(EMG_MUSCLE_CONFIGS):
        bio_traces = emg_ref["traces"].get(muscle_name)
        if bio_traces is None:
            for row in range(n_trials):
                axes[row, col].set_title(f"{muscle_name} - no EMG data")
            continue
        n_bio = bio_traces.shape[0]
        for row in range(n_trials):
            ax = axes[row, col]
            sim_trace = sim_actions[row, :, sim_idx]
            bio_idx = row % n_bio
            bio_trace = bio_traces[bio_idx, :target_timesteps]
            ax.plot(time_axis, sim_trace, color=colors_sim, linewidth=1.5,
                    label="Sim" if row == 0 else None)
            ax.plot(time_axis, bio_trace, color=colors_bio, linewidth=1.5,
                    label="Bio" if row == 0 else None)
            corr = float(np.corrcoef(sim_trace, bio_trace)[0, 1])
            ax.set_title(f"{muscle_name} trial {row} (r={corr:.3f})", fontsize=9)
            ax.set_ylim(0, 1.2)
            if row == 0:
                ax.legend(fontsize=7, loc="upper right")
            if row == n_trials - 1:
                ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("Activation")
    fig.suptitle("Single-trial EMG vs Simulated", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================================
# CLI arguments
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Janelia mouse forelimb imitation with intention network"
    )

    # Sweep metadata
    p.add_argument("--tag", type=str, default=None, help="Sweep run tag")
    p.add_argument("--run-name", type=str, default=None, help="Full run name")
    p.add_argument("--wandb-group", type=str, default=None, help="Wandb group")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb")

    # Physics overrides
    p.add_argument("--walker-xml", type=str, default=None,
                   help="Override walker XML path (e.g., .../mouse_forelimb_right_test2.xml)")
    p.add_argument("--joint-damping", type=float, default=None)
    p.add_argument("--joint-armature", type=float, default=None)
    p.add_argument("--joint-stiffness", type=float, default=None)
    p.add_argument("--force-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="Base RNG seed for policy/value init and rollouts. Vary for seed-variance studies.")
    p.add_argument("--biceps-force", type=float, default=None,
                   help="Absolute actuator_gainprm for Biceps_Long (pre-fs). Effective = this * force_scale.")
    p.add_argument("--brachialis-force", type=float, default=None,
                   help="Absolute actuator_gainprm for Brachialis (pre-fs).")
    p.add_argument("--triceps-long-force", type=float, default=None,
                   help="Absolute actuator_gainprm for Triceps_Long (pre-fs).")
    p.add_argument("--triceps-lat-force", type=float, default=None,
                   help="Absolute actuator_gainprm for Triceps_Lateral (pre-fs).")
    p.add_argument("--shoulder-damping", type=float, default=None,
                   help="Per-joint damping for sh_elv/sh_extension/sh_rotation. Overrides --joint-damping for these joints.")
    p.add_argument("--elbow-damping", type=float, default=None,
                   help="Per-joint damping for elbow. Overrides --joint-damping for elbow.")
    p.add_argument("--body-diaginertia", type=float, default=None,
                   help="Override diagonal inertia (scalar applied to all bodies) — leave unset to use the loaded XML's native per-body inertials")
    p.add_argument("--muscle-tau-act", type=float, default=None,
                   help="Override muscle activation time constant (sec) for all muscles")
    p.add_argument("--muscle-tau-deact", type=float, default=None,
                   help="Override muscle deactivation time constant (sec) for all muscles")
    # Per-muscle tau overrides (override global --muscle-tau-* for the named actuator only)
    p.add_argument("--biceps-tau-act", type=float, default=None)
    p.add_argument("--biceps-tau-deact", type=float, default=None)
    p.add_argument("--brachialis-tau-act", type=float, default=None)
    p.add_argument("--brachialis-tau-deact", type=float, default=None)
    p.add_argument("--triceps-long-tau-act", type=float, default=None)
    p.add_argument("--triceps-long-tau-deact", type=float, default=None)
    p.add_argument("--triceps-lat-tau-act", type=float, default=None)
    p.add_argument("--triceps-lat-tau-deact", type=float, default=None)
    p.add_argument("--emg-norm-percentile", type=float, default=100.0,
                   help="Percentile used to normalize reference EMG envelopes (arr / np.percentile(arr, P)). "
                        "Default 100 (true max) ensures no reference sample exceeds 1.0 pre-clip. "
                        "Pre-s15 default was 98 — use 98.0 to reproduce old metrics.")

    # Timestep overrides
    p.add_argument("--ctrl-dt", type=float, default=None,
                   help="Control timestep in seconds (default 0.0025 = 400Hz). "
                        "Larger values = slower decisions = naturally smoother actions.")
    p.add_argument("--sim-dt", type=float, default=None,
                   help="Physics simulation timestep in seconds (default 0.00125 = 800Hz). "
                        "Must be <= ctrl_dt and ctrl_dt/sim_dt must be integer.")

    # Training / env overrides
    p.add_argument("--qvel-init", type=str, default="zeros",
                   choices=["zeros", "reference"],
                   help="Initial qvel: 'zeros' (top-5 historical default, "
                        "shining-star config) or 'reference' (round-3 default, underperforms)")
    p.add_argument("--reference-length", type=int, default=None)
    p.add_argument("--episode-length", type=int, default=None)
    p.add_argument("--entropy-cost", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--discounting", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-minibatches", type=int, default=None)
    p.add_argument("--num-timesteps", type=int, default=None)
    p.add_argument("--num-evals", type=int, default=None)

    # Reward overrides
    p.add_argument("--control-cost", type=float, default=None)
    p.add_argument("--control-diff-cost", type=float, default=None)
    p.add_argument("--saturation-cost", type=float, default=None,
                   help="Weight for the saturation penalty (0 = off, recommended 0.01-0.05). "
                        "Penalizes |action| > saturation_margin to discourage bang-bang strategies.")
    p.add_argument("--saturation-margin", type=float, default=None,
                   help="Saturation dead-zone radius in policy output space (default 0.8). "
                        "Inside [-margin, +margin] no penalty; outside, quadratic ramp.")
    p.add_argument("--joints-weight", type=float, default=None)
    p.add_argument("--joints-vel-weight", type=float, default=None)
    p.add_argument("--wrist-pos-weight", type=float, default=None)
    p.add_argument("--bodies-pos-weight", type=float, default=None)

    # Intention network overrides
    p.add_argument("--latent-size", type=int, default=None)
    p.add_argument("--kl-weight", type=float, default=None)
    p.add_argument("--ar1-weight", type=float, default=None)

    # Sigmoid-normal distribution overrides
    p.add_argument("--init-scale", type=float, default=None,
                   help="Bias on pre-sigmoid loc (0.0 → sigmoid=0.5 center)")
    p.add_argument("--init-log-std", type=float, default=None,
                   help="Bias on pre-softplus scale (-3.0 → std≈0.049, tight)")

    return p.parse_args()


args = parse_args()


# =============================================================================
# Sigmoid-Normal Distribution (replaces NormalTanhDistribution)
# =============================================================================


class _NormalDist:
    """Minimal Normal distribution (matches brax's internal _NormalDistribution)."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jp.log(2.0 * jp.pi) + jp.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        return 0.5 * jp.log(2.0 * jp.pi * jp.e) + jp.log(self.scale)


class SigmoidBijector:
    """Sigmoid bijector: maps reals → (0, 1)."""

    def forward(self, x):
        return jax.nn.sigmoid(x)

    def inverse(self, y):
        return jp.log(y / (1.0 - y))  # logit

    def forward_log_det_jacobian(self, x):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        # log|det| = log(sigmoid(x)) + log(1 - sigmoid(x))
        #          = -softplus(-x) + (-x - softplus(-x))
        #          = x - 2 * softplus(x)      [numerically stable form]
        return -x - 2.0 * jax.nn.softplus(-x)


class SigmoidNormalDistribution(ParametricDistribution):
    """Normal distribution followed by sigmoid → outputs in (0, 1).

    Suitable for muscle activations bounded to [0, 1].
    sigmoid(loc=0) = 0.5, giving a natural centered starting point.
    """

    def __init__(self, event_size, min_std=0.001, var_scale=1.0):
        super().__init__(
            param_size=2 * event_size,
            postprocessor=SigmoidBijector(),
            event_ndims=1,
            reparametrizable=True,
        )
        self._min_std = min_std
        self._var_scale = var_scale
        self._event_size = event_size

    def create_dist(self, parameters):
        loc, scale = jp.split(parameters, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return _NormalDist(loc=loc, scale=scale)



# =============================================================================
# Variational Intention Network (Encoder → Gaussian → Decoder)
# =============================================================================


class Encoder(linen.Module):
    """Maps task observations → (mean, logvar) of latent Gaussian."""
    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = linen.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = linen.Dense(
                hidden_size, name=f"hidden_{i}",
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation(x)
            x = linen.LayerNorm()(x)
        mean = linen.Dense(self.latents, name="fc_mean")(x)
        logvar = linen.Dense(self.latents, name="fc_logvar")(x)
        return mean, logvar


class Decoder(linen.Module):
    """Maps concat(z, proprioception) → action distribution params."""
    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = linen.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = linen.Dense(
                hidden_size, name=f"hidden_{i}",
                kernel_init=self.kernel_init,
            )(x)
            if i != len(self.layer_sizes) - 1:
                x = self.activation(x)
                x = linen.LayerNorm()(x)
        return x


class IntentionPolicy(linen.Module):
    """Encoder-decoder VAE policy with biased output for sigmoid-normal dist.

    The decoder outputs 2*act_size params: [loc, scale_raw].
    init_scale biases loc so that sigmoid(loc + init_scale) sets the initial
    mean activation. init_log_std biases scale_raw so that
    softplus(scale_raw + init_log_std) sets initial exploration width.

    init_scale=0.0  → sigmoid(0) = 0.5 (centered muscle activation)
    init_log_std=-3.0 → softplus(-3) ≈ 0.049 (tight initial exploration)
    """
    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int
    proprio_size: int
    act_size: int = 12
    init_scale: float = -4.0
    init_log_std: float = -5.0

    def setup(self):
        self.encoder = Encoder(
            layer_sizes=self.encoder_layers, latents=self.latents
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(self, obs_flat, key, deterministic=False):
        proprio = obs_flat[..., :self.proprio_size]
        task_obs = obs_flat[..., self.proprio_size:]

        mean, logvar = self.encoder(task_obs)

        std = jp.exp(0.5 * logvar)
        eps = jax.random.normal(key, logvar.shape)
        z_sampled = mean + eps * std
        z = jp.where(deterministic, mean, z_sampled)

        decoder_input = jp.concatenate([z, proprio], axis=-1)
        logits = self.decoder(decoder_input)

        # Bias the loc and scale_raw halves of the output
        loc = logits[..., :self.act_size] + self.init_scale
        scale_raw = logits[..., self.act_size:] + self.init_log_std
        logits = jp.concatenate([loc, scale_raw], axis=-1)

        return logits, mean, logvar


# =============================================================================
# VAE Loss Components
# =============================================================================


def compute_kl_to_gaussian_prior(latent_mean, latent_logvar):
    """KL(q(z|x) || N(0,I)). Inputs shape [T, B, D] or [N, D]."""
    return -0.5 * jp.mean(
        1 + latent_logvar - jp.square(latent_mean) - jp.exp(latent_logvar)
    )


def compute_ar1_temporal_loss(latent_mean, discount, truncation):
    """L2 smoothness between consecutive latent means.

    Masks out episode boundaries (done or truncated).
    latent_mean: (T, B, D), discount/truncation: (T, B).
    """
    z_prev = latent_mean[:-1]
    z_curr = latent_mean[1:]
    valid_mask = discount[:-1] * (1.0 - truncation[:-1])
    l2_diff = jp.mean(jp.square(z_curr - z_prev), axis=-1)
    masked_l2 = l2_diff * valid_mask
    return jp.sum(masked_l2) / jp.maximum(jp.sum(valid_mask), 1.0)


def create_ramp_schedule(
    max_value=0.1, min_value=0.0001, ramp_steps=1000,
    warmup_steps=0, schedule="linear", period=45,
):
    def schedule_fn(step):
        step = jp.asarray(step, dtype=jp.float32)
        if schedule == "linear":
            progress = jp.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0)
            is_warmup = step < warmup_steps
            return jp.where(
                is_warmup, min_value,
                min_value + progress * (max_value - min_value),
            )
        elif schedule == "cosine":
            angle = (2 * jp.pi * step) / period
            amp = (max_value - min_value) / 2
            mid = (max_value + min_value) / 2
            return mid + amp * jp.cos(angle)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    return schedule_fn


# =============================================================================
# PPO Utilities
# =============================================================================


class Transition(NamedTuple):
    obs: Any          # (B, obs_dim) flat
    action: Any       # (B, act_dim)
    raw_action: Any   # (B, act_dim)
    log_prob: Any     # (B,)
    value: Any        # (B,)
    reward: Any       # (B,)
    done: Any         # (B,)
    truncation: Any   # (B,)


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Vectorised GAE via reverse scan. All inputs (T, B)."""
    T = rewards.shape[0]

    def body(carry, t_rev):
        gae, next_val = carry
        t = T - 1 - t_rev
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        return (gae, values[t]), gae

    _, advantages_rev = jax.lax.scan(
        body, (jp.zeros_like(last_value), last_value), jp.arange(T)
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def flatten_obs(obs):
    """Flatten nested observation dict to a single array (sorted keys)."""
    flat_parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, dict):
            flat_parts.append(flatten_obs(val))
        else:
            flat_parts.append(val.flatten())
    return jp.concatenate(flat_parts)


# =============================================================================
# Environment Config (Janelia model)
# =============================================================================

env_cfg = default_config()
if args.walker_xml is not None:
    env_cfg.walker_xml_path = epath.Path(args.walker_xml)
else:
    env_cfg.walker_xml_path = JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH

env_cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH)
# Janelia model bodies
env_cfg.tracked_bodies = ["scapula", "humerus", "ulna", "wrist"]
env_cfg.end_effector = "wrist"
env_cfg.recompute_kinematics = False  # IK from same kinematic chain as sim model
env_cfg.ik_driven_dims = 3

# S9-redo (moving-shoulder) best-frontier winner: d8em7-fs1p0
# (reward=403, triceps_corr=0.77, biceps_corr=0.48, trip_trial_mae=0.16,
#  bic_trial_mae=0.23 at clip_length=50, episode_length=100, ref_length=2).
# Pattern across the redo: fs=1.0 dominates on reward at every damping;
# fs<=0.4 caps reward near 100 with no EMG benefit worth the trade.
env_cfg.joint_damping = 8e-7
env_cfg.joint_armature = 4e-10
env_cfg.force_scale = 1.0

# Reward defaults matching S9 winners (cc=0.05, cdc=0.1 — the cdc=0.1 weight
# is load-bearing for EMG quality; every historical run with tc+bc>1.2 used it).
env_cfg.reward_terms["control_cost"]["weight"] = 0.05
env_cfg.reward_terms["control_diff_cost"]["weight"] = 0.1

# Apply physics overrides from CLI (override defaults above)
if args.joint_damping is not None:
    env_cfg.joint_damping = args.joint_damping
if args.joint_armature is not None:
    env_cfg.joint_armature = args.joint_armature
if args.joint_stiffness is not None:
    env_cfg.joint_stiffness = args.joint_stiffness
if args.force_scale is not None:
    env_cfg.force_scale = args.force_scale
if args.biceps_force is not None:
    env_cfg.biceps_force = args.biceps_force
if args.brachialis_force is not None:
    env_cfg.brachialis_force = args.brachialis_force
if args.triceps_long_force is not None:
    env_cfg.triceps_long_force = args.triceps_long_force
if args.triceps_lat_force is not None:
    env_cfg.triceps_lat_force = args.triceps_lat_force
if args.shoulder_damping is not None:
    env_cfg.shoulder_damping = args.shoulder_damping
if args.elbow_damping is not None:
    env_cfg.elbow_damping = args.elbow_damping
if args.body_diaginertia is not None:
    env_cfg.body_diaginertia = args.body_diaginertia
if args.muscle_tau_act is not None:
    env_cfg.muscle_tau_act = args.muscle_tau_act
if args.muscle_tau_deact is not None:
    env_cfg.muscle_tau_deact = args.muscle_tau_deact
if args.biceps_tau_act is not None:
    env_cfg.biceps_tau_act = args.biceps_tau_act
if args.biceps_tau_deact is not None:
    env_cfg.biceps_tau_deact = args.biceps_tau_deact
if args.brachialis_tau_act is not None:
    env_cfg.brachialis_tau_act = args.brachialis_tau_act
if args.brachialis_tau_deact is not None:
    env_cfg.brachialis_tau_deact = args.brachialis_tau_deact
if args.triceps_long_tau_act is not None:
    env_cfg.triceps_long_tau_act = args.triceps_long_tau_act
if args.triceps_long_tau_deact is not None:
    env_cfg.triceps_long_tau_deact = args.triceps_long_tau_deact
if args.triceps_lat_tau_act is not None:
    env_cfg.triceps_lat_tau_act = args.triceps_lat_tau_act
if args.triceps_lat_tau_deact is not None:
    env_cfg.triceps_lat_tau_deact = args.triceps_lat_tau_deact
if args.ctrl_dt is not None:
    env_cfg.ctrl_dt = args.ctrl_dt
if args.sim_dt is not None:
    env_cfg.sim_dt = args.sim_dt
# Validate timestep relationship
assert env_cfg.ctrl_dt >= env_cfg.sim_dt, (
    f"ctrl_dt ({env_cfg.ctrl_dt}) must be >= sim_dt ({env_cfg.sim_dt})")
n_substeps = env_cfg.ctrl_dt / env_cfg.sim_dt
assert abs(n_substeps - round(n_substeps)) < 1e-9, (
    f"ctrl_dt/sim_dt must be integer, got {n_substeps:.4f}")

# Always apply qvel_init from CLI so it's explicit in wandb config (default "zeros")
env_cfg.qvel_init = args.qvel_init

# Apply env/reward overrides from CLI
if args.reference_length is not None:
    env_cfg.reference_length = args.reference_length
if args.control_cost is not None:
    env_cfg.reward_terms["control_cost"]["weight"] = args.control_cost
if args.control_diff_cost is not None:
    env_cfg.reward_terms["control_diff_cost"]["weight"] = args.control_diff_cost
if args.saturation_cost is not None:
    env_cfg.reward_terms["saturation_cost"]["weight"] = args.saturation_cost
if args.saturation_margin is not None:
    env_cfg.reward_terms["saturation_cost"]["margin"] = args.saturation_margin
if args.joints_weight is not None:
    env_cfg.reward_terms["joints"]["weight"] = args.joints_weight
if args.joints_vel_weight is not None:
    env_cfg.reward_terms["joints_vel"]["weight"] = args.joints_vel_weight
if args.wrist_pos_weight is not None:
    env_cfg.reward_terms["wrist_pos"]["weight"] = args.wrist_pos_weight
if args.bodies_pos_weight is not None:
    env_cfg.reward_terms["bodies_pos"]["weight"] = args.bodies_pos_weight


# =============================================================================
# PPO + Intention Network Config
# =============================================================================

ppo_params = config_dict.create(
    num_timesteps=1_000_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=100,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=8192,
    max_grad_norm=1.0,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    latent_kl_weight=1e-3,
    latent_ar1_weight=1e-3,
    network_factory=config_dict.create(
        encoder_hidden_layer_sizes=(512, 512, 512),
        decoder_hidden_layer_sizes=(512, 512, 512),
        value_hidden_layer_sizes=(512, 512, 512),
        latent_size=4,
    ),
)

# Apply training overrides from CLI
if args.entropy_cost is not None:
    ppo_params.entropy_cost = args.entropy_cost
if args.learning_rate is not None:
    ppo_params.learning_rate = args.learning_rate
if args.discounting is not None:
    ppo_params.discounting = args.discounting
if args.batch_size is not None:
    ppo_params.batch_size = args.batch_size
if args.num_minibatches is not None:
    ppo_params.num_minibatches = args.num_minibatches
if args.episode_length is not None:
    ppo_params.episode_length = args.episode_length
if args.num_timesteps is not None:
    ppo_params.num_timesteps = args.num_timesteps
if args.num_evals is not None:
    ppo_params.num_evals = args.num_evals
if args.latent_size is not None:
    ppo_params.network_factory.latent_size = args.latent_size
if args.kl_weight is not None:
    ppo_params.latent_kl_weight = args.kl_weight
if args.ar1_weight is not None:
    ppo_params.latent_ar1_weight = args.ar1_weight

pprint(ppo_params)


# =============================================================================
# Experiment naming / checkpoint setup
# =============================================================================

env_name = "janelia-mouse-sigmoid-moving-shoulder"
FINETUNE_PATH = None

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")

if args.run_name is not None:
    exp_name = args.run_name
else:
    exp_name = f"{env_name}-{timestamp}"
    if args.tag is not None:
        exp_name += f"-{args.tag}"

# Build param summary for wandb name
_param_parts = []
_param_map = [
    ("damp", args.joint_damping), ("arm", args.joint_armature),
    ("stiff", args.joint_stiffness), ("fscale", args.force_scale),
    ("ref", args.reference_length), ("ep", args.episode_length),
    ("ent", args.entropy_cost), ("lr", args.learning_rate),
    ("disc", args.discounting), ("bs", args.batch_size),
    ("mb", args.num_minibatches), ("ctrl", args.control_cost),
    ("cdiff", args.control_diff_cost), ("lat", args.latent_size),
    ("kl", args.kl_weight), ("ar1", args.ar1_weight),
    ("iscale", args.init_scale), ("istd", args.init_log_std),
]
for short, val in _param_map:
    if val is not None:
        _param_parts.append(f"{short}={val:g}" if isinstance(val, float) else f"{short}={val}")
param_suffix = "_".join(_param_parts)
wandb_name = f"{exp_name}_{param_suffix}" if param_suffix else exp_name

print(f"Experiment name: {exp_name}")
print(f"Wandb name: {wandb_name}")

if FINETUNE_PATH is not None:
    FINETUNE_PATH = epath.Path(FINETUNE_PATH)
    latest_ckpts = [c for c in FINETUNE_PATH.glob("*") if c.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    restore_checkpoint_path = latest_ckpts[-1]
    print(f"Restoring from: {restore_checkpoint_path}")
else:
    restore_checkpoint_path = None

ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint dir: {ckpt_path}")

env_cfg_dict = env_cfg.to_dict()
for k, v in env_cfg_dict.items():
    if hasattr(v, "__fspath__"):
        env_cfg_dict[k] = str(v)
with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg_dict, fp, indent=4, default=str)


# =============================================================================
# Wandb
# =============================================================================

USE_WANDB = not args.no_wandb

if USE_WANDB:
    wandb_kwargs = dict(
        project="vnl-mjx-rl",
        config=env_cfg,
        name=wandb_name,
        id=f"janelia-sigmoid-{exp_name}-{timestamp}",
    )
    if args.wandb_group is not None:
        wandb_kwargs["group"] = args.wandb_group
    if args.wandb_tags is not None:
        wandb_kwargs["tags"] = args.wandb_tags

    wandb.init(**wandb_kwargs)
    wandb.config.update({
        "env_name": env_name,
        "policy_type": "variational_intention_sigmoid_normal",
        **dict(ppo_params.network_factory),
    })
    sweep_config = {
        "sweep/tag": args.tag,
        "sweep/joint_damping": args.joint_damping,
        "sweep/joint_armature": args.joint_armature,
        "sweep/joint_stiffness": args.joint_stiffness,
        "sweep/force_scale": args.force_scale,
        "sweep/reference_length": args.reference_length,
        "sweep/episode_length": args.episode_length,
        "sweep/entropy_cost": args.entropy_cost,
        "sweep/learning_rate": args.learning_rate,
        "sweep/discounting": args.discounting,
        "sweep/latent_size": args.latent_size,
        "sweep/kl_weight": args.kl_weight,
        "sweep/ar1_weight": args.ar1_weight,
        "sweep/init_scale": args.init_scale,
        "sweep/init_log_std": args.init_log_std,
    }
    wandb.config.update({k: v for k, v in sweep_config.items() if v is not None})

    # Log all reward term weights
    reward_weights = {
        f"reward_weights/{name}": term["weight"]
        for name, term in env_cfg.reward_terms.items()
    }
    wandb.config.update(reward_weights)

    # Log XML-derived diagnostics so silent edits to the walker XML
    # (joint ranges, inertials, muscle forces, time constants) show up in
    # the run config for after-the-fact drift detection (closes drifts
    # D2/D3/D4 from docs/2026-04-11-shining-star-replication-sweep.md).
    import hashlib
    _xml_path = str(env_cfg.walker_xml_path)
    _xml_model = mujoco.MjModel.from_xml_path(_xml_path)
    _xml_diag = {"xml/path": _xml_path}
    with open(_xml_path, "rb") as _f:
        _xml_diag["xml/sha256"] = hashlib.sha256(_f.read()).hexdigest()
    # Joint ranges (radians)
    for _j in range(_xml_model.njnt):
        _name = _xml_model.joint(_j).name
        _xml_diag[f"xml/joint_lo/{_name}"] = float(_xml_model.jnt_range[_j, 0])
        _xml_diag[f"xml/joint_hi/{_name}"] = float(_xml_model.jnt_range[_j, 1])
    # Body diagonal inertias (skip body 0 = world; bodies are spheroidal so ix=iy=iz)
    for _b in range(1, _xml_model.nbody):
        _name = _xml_model.body(_b).name
        _xml_diag[f"xml/diaginertia/{_name}"] = float(_xml_model.body_inertia[_b, 0])
    # Muscle actuator params: force (gainprm[2]), vmax (gainprm[6]),
    # tau_act/tau_deact (dynprm[0:2]).  Skip non-muscle actuators.
    _MUSCLE_GAIN = int(mujoco.mjtGain.mjGAIN_MUSCLE.value)
    for _a in range(_xml_model.nu):
        _name = _xml_model.actuator(_a).name
        if int(_xml_model.actuator_gaintype[_a]) == _MUSCLE_GAIN:
            _xml_diag[f"xml/muscle_force/{_name}"] = float(_xml_model.actuator_gainprm[_a, 2])
            _xml_diag[f"xml/muscle_vmax/{_name}"]  = float(_xml_model.actuator_gainprm[_a, 6])
            _xml_diag[f"xml/muscle_tau_act/{_name}"]   = float(_xml_model.actuator_dynprm[_a, 0])
            _xml_diag[f"xml/muscle_tau_deact/{_name}"] = float(_xml_model.actuator_dynprm[_a, 1])
    wandb.config.update(_xml_diag)
    print(f"Logged {len(_xml_diag)} XML diagnostics to wandb config "
          f"(sha256={_xml_diag['xml/sha256'][:12]}...)")


# =============================================================================
# Rendering (Janelia-specific camera + ghost)
# =============================================================================


def build_render_model(env):
    """Build MuJoCo model for rendering with camera aimed at Janelia arm."""
    render_spec = mujoco.MjSpec.from_file(str(env.arena_xml_path))
    walker_spec = mujoco.MjSpec.from_file(str(env.walker_xml_path))
    spawn_frame = render_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
    spawn_frame.attach_body(walker_spec.body("clavicle"), "", "-mouse")

    # Ghost walker for reference motion
    ghost_spec = mujoco.MjSpec.from_file(str(env.walker_xml_path))
    ghost_frame = render_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
    ghost_body = ghost_frame.attach_body(ghost_spec.body("clavicle"), "", "-ghost")

    def recolor_geoms(body, rgba):
        for g in body.geoms:
            g.rgba = rgba
            g.contype = 0
            g.conaffinity = 0
        for child in body.bodies:
            recolor_geoms(child, rgba)

    recolor_geoms(ghost_body, [0.3, 0.8, 1.0, 0.4])

    render_spec.worldbody.add_camera(
        name="janelia_cam",
        pos=[0.020, -0.035, 0.065],
        xyaxes=[1, 0, 0, 0, 0, 1],
        fovy=55,
    )

    rm = render_spec.compile()
    rm.opt.timestep = env._config.sim_dt
    return rm


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Janelia Mouse Arm Imitation -- Sigmoid-Normal Distribution")
    print("=" * 80)
    nf = ppo_params.network_factory
    print(f"Encoder layers: {nf.encoder_hidden_layer_sizes}")
    print(f"Decoder layers: {nf.decoder_hidden_layer_sizes}")
    print(f"Latent size:    {nf.latent_size}")
    print(f"KL weight:      {ppo_params.latent_kl_weight}")
    print(f"AR(1) weight:   {ppo_params.latent_ar1_weight}")

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    print(f"Loading reference clips from {env_cfg.reference_data_path}...")
    reference_clips = MouseReferenceClips(
        str(env_cfg.reference_data_path),
        n_frames_per_clip=env_cfg.clip_length,
    )
    train_clips, test_clips = reference_clips.split(train_ratio=0.8, seed=42)

    env = MouseImitationMovingShoulder(config=env_cfg, clips=train_clips)
    eval_env = MouseImitationMovingShoulder(config=env_cfg, clips=test_clips)
    # EMG comparison env uses ALL clips (not train/test split)
    emg_env = MouseImitationMovingShoulder(config=env_cfg, clips=reference_clips)
    n_emg_clips = reference_clips.qpos.shape[0]
    print(f"Action size: {env.action_size}  Obs size: {env.observation_size}")

    steps_per_frame = (1 / env_cfg.mocap_hz) / env_cfg.ctrl_dt
    episode_length = int(
        (env_cfg.clip_length - env_cfg.start_frame_range[-1]
         - env_cfg.reference_length) * steps_per_frame
    )
    if args.episode_length is not None:
        episode_length = args.episode_length
    print(f"Episode length: {episode_length}")

    # Determine obs split: flatten_obs sorts keys alphabetically
    # obs structure: {"state": {"proprioception": ..., "task_obs": {...}}}
    # flatten_obs recurses into "state", then sorts: proprioception < task_obs
    # => flat_obs = [proprioception | task_obs]
    _dummy_rng = jax.random.PRNGKey(99)
    _dummy_state = env.reset(_dummy_rng)
    _dummy_obs = _dummy_state.obs["state"]  # unwrap outer "state" key
    _proprio_flat = _dummy_obs["proprioception"].flatten()
    _task_obs = _dummy_obs["task_obs"]
    if isinstance(_task_obs, dict):
        _task_parts = []
        for k in sorted(_task_obs.keys()):
            _task_parts.append(_task_obs[k].flatten())
        _task_flat = jp.concatenate(_task_parts)
    else:
        _task_flat = _task_obs.flatten()
    proprio_size = _proprio_flat.shape[0]
    task_obs_size = _task_flat.shape[0]
    obs_size = proprio_size + task_obs_size
    act_size = env.action_size
    print(f"Proprio size: {proprio_size}  Task obs size: {task_obs_size}  "
          f"Total obs: {obs_size}")

    # ------------------------------------------------------------------
    # Wrap envs for vectorised rollouts
    # ------------------------------------------------------------------
    def flatten_obs_wrapper(env_fn):
        class W:
            def __init__(self, e):
                self._e = e
            def reset(self, rng):
                s = self._e.reset(rng)
                return s.replace(obs=flatten_obs(s.obs))
            def step(self, state, action):
                # SigmoidNormalDistribution already outputs [0, 1] — no rescaling
                s = self._e.step(state, action)
                return s.replace(obs=flatten_obs(s.obs))
            @property
            def observation_size(self):
                return self._e.observation_size
            @property
            def action_size(self):
                return self._e.action_size
            @property
            def dt(self):
                return self._e.dt
            def __getattr__(self, name):
                return getattr(self._e, name)
        return W(env_fn)

    wrapped_env = flatten_obs_wrapper(env)
    wrapped_eval_env = flatten_obs_wrapper(eval_env)

    wrap_fn = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)
    train_env = wrap_fn(
        wrapped_env,
        episode_length=episode_length,
        action_repeat=ppo_params.action_repeat,
    )
    test_env = wrap_fn(
        wrapped_eval_env,
        episode_length=episode_length,
        action_repeat=ppo_params.action_repeat,
    )

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    action_dist = SigmoidNormalDistribution(event_size=act_size)
    param_size = action_dist.param_size

    policy_module = IntentionPolicy(
        encoder_layers=nf.encoder_hidden_layer_sizes,
        decoder_layers=list(nf.decoder_hidden_layer_sizes) + [param_size],
        latents=nf.latent_size,
        proprio_size=proprio_size,
        act_size=act_size,
        init_scale=args.init_scale if args.init_scale is not None else 0.0,
        init_log_std=args.init_log_std if args.init_log_std is not None else -3.0,
    )
    value_module = networks.MLP(
        layer_sizes=list(nf.value_hidden_layer_sizes) + [1],
        activation=linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    num_envs = ppo_params.num_envs
    dummy_obs = jp.zeros((1, obs_size))
    dummy_key = jax.random.PRNGKey(0)

    key = jax.random.PRNGKey(args.seed)
    key, pk, vk, ek = jax.random.split(key, 4)
    policy_params = policy_module.init(pk, dummy_obs, dummy_key)
    value_params = value_module.init(vk, dummy_obs)
    normalizer_params = running_statistics.init_state(jp.zeros(obs_size))

    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo_params.max_grad_norm),
        optax.adam(ppo_params.learning_rate),
    )
    opt_state = optimizer.init((policy_params, value_params))

    # ------------------------------------------------------------------
    # Hyperparameters for JIT closures
    # ------------------------------------------------------------------
    unroll_length = ppo_params.unroll_length
    num_updates = ppo_params.num_updates_per_batch
    num_minibatches = ppo_params.num_minibatches
    gamma = ppo_params.discounting
    gae_lambda = ppo_params.gae_lambda
    reward_scaling = ppo_params.reward_scaling
    clip_eps = ppo_params.clip_eps
    vf_coef = ppo_params.vf_coef
    entropy_cost = ppo_params.entropy_cost
    kl_weight = ppo_params.latent_kl_weight
    ar1_weight = ppo_params.latent_ar1_weight
    mb_env_size = num_envs // num_minibatches

    # ------------------------------------------------------------------
    # JIT-compiled core functions
    # ------------------------------------------------------------------

    @jax.jit
    def collect_rollout(policy_params, value_params, normalizer_params,
                        env_state, rng):
        """Collect unroll_length transitions."""

        def step_fn(carry, _):
            state, k = carry
            k, ak, pk = jax.random.split(k, 3)

            obs_norm = running_statistics.normalize(state.obs, normalizer_params)
            logits, _, _ = policy_module.apply(
                policy_params, obs_norm, pk
            )
            raw_action = action_dist.sample_no_postprocessing(logits, ak)
            log_prob = action_dist.log_prob(logits, raw_action)
            action = action_dist.postprocess(raw_action)
            value = jp.squeeze(
                value_module.apply(value_params, obs_norm), axis=-1
            )

            next_state = train_env.step(state, action)
            truncation = next_state.info.get(
                "truncation", jp.zeros_like(next_state.done)
            )

            transition = Transition(
                obs=state.obs,
                action=action,
                raw_action=raw_action,
                log_prob=log_prob,
                value=value,
                reward=next_state.reward,
                done=next_state.done,
                truncation=truncation,
            )
            return (next_state, k), transition

        (final_state, _), rollout = jax.lax.scan(
            step_fn,
            (env_state, rng),
            None,
            length=unroll_length,
        )
        return final_state, rollout

    def _sgd_step(policy_params, value_params, opt_state, normalizer_params,
                  mb_obs, mb_raw, mb_lp, mb_adv, mb_ret, mb_done, mb_trunc,
                  rng):
        """Single gradient update on a temporal minibatch.

        All mb_* arrays have shape (T, B', ...) where B' = mb_env_size.
        """

        def loss_fn(params):
            pp, vp = params
            T, Bp = mb_obs.shape[:2]

            obs_norm = running_statistics.normalize(mb_obs, normalizer_params)

            flat_obs = obs_norm.reshape(T * Bp, -1)
            enc_key = jax.random.fold_in(rng, 0)
            flat_logits, flat_mean, flat_logvar = policy_module.apply(
                pp, flat_obs, enc_key
            )

            logits = flat_logits.reshape(T, Bp, -1)
            latent_mean = flat_mean.reshape(T, Bp, -1)
            latent_logvar = flat_logvar.reshape(T, Bp, -1)

            new_log_prob = action_dist.log_prob(
                logits.reshape(T * Bp, -1),
                mb_raw.reshape(T * Bp, -1),
            ).reshape(T, Bp)

            ratio = jp.exp(new_log_prob - mb_lp)
            adv = (mb_adv - jp.mean(mb_adv)) / (jp.std(mb_adv) + 1e-8)

            pg1 = -adv * ratio
            pg2 = -adv * jp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = jp.mean(jp.maximum(pg1, pg2))

            new_value = jp.squeeze(
                value_module.apply(vp, flat_obs), axis=-1
            ).reshape(T, Bp)
            value_loss = jp.mean(jp.square(new_value - mb_ret))

            ent_key = jax.random.fold_in(rng, 1)
            entropy = jp.mean(
                action_dist.entropy(logits.reshape(T * Bp, -1), ent_key)
            )

            kl_loss = compute_kl_to_gaussian_prior(latent_mean, latent_logvar)

            discount = 1.0 - mb_done
            ar1_loss = compute_ar1_temporal_loss(
                latent_mean, discount, mb_trunc
            )

            total = (policy_loss
                     + vf_coef * value_loss
                     - entropy_cost * entropy
                     + kl_weight * kl_loss
                     + ar1_weight * ar1_loss)

            kl_per_dim = -0.5 * (
                1 + latent_logvar - jp.square(latent_mean)
                - jp.exp(latent_logvar)
            )
            mean_kl_per_dim = jp.mean(kl_per_dim, axis=(0, 1))
            active_dims = jp.sum(mean_kl_per_dim > 0.01)

            latent_stds = jp.exp(0.5 * latent_logvar)
            mean_std_per_dim = jp.mean(latent_stds, axis=(0, 1))

            return total, {
                "total_loss": total,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "kl_loss": kl_loss,
                "ar1_loss": ar1_loss,
                "approx_kl": jp.mean((ratio - 1.0) - jp.log(ratio)),
                "latent_kl_weight": kl_weight,
                "latent_ar1_weight": ar1_weight,
                "latent_mean_norm": jp.mean(jp.sqrt(
                    jp.sum(jp.square(latent_mean), axis=-1)
                )),
                "latent_std_mean": jp.mean(latent_stds),
                "latent_std_min": jp.min(mean_std_per_dim),
                "latent_std_max": jp.max(mean_std_per_dim),
                "latent_mean_abs": jp.mean(jp.abs(latent_mean)),
                "active_latent_dims": active_dims,
                "collapsed_dims": jp.sum(mean_kl_per_dim < 0.001),
                "latent_rate_nats": jp.mean(
                    jp.sum(kl_per_dim, axis=-1)),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            (policy_params, value_params)
        )
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, (policy_params, value_params)
        )
        new_pp, new_vp = optax.apply_updates(
            (policy_params, value_params), updates
        )
        metrics["grad_norm"] = grad_norm
        return new_pp, new_vp, new_opt_state, loss, metrics

    @jax.jit
    def prepare_ppo_data(normalizer_params, rollout, env_state_obs, value_params):
        normalizer_params = running_statistics.update(
            normalizer_params, rollout.obs.reshape(-1, obs_size),
        )
        last_obs_norm = running_statistics.normalize(
            env_state_obs, normalizer_params
        )
        last_value = jp.squeeze(
            value_module.apply(value_params, last_obs_norm), axis=-1
        )
        rewards = rollout.reward * reward_scaling
        advantages, returns = compute_gae(
            rewards, rollout.value, rollout.done, last_value, gamma, gae_lambda,
        )
        return normalizer_params, advantages, returns

    @jax.jit
    def run_ppo_epochs(policy_params, value_params, opt_state,
                       normalizer_params,
                       r_obs, r_raw, r_lp, r_adv, r_ret, r_done, r_trunc,
                       key):
        """PPO update with temporal minibatching along the env (B) dimension."""
        B = r_obs.shape[1]

        def epoch_step(carry, _):
            pp, vp, os, k = carry
            k, perm_key = jax.random.split(k)
            perm = jax.random.permutation(perm_key, B)

            def mb_step(carry2, mb_idx):
                pp2, vp2, os2, k2 = carry2
                k2, ek = jax.random.split(k2)
                start = mb_idx * mb_env_size
                idx = jax.lax.dynamic_slice(perm, (start,), (mb_env_size,))

                pp2, vp2, os2, loss, metrics = _sgd_step(
                    pp2, vp2, os2, normalizer_params,
                    r_obs[:, idx], r_raw[:, idx], r_lp[:, idx],
                    r_adv[:, idx], r_ret[:, idx],
                    r_done[:, idx], r_trunc[:, idx], ek,
                )
                return (pp2, vp2, os2, k2), (loss, metrics)

            (pp, vp, os, k), (losses, all_metrics) = jax.lax.scan(
                mb_step, (pp, vp, os, k), jp.arange(num_minibatches)
            )
            last_metrics = jax.tree.map(lambda x: x[-1], all_metrics)
            return (pp, vp, os, k), last_metrics

        (pp, vp, os, k), epoch_metrics = jax.lax.scan(
            epoch_step,
            (policy_params, value_params, opt_state, key),
            None, length=num_updates,
        )
        final_metrics = jax.tree.map(lambda x: x[-1], epoch_metrics)
        return pp, vp, os, k, final_metrics

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    num_eval_envs = 128

    @jax.jit
    def jit_eval_rollout(policy_params, normalizer_params, eval_state, rng):
        def step_fn(carry, _):
            state, k = carry
            k, _ = jax.random.split(k)
            obs_norm = running_statistics.normalize(state.obs, normalizer_params)
            logits, _, _ = policy_module.apply(
                policy_params, obs_norm, k, deterministic=True
            )
            action = action_dist.mode(logits)
            next_state = test_env.step(state, action)
            return (next_state, k), (next_state.reward, next_state.done)

        (final_state, _), (rewards, _) = jax.lax.scan(
            step_fn, (eval_state, rng), None, length=episode_length,
        )
        mean_reward = jp.mean(jp.sum(rewards, axis=0))
        std_reward = jp.std(jp.sum(rewards, axis=0))
        return final_state, {
            "eval/episode_reward": mean_reward,
            "eval/episode_reward_std": std_reward,
            "eval/mean_step_reward": jp.mean(rewards),
        }

    jit_eval_reset = jax.jit(eval_env.reset)
    jit_eval_step = jax.jit(eval_env.step)

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def jit_policy_apply(params, obs, key, deterministic=False):
        return policy_module.apply(params, obs, key, deterministic=deterministic)

    # ------------------------------------------------------------------
    # EMG comparison: JIT-compiled batched rollout over all clips
    # ------------------------------------------------------------------

    # Compute target timesteps for EMG resampling (first 250ms of episode)
    emg_target_timesteps = min(int(EMG_DURATION_MS / 1000 / env_cfg.ctrl_dt), episode_length)

    # Pre-process biological EMG (static, done once)
    print("Loading biological EMG reference data...")
    emg_reference = load_emg_reference(
        n_emg_clips,
        emg_target_timesteps,
        clip_start_frame=int(env_cfg.start_frame_range[0]),
        norm_percentile=args.emg_norm_percentile,
    )

    @jax.jit
    def jit_emg_rollout(policy_params, normalizer_params, rng):
        """Batched deterministic rollout over all clips.

        Returns the internal muscle activation state (`data.act`) at each
        timestep rather than the raw policy action. Activation is the first-
        order-filtered version of the action (τ_act ~ 10 ms) and corresponds
        to what biological EMG actually measures, so this aligns the sim
        signal with the bio EMG temporally for honest comparison.
        """

        emg_start_frame = int(env_cfg.start_frame_range[0])

        def single_clip(clip_idx):
            clip_rng = jax.random.fold_in(rng, clip_idx)
            state = emg_env.reset(clip_rng, clip_idx=clip_idx, start_frame=emg_start_frame)
            state = state.replace(obs=flatten_obs(state.obs))

            def step_fn(carry, _):
                s, k = carry
                k, pk = jax.random.split(k)
                obs_norm = running_statistics.normalize(s.obs, normalizer_params)
                logits, _, _ = policy_module.apply(
                    policy_params, obs_norm[None], pk, deterministic=True
                )
                action = jp.squeeze(action_dist.mode(logits), axis=0)
                # sigmoid already outputs [0, 1] — no rescaling needed
                ns = emg_env.step(s, action)
                ns = ns.replace(obs=flatten_obs(ns.obs))
                # Use post-activation-filter muscle state, not the raw action,
                # so the comparison with bio EMG is apples-to-apples in time.
                muscle_activation = ns.data.act
                return (ns, k), muscle_activation

            _, actions = jax.lax.scan(
                step_fn, (state, clip_rng), None, length=episode_length
            )
            return actions  # (T, act_size) — now muscle activations

        all_actions = jax.vmap(single_clip)(jp.arange(n_emg_clips))
        return all_actions  # (n_clips, T, act_size)

    n_joints = proprio_size // 2  # qpos + qvel
    # Moving-shoulder xml qpos order: sh_tx, sh_ty, sh_tz (IK-driven),
    # then sh_rotation, sh_extension, sh_elv (humerus hinges), then elbow.
    joint_labels = (
        ["sh_tx", "sh_ty", "sh_tz", "sh_rot", "sh_ext", "sh_elv", "elbow"][:n_joints]
    )
    muscle_labels = [
        "Pec_C", "Lat", "AD", "PD", "MD",
        "Tri_Lat", "Tri_Long", "Brach", "Bic_Long",
        "Supra", "Infra", "Subscap",
    ][:act_size]

    def diagnostic_rollout(policy_params, normalizer_params, seed=0):
        """Single-episode rollout collecting states + per-step diagnostics."""
        rng = jax.random.PRNGKey(seed)
        state = jit_eval_reset(rng)
        rollout_states = [state]
        means_list, logvars_list = [], []
        rewards_list, actions_list = [], []
        proprio_list, task_obs_list = [], []
        qpos_list, ref_qpos_list = [], []
        reward_terms_list = []
        wrist_pos_list, ref_wrist_pos_list = [], []

        for _ in range(episode_length):
            flat = flatten_obs(state.obs)
            obs_norm = running_statistics.normalize(
                flat[None], normalizer_params
            )
            logits, mean, logvar = jit_policy_apply(
                policy_params, obs_norm, rng, deterministic=True
            )
            action = jp.squeeze(action_dist.mode(logits), axis=0)
            # sigmoid already outputs [0, 1] — no rescaling needed

            means_list.append(np.array(mean[0]))
            logvars_list.append(np.array(logvar[0]))
            actions_list.append(np.array(action))
            proprio_list.append(np.array(flat[:proprio_size]))
            task_obs_list.append(np.array(flat[proprio_size:]))

            # Actual joint positions
            qpos_list.append(np.array(state.data.qpos))

            # Reference joint positions
            frame_idx = eval_env._get_cur_frame(state.data, state.info)
            clip_idx = state.info["reference_clip"]
            ref = ref_clips.at(clip=clip_idx, frame=frame_idx)
            ref_qpos_list.append(np.array(ref.qpos))

            # Wrist positions (actual vs reference)
            try:
                wrist_id = eval_env._body_ids.get(eval_env._config.end_effector, None)
                if wrist_id is not None:
                    wrist_pos_list.append(np.array(state.data.xpos[wrist_id]))
                    ref_wrist_pos_list.append(np.array(ref.body_xpos(eval_env._config.end_effector)))
            except Exception:
                pass

            # Per-reward-term metrics from info
            step_rewards = {}
            for rkey in ["rewards/joints", "rewards/joints_vel", "rewards/wrist_pos",
                         "rewards/bodies_pos", "rewards/control_cost"]:
                if rkey in state.info.get("metrics", {}):
                    step_rewards[rkey] = float(state.info["metrics"][rkey])
            reward_terms_list.append(step_rewards)

            state = jit_eval_step(state, action)
            rollout_states.append(state)
            rewards_list.append(float(state.reward))

        episode_data = {
            "latent_mean": np.stack(means_list),
            "latent_logvar": np.stack(logvars_list),
            "reward": np.array(rewards_list),
            "action": np.stack(actions_list),
            "proprioception": np.stack(proprio_list),
            "task_obs": np.stack(task_obs_list),
            "qpos": np.stack(qpos_list),
            "ref_qpos": np.stack(ref_qpos_list),
            "reward_terms": reward_terms_list,
        }
        if wrist_pos_list:
            episode_data["wrist_pos"] = np.stack(wrist_pos_list)
            episode_data["ref_wrist_pos"] = np.stack(ref_wrist_pos_list)
        return rollout_states, episode_data

    def plot_intention_diagnostics(episode_data, save_path=None):
        """Multi-panel diagnostic figure for the intention bottleneck.

        4 rows x 3 cols = 12 panels:
          Row 0: Latent mean heatmap  | Latent std heatmap   | KL per step
          Row 1: ||μ|| over time      | AR(1) Δ over time    | Per-dim mean σ
          Row 2: Reward + cumulative  | Muscle activations   | Joint tracking error
          Row 3: Ref vs actual joints | Wrist 3D trajectory  | Reward term breakdown
        """
        means = episode_data["latent_mean"]
        logvars = episode_data["latent_logvar"]
        rewards = episode_data["reward"]
        actions = episode_data["action"]
        task_obs = episode_data["task_obs"]
        qpos = episode_data["qpos"]
        ref_qpos = episode_data["ref_qpos"]
        stds = np.exp(0.5 * logvars)
        T, D = means.shape

        fig, axes = plt.subplots(4, 3, figsize=(18, 16), squeeze=False)

        # ── Row 0, Col 0: Latent mean heatmap ──
        ax = axes[0, 0]
        vbound = max(np.percentile(np.abs(means), 95), 1e-6)
        im = ax.imshow(means.T, aspect="auto", cmap="RdBu_r",
                        interpolation="nearest", vmin=-vbound, vmax=vbound)
        ax.set_xlabel("env step")
        ax.set_ylabel("latent dim")
        ax.set_title("Latent mean μ")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # ── Row 0, Col 1: Latent std heatmap ──
        ax = axes[0, 1]
        im = ax.imshow(stds.T, aspect="auto", cmap="viridis",
                        interpolation="nearest")
        ax.set_xlabel("env step")
        ax.set_ylabel("latent dim")
        ax.set_title("Latent std σ")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # ── Row 0, Col 2: KL divergence per step ──
        ax = axes[0, 2]
        kl_per_step = -0.5 * np.sum(
            1 + logvars - means**2 - np.exp(logvars), axis=-1
        )
        ax.plot(kl_per_step, linewidth=0.8, color="steelblue")
        ax.fill_between(range(T), kl_per_step, alpha=0.2, color="steelblue")
        ax.set_xlabel("env step")
        ax.set_ylabel("KL(q || N(0,I))")
        ax.set_title(f"KL divergence  (mean={np.mean(kl_per_step):.2f})")

        # ── Row 1, Col 0: Latent mean magnitude ──
        ax = axes[1, 0]
        mean_norm = np.linalg.norm(means, axis=-1)
        ax.plot(mean_norm, linewidth=0.8, color="darkorange")
        ax.set_xlabel("env step")
        ax.set_ylabel("||μ||₂")
        ax.set_title("Latent mean magnitude")

        # ── Row 1, Col 1: AR(1) delta ──
        ax = axes[1, 1]
        if T > 1:
            ar1_delta = np.linalg.norm(means[1:] - means[:-1], axis=-1)
            ax.plot(ar1_delta, linewidth=0.8, color="forestgreen")
            ax.fill_between(range(T - 1), ar1_delta, alpha=0.2,
                            color="forestgreen")
            ax.set_title(f"AR(1) ||Δμ||₂  (mean={np.mean(ar1_delta):.3f})")
        else:
            ax.set_title("AR(1) ||Δμ||₂  (N/A)")
        ax.set_xlabel("env step")
        ax.set_ylabel("||μₜ - μₜ₋₁||₂")

        # ── Row 1, Col 2: Per-dimension mean σ ──
        ax = axes[1, 2]
        mean_std_per_dim = np.mean(stds, axis=0)
        colors = plt.cm.viridis(np.linspace(0, 1, D))
        ax.bar(range(D), mean_std_per_dim, color=colors, alpha=0.8)
        ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="prior σ=1")
        ax.set_xlabel("latent dim")
        ax.set_ylabel("mean σ")
        ax.set_title("Per-dimension latent std")
        ax.legend(fontsize=7)

        # ── Row 2, Col 0: Reward with cumulative ──
        ax = axes[2, 0]
        ax.plot(rewards, linewidth=0.8, color="crimson")
        ax.fill_between(range(T), rewards, alpha=0.15, color="crimson")
        cum_reward = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(cum_reward, linewidth=0.8, color="gray", alpha=0.6, ls="--")
        ax2.set_ylabel("cumulative", color="gray", fontsize=8)
        ax.set_xlabel("env step")
        ax.set_ylabel("reward")
        ax.set_title(f"Episode reward  (total={cum_reward[-1]:.1f})")

        # ── Row 2, Col 1: Muscle activation heatmap ──
        ax = axes[2, 1]
        im = ax.imshow(actions.T, aspect="auto", cmap="hot",
                        interpolation="nearest", vmin=0)
        ax.set_xlabel("env step")
        ax.set_ylabel("muscle")
        ax.set_yticks(range(min(len(muscle_labels), actions.shape[1])))
        ax.set_yticklabels(muscle_labels[:actions.shape[1]], fontsize=6)
        ax.set_title("Muscle activations")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # ── Row 2, Col 2: Joint tracking error ──
        ax = axes[2, 2]
        joint_delta = task_obs[:, :n_joints]
        for j in range(n_joints):
            ax.plot(np.abs(joint_delta[:, j]), linewidth=0.8,
                    label=joint_labels[j])
        ax.set_xlabel("env step")
        ax.set_ylabel("|joint target - current|")
        ax.set_title("Joint tracking error")
        ax.legend(fontsize=7)

        # ── Row 3, Col 0: Reference vs actual joint trajectories ──
        ax = axes[3, 0]
        for j in range(n_joints):
            color = f"C{j}"
            ax.plot(ref_qpos[:, j], linewidth=1.2, color=color, ls="--",
                    alpha=0.7, label=f"{joint_labels[j]} ref")
            ax.plot(qpos[:, j], linewidth=0.8, color=color,
                    label=f"{joint_labels[j]} actual")
        ax.set_xlabel("env step")
        ax.set_ylabel("joint angle (rad)")
        ax.set_title("Reference vs actual joints")
        ax.legend(fontsize=5, ncol=2)

        # ── Row 3, Col 1: Wrist 3D trajectory ──
        ax = axes[3, 1]
        if "wrist_pos" in episode_data:
            wrist = episode_data["wrist_pos"]
            ref_wrist = episode_data["ref_wrist_pos"]
            axis_labels = ["x", "y", "z"]
            for d in range(3):
                color = f"C{d}"
                ax.plot(ref_wrist[:, d], linewidth=1.2, color=color, ls="--",
                        alpha=0.7, label=f"{axis_labels[d]} ref")
                ax.plot(wrist[:, d], linewidth=0.8, color=color,
                        label=f"{axis_labels[d]} actual")
            wrist_err = np.linalg.norm(wrist - ref_wrist, axis=-1)
            ax.set_title(f"Wrist pos  (mean err={np.mean(wrist_err):.4f})")
            ax.legend(fontsize=5, ncol=2)
        else:
            ax.set_title("Wrist pos (N/A)")
        ax.set_xlabel("env step")
        ax.set_ylabel("position (m)")

        # ── Row 3, Col 2: Per-reward-term breakdown ──
        ax = axes[3, 2]
        reward_terms = episode_data["reward_terms"]
        if reward_terms and reward_terms[0]:
            term_keys = sorted(reward_terms[0].keys())
            for rkey in term_keys:
                vals = [rt.get(rkey, 0.0) for rt in reward_terms]
                short_name = rkey.replace("rewards/", "")
                ax.plot(vals, linewidth=0.8, label=short_name)
            ax.legend(fontsize=6, ncol=2)
        ax.set_xlabel("env step")
        ax.set_ylabel("reward")
        ax.set_title("Per-term reward breakdown")

        fig.suptitle("Janelia Intention Bottleneck Diagnostics", fontsize=13,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Rendering setup
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(args.seed)
    start_state = jit_eval_reset(rng)

    render_mj_model = build_render_model(eval_env)
    render_mj_data = mujoco.MjData(render_mj_model)
    renderer = mujoco.Renderer(render_mj_model, height=480, width=854)

    # Compute centroid for camera lookat from initial pose
    render_mj_data.qpos[:render_mj_model.nq // 2] = np.array(start_state.data.qpos)
    mujoco.mj_forward(render_mj_model, render_mj_data)
    centroid = render_mj_data.xpos[1:render_mj_model.nbody // 2 + 1].mean(axis=0)

    cam_right = np.array([-np.sin(np.radians(130)), np.cos(np.radians(130)), 0.0])
    cam_shift = -cam_right * 0.018

    render_cam = mujoco.MjvCamera()
    render_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    render_cam.lookat[:] = centroid + np.array([-0.008, 0.005, 0.008]) + cam_shift
    render_cam.distance = 0.05
    render_cam.azimuth = 130
    render_cam.elevation = -25

    ref_clips = eval_env.reference_clips

    def render_ghost_video(rollout_states, video_path):
        """Render video with ghost reference motion overlay."""
        with imageio.get_writer(video_path, fps=30) as video:
            for s in rollout_states[:-1]:  # skip terminal state
                frame_idx = eval_env._get_cur_frame(s.data, s.info)
                clip_idx = s.info["reference_clip"]
                ref = ref_clips.at(clip=clip_idx, frame=frame_idx)

                render_mj_data.qpos[:] = np.concatenate(
                    [np.array(s.data.qpos), np.array(ref.qpos)]
                )
                render_mj_data.qvel[:] = np.concatenate(
                    [np.array(s.data.qvel), np.array(ref.qvel)]
                )
                mujoco.mj_forward(render_mj_model, render_mj_data)
                renderer.update_scene(render_mj_data, camera=render_cam)
                video.append_data(renderer.render())

    # ------------------------------------------------------------------
    # Eval + logging
    # ------------------------------------------------------------------

    # Accumulate eval history for combined summary plot
    eval_history = {"steps": [], "reward": [],
                    "triceps_mae": [], "biceps_mae": []}

    def plot_combined_summary():
        """Reward + EMG MAE on one figure with dual y-axes."""
        h = eval_history
        if len(h["steps"]) < 1:
            return None
        steps_m = np.array(h["steps"]) / 1e6  # in millions
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Left axis: reward
        c_rew = "#2ca02c"
        ax1.plot(steps_m, h["reward"], color=c_rew, linewidth=2, marker="o",
                 markersize=4, label="Eval reward")
        ax1.set_xlabel("Training steps (M)")
        ax1.set_ylabel("Episode reward", color=c_rew)
        ax1.tick_params(axis="y", labelcolor=c_rew)

        # Right axis: MAE
        ax2 = ax1.twinx()
        if any(v is not None for v in h["triceps_mae"]):
            tri_vals = [v if v is not None else float("nan") for v in h["triceps_mae"]]
            ax2.plot(steps_m, tri_vals, color="#d62728", linewidth=2, marker="s",
                     markersize=4, label="Triceps trial MAE")
        if any(v is not None for v in h["biceps_mae"]):
            bic_vals = [v if v is not None else float("nan") for v in h["biceps_mae"]]
            ax2.plot(steps_m, bic_vals, color="#1f77b4", linewidth=2, marker="^",
                     markersize=4, label="Biceps trial MAE")
        ax2.set_ylabel("Trial MAE", color="#666666")
        ax2.tick_params(axis="y", labelcolor="#666666")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

        fig.suptitle("Reward + EMG MAE over training", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def eval_and_log(step, policy_params, value_params, normalizer_params):
        import time as _time
        t_eval = _time.time()

        eval_key_step = jax.random.PRNGKey(step)
        eval_state_reset = test_env.reset(
            jax.random.split(eval_key_step, num_eval_envs)
        )
        _, eval_metrics = jit_eval_rollout(
            policy_params, normalizer_params, eval_state_reset, eval_key_step,
        )
        jax.tree.map(lambda x: x.block_until_ready(), eval_metrics)
        eval_time = _time.time() - t_eval

        eval_log = {k: float(v) for k, v in eval_metrics.items()}
        eval_log["eval/eval_time"] = eval_time
        pprint(eval_log)

        # Collect all wandb data into one dict for a single log call
        wandb_log = dict(eval_log)

        # Diagnostic rollout
        rollout_states, episode_data = diagnostic_rollout(
            policy_params, normalizer_params, seed=step
        )

        # Ghost video
        try:
            video_path = f"{ckpt_path}/{step}.mp4"
            render_ghost_video(rollout_states, video_path)
            wandb_log["eval/rollout"] = wandb.Video(video_path, format="mp4")
            print(f"  video -> {video_path}")
        except Exception as e:
            print(f"  video failed: {e}")

        # Latent diagnostics
        try:
            diag_path = f"{ckpt_path}/{step}_latent_diag.png"
            fig = plot_intention_diagnostics(episode_data, save_path=diag_path)
            wandb_log["eval/latent_diagnostics"] = wandb.Image(fig)
            plt.close(fig)
            print(f"  latent diagnostics -> {diag_path}")
        except Exception as e:
            print(f"  latent diagnostics failed: {e}")

        # Latent summary scalars
        means = episode_data["latent_mean"]  # (T, D)
        logvars = episode_data["latent_logvar"]  # (T, D)
        has_nan = bool(np.any(np.isnan(means)) or np.any(np.isnan(logvars)))
        if has_nan:
            print(f"  WARNING: NaN in latent means/logvars — using nan-safe aggregations")
        stds = np.exp(0.5 * logvars)
        kl_per_dim = -0.5 * (1 + logvars - means**2 - np.exp(logvars))
        kl_total = float(np.nanmean(np.nansum(kl_per_dim, axis=-1)))
        mean_kl_per_dim = np.nanmean(kl_per_dim, axis=0)  # (D,)
        mean_std_per_dim = np.nanmean(stds, axis=0)  # (D,)
        mean_mean_per_dim = np.nanmean(means, axis=0)  # (D,)
        std_of_mean_per_dim = np.nanstd(means, axis=0)  # (D,) temporal variation
        active_dims = int(np.nansum(mean_kl_per_dim > 0.01))
        ar1_mean = float(np.nanmean(
            np.linalg.norm(np.nan_to_num(means[1:] - means[:-1]), axis=-1)
        )) if len(means) > 1 else 0.0

        wandb_log.update({
            "eval/latent_kl_mean": kl_total,
            "eval/latent_rate_nats": kl_total,
            "eval/latent_ar1_mean": ar1_mean,
            "eval/latent_mean_norm": float(np.nanmean(
                np.linalg.norm(np.nan_to_num(means), axis=-1)
            )),
            "eval/latent_std_mean": float(np.nanmean(stds)),
            "eval/latent_std_min": float(np.nanmin(mean_std_per_dim)),
            "eval/latent_std_max": float(np.nanmax(mean_std_per_dim)),
            "eval/latent_std_median": float(np.nanmedian(mean_std_per_dim)),
            "eval/latent_mean_abs_mean": float(np.nanmean(np.abs(means))),
            "eval/latent_mean_abs_max": float(np.nanmax(np.abs(mean_mean_per_dim))),
            "eval/latent_mean_temporal_std": float(np.nanmean(std_of_mean_per_dim)),
            "eval/active_latent_dims": active_dims,
            "eval/collapsed_dims": int(np.nansum(mean_kl_per_dim < 0.001)),
            "eval/episode_reward_single": float(np.nansum(
                episode_data["reward"]
            )),
        })

        # Histograms (skip if NaN)
        if not has_nan:
            wandb_log.update({
                "eval/hist_kl_per_dim": wandb.Histogram(mean_kl_per_dim),
                "eval/hist_std_per_dim": wandb.Histogram(mean_std_per_dim),
                "eval/hist_mean_per_dim": wandb.Histogram(mean_mean_per_dim),
                "eval/hist_mean_temporal_std": wandb.Histogram(std_of_mean_per_dim),
            })

        # Per-dimension scalar traces (for latent_size <= 64)
        D = means.shape[1]
        if D <= 64:
            for d in range(D):
                wandb_log[f"latent_dims/mean_d{d:02d}"] = float(mean_mean_per_dim[d])
                wandb_log[f"latent_dims/std_d{d:02d}"] = float(mean_std_per_dim[d])
                wandb_log[f"latent_dims/kl_d{d:02d}"] = float(mean_kl_per_dim[d])
                wandb_log[f"latent_dims/temporal_std_d{d:02d}"] = float(std_of_mean_per_dim[d])

        # EMG comparison (JIT-compiled batched rollout over all clips)
        if emg_reference is not None:
            try:
                import time as _t
                t_emg = _t.time()
                all_actions = jit_emg_rollout(
                    policy_params, normalizer_params, jax.random.PRNGKey(step)
                )
                all_actions = np.array(all_actions)  # (n_clips, T, act_size)

                # all_actions is actually MUSCLE ACTIVATION (ns.data.act), not
                # raw action. Activation is already in [0,1] by construction of
                # the Hill muscle model, but clip defensively for numerics.
                sim_actions = np.clip(all_actions[:, :emg_target_timesteps, :], 0.0, 1.0)

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

                # Co-contraction index (biceps * triceps)
                biceps_act = sim_actions[:, :, 8]
                triceps_act = sim_actions[:, :, 5]
                wandb_log["eval/emg_cocontraction"] = float(np.mean(biceps_act * triceps_act))
                wandb_log["eval/emg_time"] = _t.time() - t_emg

                print(f"  EMG comparison: " + ", ".join(
                    f"{k}(r={m['mean_corr']:.3f}, meanMAE={m['mean_mae']:.4f}"
                    f", trialMAE={m.get('trial_mae', 0):.4f})"
                    for k, m in emg_metrics.items()
                ))

                fig = plot_emg_comparison_fig(
                    sim_actions, emg_reference, emg_metrics, emg_target_timesteps,
                    ctrl_dt=env_cfg.ctrl_dt
                )
                wandb_log["eval/emg_comparison"] = wandb.Image(fig)
                plt.close(fig)

                # Single-trial overlays (sim vs bio, individual traces)
                single_fig = plot_emg_single_trials_fig(
                    sim_actions, emg_reference, emg_target_timesteps,
                    ctrl_dt=env_cfg.ctrl_dt, n_trials=4,
                )
                wandb_log["eval/emg_single_trials"] = wandb.Image(single_fig)
                plt.close(single_fig)

                # Per-timestep error plot (sim - bio)
                err_fig = plot_emg_error_fig(
                    sim_actions, emg_reference, emg_target_timesteps, env_cfg.ctrl_dt
                )
                wandb_log["eval/emg_error"] = wandb.Image(err_fig)
                plt.close(err_fig)

                # Action power spectrum (log-log) — uses FULL episode actions
                spec_fig = plot_action_spectrum_fig(all_actions, env_cfg.ctrl_dt)
                if spec_fig is not None:
                    wandb_log["eval/action_spectrum"] = wandb.Image(spec_fig)
                    plt.close(spec_fig)

                # Scalar spectral metrics
                spec_metrics = compute_spectral_metrics(all_actions, env_cfg.ctrl_dt)
                wandb_log.update(spec_metrics)
                if spec_metrics:
                    print(f"  Action spectrum: " + ", ".join(
                        f"{k.split('/')[-1]}={v:.3f}" for k, v in spec_metrics.items()
                    ))
            except Exception as e:
                print(f"  EMG comparison failed: {e}")

        # Accumulate history for combined summary plot
        eval_history["steps"].append(step)
        eval_history["reward"].append(
            wandb_log.get("eval/episode_reward", 0))
        eval_history["triceps_mae"].append(
            wandb_log.get("eval/emg_triceps_trial_mae"))
        eval_history["biceps_mae"].append(
            wandb_log.get("eval/emg_biceps_trial_mae"))
        try:
            combo_fig = plot_combined_summary()
            if combo_fig is not None:
                wandb_log["eval/reward_emg_summary"] = wandb.Image(combo_fig)
                plt.close(combo_fig)
        except Exception as e:
            print(f"  combined summary plot failed: {e}")

        # Single wandb.log call with everything
        if USE_WANDB:
            wandb.log(wandb_log, step=step)

        # Checkpoint: (normalizer, policy, value) for SCAMPER compat
        params_to_save = (normalizer_params, policy_params, value_params)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params_to_save)
        path = ckpt_path / f"{step}"
        orbax_checkpointer.save(path, params_to_save, force=True,
                                save_args=save_args)
        print(f"  checkpoint -> {path}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    print("Initialising training env state...")
    env_state = train_env.reset(jax.random.split(ek, num_envs))

    total_steps = 0
    steps_per_unroll = unroll_length * num_envs
    num_timesteps = ppo_params.num_timesteps
    num_evals = ppo_params.num_evals
    steps_per_eval = num_timesteps // num_evals
    next_eval_at = steps_per_eval

    print(f"Training for {num_timesteps:,} steps  "
          f"({steps_per_unroll:,} per unroll, eval every {steps_per_eval:,})")
    print(f"Temporal minibatch: {unroll_length} steps x {mb_env_size} envs "
          f"= {unroll_length * mb_env_size} samples")
    print("=" * 80)

    # Step-0 baseline eval skipped for sweep speed — uncomment for debugging
    # print("Running step-0 baseline eval...")
    # eval_and_log(0, policy_params, value_params, normalizer_params)

    import time as _time
    t0 = _time.time()
    log_every_steps = 10_000_000  # Log training metrics every ~10M steps
    next_log_at = log_every_steps

    while total_steps < num_timesteps:
        key, rollout_key = jax.random.split(key)
        env_state, rollout = collect_rollout(
            policy_params, value_params, normalizer_params,
            env_state, rollout_key,
        )
        total_steps += steps_per_unroll

        normalizer_params, advantages, returns = prepare_ppo_data(
            normalizer_params, rollout, env_state.obs, value_params,
        )

        key, update_key = jax.random.split(key)
        policy_params, value_params, opt_state, _, metrics = run_ppo_epochs(
            policy_params, value_params, opt_state, normalizer_params,
            rollout.obs, rollout.raw_action, rollout.log_prob,
            advantages, returns,
            rollout.done, rollout.truncation,
            update_key,
        )

        # -- Stream training metrics to wandb frequently --
        if total_steps >= next_log_at:
            elapsed = _time.time() - t0
            sps = total_steps / max(elapsed, 1e-6)
            log_metrics = {k: float(v) for k, v in metrics.items()}
            log_metrics["sps"] = sps
            log_metrics["total_steps"] = total_steps
            log_metrics["mean_reward"] = float(jp.mean(rollout.reward))
            if USE_WANDB:
                wandb.log(log_metrics, step=total_steps)
            # Print to console every 10M steps only
            if total_steps % 10_000_000 < log_every_steps:
                print(f"Step {total_steps:>12,}  SPS {sps:,.0f}  "
                      f"reward={log_metrics['mean_reward']:.2f}  "
                      f"kl={log_metrics.get('kl_loss', 0):.3f}  "
                      f"active_dims={log_metrics.get('active_latent_dims', 0):.0f}")
            next_log_at += log_every_steps

        # -- Full eval (videos + diagnostics + checkpoint) less frequently --
        if total_steps >= next_eval_at or total_steps >= num_timesteps:
            eval_and_log(total_steps, policy_params, value_params,
                         normalizer_params)
            next_eval_at += steps_per_eval

    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
