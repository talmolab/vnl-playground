"""Pairwise kinematic & EMG cosine similarity for the Physics-Aware checkpoint.

Spec: docs/superpowers/specs/2026-05-13-pairwise-similarity-fpca-design.md.

Produces, for the s18-ms-F4-fs1p2 checkpoint:
  fig_sim_heatmap_bio_kin           — 204x204 cosine, sorted by animal
  fig_sim_heatmap_bio_emg
  fig_sim_heatmap_sim_kin
  fig_sim_heatmap_sim_emg
  fig_sim_block_summary             — within vs between animal bars, all 4 modalities
  fig_fpca_modes                    — bio vs sim top-3 fPCA mode shapes per muscle
  fig_fpca_bio_basis_scatter        — bio + sim envelopes projected onto bio fPCA basis

Plus figs/similarity_rankings.npz with per-modality top-10 nearest neighbours.

Inputs (read from cache, no rollouts):
  vnl_playground/bayesian_emg/cache/v1/envelopes/<net>/<animal>.npz
  notebooks/talk_figures/figs/rollout_activations/<net>_278clips.npz
  vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/*_ik.h5
  /root/vast/eric/mouse-reach-mjx-neurips/trial_info/*.csv      (validity filter)
  /root/vast/eric/mouse-reach-mjx-neurips/emg_data/...csv       (length filter)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from glob import glob
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles

mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 8


# ── Constants ────────────────────────────────────────────────────────────
PHYSICS_RUN = "s18-ms-F4-fs1p2-20260502-014751"
ANIMALS = ("A36-1", "AT006", "AT009", "AT012", "AT013")
ANIMAL_COLORS = {
    "A36-1": "#1f77b4",
    "AT006": "#ff7f0e",
    "AT009": "#2ca02c",
    "AT012": "#d62728",
    "AT013": "#9467bd",
}

# Per-animal session id (matches scripts/emg_comparison.py).
ANIMAL_SESSIONS = {
    "A36-1": "A36-1_2023-07-18_16-54-01_lightOff_tone_on",
    "AT006": "AT006_2024-02-28_15-39-40_LightOff_tone_on",
    "AT009": "AT009_2024-04-25_13-50-34_LightOff_tone_on",
    "AT012": "AT012_2024-06-21_12-32-30_LightOff_video",
    "AT013": "AT013_2024-06-20_13-41-20_LightOff_tone_on",
}

TRIAL_INFO_DIR = Path("/root/vast/eric/mouse-reach-mjx-neurips/trial_info")
EMG_DIR = Path("/root/vast/eric/mouse-reach-mjx-neurips/emg_data")

REF_DIR = (
    REPO_ROOT
    / "vnl_playground"
    / "tasks"
    / "mouse"
    / "reference_data_moving_shoulder_v16_5animals"
)

# STAC XML has the `shoulder_base` body needed for FK frames matching kp_data.
IK_XML_PATH = Path(
    "/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml"
)

ENVELOPE_CACHE = REPO_ROOT / "vnl_playground" / "bayesian_emg" / "cache" / "v1" / "envelopes"
ROLLOUT_CACHE = Path(__file__).resolve().parent / "figs" / "rollout_activations"

KP_NAMES = ["Shoulder", "Elbow", "Wrist"]
KP_BODY = {"Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist"}

TARGET_T = 60          # canonical post-onset grid, matches EMG cache
MUSCLES = ("AD", "Triceps", "Biceps")
N_CLIPS_CACHE = 46     # bayes_emg_build_cache.py default (--n-clips)
DURATION_S = 0.25
EMG_FS = 30000
EMG_LEN_SAMPLES_LIMIT = 90000    # process_emg_data's hard cap


# ── Trial-number recovery (matches process_emg_data) ─────────────────────
def reach_window_passes(row: pd.Series) -> bool:
    """Replicates the emg_start/emg_end check in scripts/emg_comparison.py."""
    emg_start = int(1 / 200 * row["start"] * EMG_FS)
    emg_end = emg_start + int(DURATION_S * EMG_FS)
    return emg_start < EMG_LEN_SAMPLES_LIMIT and emg_end <= EMG_LEN_SAMPLES_LIMIT


def cached_trial_numbers(animal: str, n_in_cache: int) -> list[int]:
    """Return the list of original trial numbers for cache row 0..n_in_cache-1.

    Iterates the same valid_trials_df process_emg_data does, filters by the
    same reach-window check, and stops at n_in_cache survivors. (The cache
    builder used --n-clips=46; trials are filtered down to whatever passes,
    which is n_in_cache.)
    """
    info_path = TRIAL_INFO_DIR / f"{ANIMAL_SESSIONS[animal]}_off_trials_edited.csv"
    df = pd.read_csv(info_path)
    valid = df[~((df["start"] == 0) & (df["end"] == 0))]

    trial_nums: list[int] = []
    for i, (idx, row) in enumerate(valid.iterrows()):
        if i >= N_CLIPS_CACHE:
            break
        if not reach_window_passes(row):
            continue
        trial_nums.append(int(idx))
        if len(trial_nums) == n_in_cache:
            break
    if len(trial_nums) != n_in_cache:
        raise RuntimeError(
            f"{animal}: derived {len(trial_nums)} trial numbers but cache has "
            f"{n_in_cache}. Pipeline filter divergence."
        )
    return trial_nums


# ── Loaders ──────────────────────────────────────────────────────────────
def load_emg_envelopes(network: str, animal: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (sim, empirical) of shape (n_trials, 60, 3) for (AD, Triceps, Biceps)."""
    path = ENVELOPE_CACHE / network / f"{animal}.npz"
    with np.load(path) as z:
        sim = np.array(z["sim"])
        emp = np.array(z["empirical"])
        muscles = tuple(str(m) for m in z["muscles"].tolist())
    if muscles != MUSCLES:
        raise RuntimeError(
            f"{animal}: cache muscle order {muscles} != expected {MUSCLES}"
        )
    return sim, emp


def load_rollout_qpos(network: str) -> tuple[np.ndarray, list[Path]]:
    """Returns (qposes_rollout (278, 100, 7), sorted_clip_paths)."""
    cache_path = ROLLOUT_CACHE / f"{network}_278clips.npz"
    with np.load(cache_path) as z:
        qposes = np.array(z["qposes_rollout"])
    clip_paths = sorted(REF_DIR.glob("*_ik.h5"))
    if len(clip_paths) != qposes.shape[0]:
        raise RuntimeError(
            f"Rollout cache has {qposes.shape[0]} rows but {len(clip_paths)} "
            f"clips on disk; aborting."
        )
    return qposes, clip_paths


def disk_clip_index(clip_paths: list[Path], animal: str, trial_num: int) -> int:
    """Find the index into sorted clip_paths for (animal, trial_num)."""
    suffix = f"trial{trial_num:03d}_ik.h5"
    matches = [
        i for i, p in enumerate(clip_paths)
        if p.name.startswith(animal) and p.name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected 1 clip for {animal} trial {trial_num}, got {len(matches)}"
        )
    return matches[0]


def load_bio_kin(clip_path: Path) -> np.ndarray:
    """Load kp_data (50, 9), resample to (60, 9) along time."""
    with h5py.File(clip_path, "r") as f:
        kp = np.array(f["kp_data"][:])           # (50, 9)
    return resample_time(kp, TARGET_T)


def load_offsets(clip_path: Path) -> np.ndarray:
    with h5py.File(clip_path, "r") as f:
        return np.array(f["offsets"][:])          # (3, 3)


# ── Time resampling ──────────────────────────────────────────────────────
def resample_time(x: np.ndarray, target_T: int) -> np.ndarray:
    """Linearly resample axis 0 of x from N to target_T."""
    N = x.shape[0]
    if N == target_T:
        return x.astype(np.float32, copy=True)
    src = np.linspace(0, 1, N)
    dst = np.linspace(0, 1, target_T)
    flat = x.reshape(N, -1)
    out = np.empty((target_T, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    return out.reshape((target_T,) + x.shape[1:])


# ── FK ───────────────────────────────────────────────────────────────────
def make_fk_model():
    m = mujoco.MjModel.from_xml_path(str(IK_XML_PATH))
    d = mujoco.MjData(m)
    body_ids = {
        kp: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        for kp, body in KP_BODY.items()
    }
    return m, d, body_ids


def fk_markers_from_qpos(model, data, qpos_seq, offsets, body_ids):
    """qpos_seq (T, 7) -> (T, 3 kp, 3 dims)."""
    T = qpos_seq.shape[0]
    out = np.zeros((T, 3, 3), dtype=np.float32)
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for ki, kp in enumerate(KP_NAMES):
            bid = body_ids[kp]
            xpos = data.xpos[bid]
            xmat = data.xmat[bid].reshape(3, 3)
            out[t, ki] = xpos + xmat @ offsets[ki]
    return out


# ── Assembly ─────────────────────────────────────────────────────────────
def detrend_per_trial(seq: np.ndarray) -> np.ndarray:
    """Subtract the per-feature time mean from a (T, F) sequence.

    Removes the absolute-position baseline so cosine on the flattened result
    reflects trajectory shape, not which body frame the reach lived in.
    """
    return seq - seq.mean(axis=0, keepdims=True)


def build_feature_matrices(network: str):
    """Assemble row-aligned (bio_kin, bio_emg, sim_kin, sim_emg) over all cache trials.

    Kinematics are per-trial detrended (subtract per-feature time mean) so cosine
    captures trajectory shape, not the static body-frame offset.
    EMG envelopes are left as raw post-onset values — they are non-negative and
    cosine in [0, 1] is meaningful for shape comparison.

    Returns:
      X = dict with 'bio_kin' (N, 540), 'bio_emg' (N, 180), 'sim_kin' (N, 540), 'sim_emg' (N, 180).
      meta = dict with 'animal' (N,), 'trial_num' (N,), 'rollout_row' (N,), 'animal_counts'.
    """
    qposes, clip_paths = load_rollout_qpos(network)
    model, data, body_ids = make_fk_model()

    rows_bio_kin, rows_bio_emg = [], []
    rows_sim_kin, rows_sim_emg = [], []
    meta_animal, meta_trial, meta_rollout = [], [], []

    counts = {}
    for animal in ANIMALS:
        sim_env, emp_env = load_emg_envelopes(network, animal)
        n = sim_env.shape[0]
        trial_nums = cached_trial_numbers(animal, n)
        counts[animal] = n
        for i, tn in enumerate(trial_nums):
            r = disk_clip_index(clip_paths, animal, tn)
            offsets = load_offsets(clip_paths[r])
            qpos_100 = qposes[r]                       # (100, 7)
            qpos_60 = resample_time(qpos_100, TARGET_T)  # (60, 7)
            sim_markers = fk_markers_from_qpos(model, data, qpos_60, offsets, body_ids)  # (60, 3, 3)
            bio_kin = load_bio_kin(clip_paths[r])      # (60, 9)

            bio_kin_dt = detrend_per_trial(bio_kin)             # (60, 9)
            sim_kin_dt = detrend_per_trial(sim_markers.reshape(TARGET_T, -1))  # (60, 9)
            rows_bio_kin.append(bio_kin_dt.reshape(-1).astype(np.float32))
            rows_sim_kin.append(sim_kin_dt.reshape(-1).astype(np.float32))
            rows_bio_emg.append(emp_env[i].reshape(-1).astype(np.float32))
            rows_sim_emg.append(sim_env[i].reshape(-1).astype(np.float32))
            meta_animal.append(animal)
            meta_trial.append(int(tn))
            meta_rollout.append(int(r))

    X = {
        "bio_kin": np.stack(rows_bio_kin),     # (N, 540)
        "bio_emg": np.stack(rows_bio_emg),     # (N, 180)
        "sim_kin": np.stack(rows_sim_kin),     # (N, 540)
        "sim_emg": np.stack(rows_sim_emg),     # (N, 180)
    }
    meta = {
        "animal": np.array(meta_animal),
        "trial_num": np.array(meta_trial, dtype=np.int32),
        "rollout_row": np.array(meta_rollout, dtype=np.int32),
        "animal_counts": counts,
    }
    return X, meta


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize and return X @ X.T."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Xn = X / norms
    return Xn @ Xn.T


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_heatmap(S: np.ndarray, animals: np.ndarray, animal_counts: dict,
                 title: str, out_path: Path, vmin=-1.0, vmax=1.0):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(S, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="equal")
    # Animal block separators.
    cumulative = 0
    midpoints = []
    for a in ANIMALS:
        c = animal_counts[a]
        midpoints.append(cumulative + c / 2)
        cumulative += c
        if cumulative < S.shape[0]:
            ax.axhline(cumulative - 0.5, color="white", lw=0.6)
            ax.axvline(cumulative - 0.5, color="white", lw=0.6)
    ax.set_xticks(midpoints)
    ax.set_xticklabels(ANIMALS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(midpoints)
    ax.set_yticklabels(ANIMALS, fontsize=7)
    ax.set_title(title, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def block_stats(S: np.ndarray, animals: np.ndarray) -> dict:
    """Mean within-animal and between-animal cosine (off-diagonal only)."""
    N = S.shape[0]
    same = animals[:, None] == animals[None, :]
    eye = np.eye(N, dtype=bool)
    within = S[same & ~eye].mean()
    between = S[~same].mean()
    return {"within": float(within), "between": float(between), "gap": float(within - between)}


def pair_vectors(S_x: np.ndarray, S_y: np.ndarray, animals: np.ndarray):
    """Return (x, y, same_animal) over the upper triangle i<j of the matrices."""
    N = S_x.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    x = S_x[iu, ju]
    y = S_y[iu, ju]
    same = animals[iu] == animals[ju]
    return x, y, same


def select_marker_features(X_flat: np.ndarray, marker_idx: int) -> np.ndarray:
    """Reshape (N, 540) back to (N, T, n_markers, n_coords) and pick one marker.

    Order of axes when flattening:  T=60, n_markers=3 (Shoulder, Elbow, Wrist),
    n_coords=3. Returns (N, T*n_coords) = (N, 180).
    """
    X = X_flat.reshape(-1, TARGET_T, 3, 3)
    return X[:, :, marker_idx, :].reshape(-1, TARGET_T * 3)


def per_body_part_matrices(X_bio_full: np.ndarray, X_sim_full: np.ndarray) -> dict:
    """Return cosine matrices per (bio/sim, body_part)."""
    out = {}
    for mi, name in enumerate(KP_NAMES):
        out[("bio_kin", name)] = cosine_matrix(select_marker_features(X_bio_full, mi))
        out[("sim_kin", name)] = cosine_matrix(select_marker_features(X_sim_full, mi))
    return out


def select_muscle_features(X_flat: np.ndarray, muscle_idx: int) -> np.ndarray:
    """X_flat: (N, 180) packed as (T=60, M=3). Returns (N, T) for one muscle."""
    X = X_flat.reshape(-1, TARGET_T, len(MUSCLES))
    return X[:, :, muscle_idx]  # (N, T)


def per_muscle_matrices(X_bio_emg: np.ndarray, X_sim_emg: np.ndarray) -> dict:
    """Return cosine matrices per (bio/sim, muscle)."""
    out = {}
    for mi, name in enumerate(MUSCLES):
        out[("bio_emg", name)] = cosine_matrix(select_muscle_features(X_bio_emg, mi))
        out[("sim_emg", name)] = cosine_matrix(select_muscle_features(X_sim_emg, mi))
    return out


def plot_body_part_distributions(S_part: dict, animals: np.ndarray, out_path: Path):
    """2 rows (bio, sim) x 3 cols (Shoulder, Elbow, Wrist).

    Each panel: histogram of within-animal vs between-animal cosines for that
    (modality, body-part) pair.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=False)
    for ri, modality in enumerate(["bio_kin", "sim_kin"]):
        for ci, name in enumerate(KP_NAMES):
            ax = axes[ri, ci]
            Sm = S_part[(modality, name)]
            N = Sm.shape[0]
            same = animals[:, None] == animals[None, :]
            eye = np.eye(N, dtype=bool)
            within = Sm[same & ~eye]
            between = Sm[~same]
            lo = min(within.min(), between.min())
            hi = max(within.max(), between.max())
            bins = np.linspace(lo, hi, 60)
            ax.hist(between, bins=bins, color="#888888", alpha=0.55, density=True,
                    label=f"between (n={between.size})")
            ax.hist(within, bins=bins, color="#d62728", alpha=0.55, density=True,
                    label=f"within  (n={within.size})")
            ax.axvline(np.median(within), color="#d62728", lw=1, ls="--")
            ax.axvline(np.median(between), color="#888888", lw=1, ls="--")
            ax.set_title(
                f"{modality} — {name}\n"
                f"within={np.median(within):+.3f}  between={np.median(between):+.3f}  "
                f"gap={np.median(within) - np.median(between):+.3f}",
                fontsize=8,
            )
            if ri == 1:
                ax.set_xlabel("cosine", fontsize=8)
            if ci == 0:
                ax.set_ylabel("density", fontsize=8)
            ax.legend(fontsize=6, loc="best")
    fig.suptitle("Per-body-part cosine distributions", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_per_muscle_distributions(S_mu: dict, animals: np.ndarray, out_path: Path):
    """2 rows (bio_emg, sim_emg) x 3 cols (AD, Triceps, Biceps): histogram of cosines."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False, sharey=False)
    for ri, modality in enumerate(["bio_emg", "sim_emg"]):
        for ci, name in enumerate(MUSCLES):
            ax = axes[ri, ci]
            Sm = S_mu[(modality, name)]
            N = Sm.shape[0]
            same = animals[:, None] == animals[None, :]
            eye = np.eye(N, dtype=bool)
            within = Sm[same & ~eye]
            between = Sm[~same]
            lo = min(within.min(), between.min())
            hi = max(within.max(), between.max())
            bins = np.linspace(lo, hi, 60)
            ax.hist(between, bins=bins, color="#888888", alpha=0.55, density=True,
                    label=f"between (n={between.size})")
            ax.hist(within, bins=bins, color="#d62728", alpha=0.55, density=True,
                    label=f"within  (n={within.size})")
            ax.axvline(np.median(within), color="#d62728", lw=1, ls="--")
            ax.axvline(np.median(between), color="#888888", lw=1, ls="--")
            ax.set_title(
                f"{modality} — {name}\n"
                f"within={np.median(within):+.3f}  between={np.median(between):+.3f}  "
                f"gap={np.median(within) - np.median(between):+.3f}",
                fontsize=8,
            )
            if ri == 1:
                ax.set_xlabel("cosine", fontsize=8)
            if ci == 0:
                ax.set_ylabel("density", fontsize=8)
            ax.legend(fontsize=6, loc="best")
    fig.suptitle("Per-muscle EMG cosine distributions", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_per_muscle_per_animal(S_mu: dict, animals: np.ndarray, out_path: Path):
    """Per-animal violins, one row per muscle, two cols per modality."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=False)
    for ri, name in enumerate(MUSCLES):
        for ci, modality in enumerate(["bio_emg", "sim_emg"]):
            ax = axes[ri, ci]
            dists = per_animal_distributions(S_mu[(modality, name)], animals)
            positions, data, colors = [], [], []
            for ai, a in enumerate(ANIMALS):
                within = dists[a]["within"]
                between = dists[a]["between"]
                positions.extend([ai * 3, ai * 3 + 1])
                data.extend([within, between])
                colors.extend([ANIMAL_COLORS[a], "#bbbbbb"])
            parts = ax.violinplot(data, positions=positions, widths=0.85,
                                  showmedians=True, showextrema=False)
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.4)
                pc.set_alpha(0.75)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(0.8)
            ax.set_xticks([ai * 3 + 0.5 for ai in range(len(ANIMALS))])
            ax.set_xticklabels(ANIMALS, fontsize=7)
            ax.set_ylabel("cosine", fontsize=8)
            ax.set_title(f"{modality} — {name}", fontsize=9)
            ax.axhline(0, color="black", lw=0.3)
            for ai in range(1, len(ANIMALS)):
                ax.axvline(ai * 3 - 0.5, color="#dddddd", lw=0.5)
    fig.suptitle("Per-animal EMG cosine by muscle (within = colored, between = gray)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_body_part_per_animal(S_part: dict, animals: np.ndarray, out_path: Path):
    """Per-animal violins of within/between cosines, one row per body part, two cols per modality."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=False)
    for ri, name in enumerate(KP_NAMES):
        for ci, modality in enumerate(["bio_kin", "sim_kin"]):
            ax = axes[ri, ci]
            dists = per_animal_distributions(S_part[(modality, name)], animals)
            positions, data, colors = [], [], []
            for ai, a in enumerate(ANIMALS):
                within = dists[a]["within"]
                between = dists[a]["between"]
                positions.extend([ai * 3, ai * 3 + 1])
                data.extend([within, between])
                colors.extend([ANIMAL_COLORS[a], "#bbbbbb"])
            parts = ax.violinplot(data, positions=positions, widths=0.85,
                                  showmedians=True, showextrema=False)
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.4)
                pc.set_alpha(0.75)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(0.8)
            ax.set_xticks([ai * 3 + 0.5 for ai in range(len(ANIMALS))])
            ax.set_xticklabels(ANIMALS, fontsize=7)
            ax.set_ylabel("cosine", fontsize=8)
            ax.set_title(f"{modality} — {name}", fontsize=9)
            ax.axhline(0, color="black", lw=0.3)
            for ai in range(1, len(ANIMALS)):
                ax.axvline(ai * 3 - 0.5, color="#dddddd", lw=0.5)
    fig.suptitle("Per-animal cosine by body part (within = colored, between = gray)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_global_distributions(S: dict[str, np.ndarray], animals: np.ndarray,
                              out_path: Path):
    """One histogram per modality of all off-diagonal cosines, split within/between."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    modalities = ["bio_kin", "bio_emg", "sim_kin", "sim_emg"]
    for ax, modality in zip(axes, modalities):
        Sm = S[modality]
        N = Sm.shape[0]
        same = animals[:, None] == animals[None, :]
        eye = np.eye(N, dtype=bool)
        within = Sm[same & ~eye]
        between = Sm[~same]
        lo = min(within.min(), between.min())
        hi = max(within.max(), between.max())
        bins = np.linspace(lo, hi, 80)
        ax.hist(between, bins=bins, color="#888888", alpha=0.55, density=True,
                label=f"between-animal (n={between.size})")
        ax.hist(within, bins=bins, color="#d62728", alpha=0.55, density=True,
                label=f"within-animal  (n={within.size})")
        ax.axvline(np.median(within), color="#d62728", lw=1, ls="--")
        ax.axvline(np.median(between), color="#888888", lw=1, ls="--")
        ax.set_xlabel("cosine", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(
            f"{modality}\nmedians: within={np.median(within):+.3f}  "
            f"between={np.median(between):+.3f}  "
            f"gap={np.median(within) - np.median(between):+.3f}",
            fontsize=8,
        )
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Cosine distributions: within- vs between-animal", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_cross_modal_scatter(S: dict[str, np.ndarray], animals: np.ndarray,
                             out_path: Path):
    """2x2 grid of pairwise-cosine scatters across the four modalities.

    Panel A: bio_kin x bio_emg  — Bernstein redundancy: do similar bio kinematics
                                   imply similar bio EMG?
    Panel B: sim_kin x sim_emg  — Same question inside the physics-aware net.
    Panel C: bio_kin x sim_kin  — Does the net agree with biology on kin similarity?
    Panel D: bio_emg x sim_emg  — Does the net agree with biology on EMG similarity?
    """
    from scipy.stats import spearmanr, pearsonr

    panels = [
        ("bio_kin", "bio_emg", "Bernstein bio: kin -> EMG"),
        ("sim_kin", "sim_emg", "Bernstein sim: kin -> EMG"),
        ("bio_kin", "sim_kin", "kin: bio vs sim"),
        ("bio_emg", "sim_emg", "EMG: bio vs sim"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()
    for ax, (xk, yk, title) in zip(axes, panels):
        x, y, same = pair_vectors(S[xk], S[yk], animals)
        rho_s, p_s = spearmanr(x, y)
        rho_p, p_p = pearsonr(x, y)
        ax.scatter(x[~same], y[~same], s=4, alpha=0.18, color="#888888",
                   linewidths=0, label=f"between-animal (n={(~same).sum()})")
        ax.scatter(x[same], y[same], s=6, alpha=0.45, color="#d62728",
                   linewidths=0, label=f"within-animal (n={same.sum()})")
        ax.set_xlabel(f"{xk} cosine", fontsize=8)
        ax.set_ylabel(f"{yk} cosine", fontsize=8)
        ax.set_title(
            f"{title}\nSpearman r={rho_s:.3f} (p={p_s:.1e})  "
            f"Pearson r={rho_p:.3f} (p={p_p:.1e})",
            fontsize=8,
        )
        ax.axhline(0, color="black", lw=0.3)
        ax.axvline(0, color="black", lw=0.3)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle("Cross-modal pairwise similarity (each dot = one reach-pair)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)

    return {
        f"{xk}_x_{yk}": {
            "spearman": float(spearmanr(*pair_vectors(S[xk], S[yk], animals)[:2])[0]),
            "pearson": float(pearsonr(*pair_vectors(S[xk], S[yk], animals)[:2])[0]),
        }
        for xk, yk, _ in panels
    }


def per_animal_distributions(S: np.ndarray, animals: np.ndarray) -> dict:
    """For each animal A, collect within-A and between-A off-diagonal cosines."""
    N = S.shape[0]
    eye = np.eye(N, dtype=bool)
    out = {}
    for a in ANIMALS:
        m = animals == a
        in_block = S[np.ix_(m, m)]                       # (n_a, n_a)
        in_off = in_block[~np.eye(in_block.shape[0], dtype=bool)]
        out_block = S[np.ix_(m, ~m)]                     # (n_a, N - n_a)
        out[a] = {"within": in_off.ravel(), "between": out_block.ravel()}
    return out


def plot_per_animal_distributions(S_dict: dict[str, np.ndarray], animals: np.ndarray,
                                  out_path: Path):
    """One subplot per modality: violin of within/between cosines per animal."""
    modalities = ["bio_kin", "bio_emg", "sim_kin", "sim_emg"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=False)
    axes = axes.ravel()

    for ax, modality in zip(axes, modalities):
        dists = per_animal_distributions(S_dict[modality], animals)
        positions, data, colors, labels = [], [], [], []
        for ai, a in enumerate(ANIMALS):
            within = dists[a]["within"]
            between = dists[a]["between"]
            positions.extend([ai * 3, ai * 3 + 1])
            data.extend([within, between])
            colors.extend([ANIMAL_COLORS[a], "#bbbbbb"])
            labels.extend([f"{a}\nwithin", f"{a}\nbetween"])
        parts = ax.violinplot(data, positions=positions, widths=0.85,
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.4)
            pc.set_alpha(0.75)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(0.8)
        ax.set_xticks([ai * 3 + 0.5 for ai in range(len(ANIMALS))])
        ax.set_xticklabels(ANIMALS, fontsize=8)
        ax.set_ylabel("cosine", fontsize=8)
        ax.set_title(modality, fontsize=9)
        ax.axhline(0, color="black", lw=0.3)
        for ai in range(1, len(ANIMALS)):
            ax.axvline(ai * 3 - 0.5, color="#dddddd", lw=0.5)
        # Per-animal medians as a line annotation.
        within_medians = [np.median(dists[a]["within"]) for a in ANIMALS]
        between_medians = [np.median(dists[a]["between"]) for a in ANIMALS]
        anchor = [ai * 3 for ai in range(len(ANIMALS))]
        ax.plot(anchor, within_medians, "k-", lw=0.6, alpha=0.5)
        ax.plot([ai * 3 + 1 for ai in range(len(ANIMALS))], between_medians,
                color="#666666", lw=0.6, ls="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="black", label="within-animal (colored)"),
        Patch(facecolor="#bbbbbb", edgecolor="black", label="between-animal (gray)"),
    ]
    fig.suptitle("Per-animal cosine similarity (within vs between)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_block_summary(stats: dict[str, dict], out_path: Path):
    modalities = ["bio_kin", "bio_emg", "sim_kin", "sim_emg"]
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    x = np.arange(len(modalities))
    w = 0.28
    within_v = [stats[m]["within"] for m in modalities]
    between_v = [stats[m]["between"] for m in modalities]
    gap_v = [stats[m]["gap"] for m in modalities]
    ax.bar(x - w, within_v, w, label="within-animal", color="#1f77b4")
    ax.bar(x, between_v, w, label="between-animal", color="#aec7e8")
    ax.bar(x + w, gap_v, w, label="gap (within − between)", color="#ff7f0e")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=8)
    ax.set_ylabel("mean cosine", fontsize=8)
    ax.set_title("Within- vs between-animal similarity", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


# ── fPCA ─────────────────────────────────────────────────────────────────
def fit_fpca(X: np.ndarray, var_target: float = 0.85):
    """L2-normalize per row, center per feature, SVD; pick k for cumvar ≥ var_target.

    Row L2-normalization removes per-trial amplitude variance so the PCs capture
    envelope SHAPE, not amplitude. Without this, EMG PC1 is dominated by overall
    activation magnitude and k jumps to ~20 for 85% variance.

    Returns (V (D, k), k, mean (D,), cumulative_variance (min(N,D),)).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Xn = X / norms
    mean = Xn.mean(axis=0)
    Xc = Xn - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, var_target) + 1)
    return Vt[:k].T, k, mean, cum


def plot_fpca_modes(V_bio: np.ndarray, V_sim: np.ndarray, k_show: int, out_path: Path):
    """Reshape each basis vector (180,) → (60, 3) and overlay bio vs sim per muscle."""
    fig, axes = plt.subplots(3, k_show, figsize=(2.0 * k_show, 4.5), sharex=True)
    if k_show == 1:
        axes = axes.reshape(3, 1)
    t = np.linspace(0, DURATION_S * 1000, TARGET_T)
    for j in range(k_show):
        mode_bio = V_bio[:, j].reshape(TARGET_T, 3)
        mode_sim = V_sim[:, j].reshape(TARGET_T, 3)
        # Sign-align to bio so plots compare like-with-like.
        s = np.sign(np.sum(mode_bio * mode_sim))
        if s == 0:
            s = 1
        mode_sim = mode_sim * s
        for mi, mname in enumerate(MUSCLES):
            ax = axes[mi, j]
            ax.plot(t, mode_bio[:, mi], color="#8E44AD", lw=1.2, label="bio" if (j == 0 and mi == 0) else None)
            ax.plot(t, mode_sim[:, mi], color="#1f77b4", lw=1.2, ls="--", label="sim" if (j == 0 and mi == 0) else None)
            if j == 0:
                ax.set_ylabel(mname, fontsize=8)
            if mi == 0:
                ax.set_title(f"PC{j+1}", fontsize=8)
            if mi == 2:
                ax.set_xlabel("time (ms)", fontsize=8)
            ax.axhline(0, color="black", lw=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_fpca_scatter(X_bio: np.ndarray, X_sim: np.ndarray, V_bio: np.ndarray,
                      mean_bio: np.ndarray, animals: np.ndarray, out_path: Path):
    """Project bio + sim onto V_bio, plot PC1 vs PC2 with animal color, bio/sim marker.

    Both bio and sim envelopes are L2-normalized (matching fit_fpca) and
    centered by the bio mean before projection.
    """
    def _normalize(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n < 1e-12, 1.0, n)
    Z_bio = (_normalize(X_bio) - mean_bio) @ V_bio  # (N, k)
    Z_sim = (_normalize(X_sim) - mean_bio) @ V_bio  # use bio mean for both — keeps axes comparable
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for a in ANIMALS:
        m = animals == a
        ax.scatter(Z_bio[m, 0], Z_bio[m, 1], s=14, marker="o",
                   facecolors=ANIMAL_COLORS[a], edgecolors="black", linewidths=0.3,
                   alpha=0.7, label=f"{a} bio")
        ax.scatter(Z_sim[m, 0], Z_sim[m, 1], s=14, marker="x",
                   color=ANIMAL_COLORS[a], alpha=0.7, label=f"{a} sim")
    ax.set_xlabel("bio fPCA PC1", fontsize=8)
    ax.set_ylabel("bio fPCA PC2", fontsize=8)
    ax.set_title("EMG envelopes projected onto bio fPCA basis", fontsize=9)
    ax.legend(fontsize=6, loc="best", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


# ── Rankings ─────────────────────────────────────────────────────────────
def topk_rankings(S: np.ndarray, animals: np.ndarray, k: int = 10):
    """For each row i, return top-k neighbors (excluding self)."""
    N = S.shape[0]
    masked = S.copy()
    np.fill_diagonal(masked, -np.inf)
    idx = np.argsort(-masked, axis=1)[:, :k]
    cos = np.take_along_axis(masked, idx, axis=1)
    nbr_animal = animals[idx]
    return idx, cos, nbr_animal


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default=PHYSICS_RUN)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "figs"))
    ap.add_argument("--var-target", type=float, default=0.85)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[similarity] network={args.network}  out_dir={out_dir}")

    print("[similarity] building feature matrices (this includes per-clip FK) …")
    X, meta = build_feature_matrices(args.network)
    N = X["bio_kin"].shape[0]
    print(f"[similarity] N = {N}; per-animal counts: {meta['animal_counts']}")
    for k, M in X.items():
        print(f"  X[{k}].shape = {M.shape}")

    print("[similarity] pairwise cosine …")
    S = {k: cosine_matrix(M) for k, M in X.items()}
    for k, Sk in S.items():
        diag = np.diag(Sk)
        if not np.allclose(diag, 1.0, atol=1e-4):
            raise RuntimeError(f"{k}: diagonal not 1: min={diag.min()}, max={diag.max()}")
        print(f"  S[{k}]: shape={Sk.shape}, off-diag range=[{(Sk[~np.eye(N, dtype=bool)]).min():.3f}, "
              f"{(Sk[~np.eye(N, dtype=bool)]).max():.3f}]")

    print("[similarity] heatmaps …")
    titles = {
        "bio_kin": "Bio kinematics (STAC kp_data)",
        "bio_emg": "Bio EMG envelopes",
        "sim_kin": "Sim kinematics (FK rollout)",
        "sim_emg": "Sim muscle activations",
    }
    for k in ("bio_kin", "bio_emg", "sim_kin", "sim_emg"):
        # Kinematics rows can have negative entries (positions span signs);
        # EMG envelopes are non-negative so cosines live in [0, 1].
        vmin = 0.0 if k.endswith("emg") else -1.0
        plot_heatmap(
            S[k], meta["animal"], meta["animal_counts"],
            title=titles[k], out_path=out_dir / f"fig_sim_heatmap_{k}",
            vmin=vmin, vmax=1.0,
        )

    print("[similarity] block stats …")
    stats = {k: block_stats(S[k], meta["animal"]) for k in S}
    print(f"  {'modality':<10s}  within   between  gap")
    for k, st in stats.items():
        print(f"  {k:<10s}  {st['within']:+.3f}  {st['between']:+.3f}  {st['gap']:+.3f}")
    plot_block_summary(stats, out_dir / "fig_sim_block_summary")

    print("[similarity] global cosine distributions (within vs between) …")
    plot_global_distributions(S, meta["animal"], out_dir / "fig_sim_distributions_global")

    print("[similarity] per-body-part decomposition …")
    S_part = per_body_part_matrices(X["bio_kin"], X["sim_kin"])
    plot_body_part_distributions(S_part, meta["animal"], out_dir / "fig_sim_body_part_hist")
    plot_body_part_per_animal(S_part, meta["animal"], out_dir / "fig_sim_body_part_per_animal")
    print(f"  {'modality':<10s} {'body':<9s} within   between  gap")
    for modality in ["bio_kin", "sim_kin"]:
        for name in KP_NAMES:
            st = block_stats(S_part[(modality, name)], meta["animal"])
            print(f"  {modality:<10s} {name:<9s} {st['within']:+.3f}  "
                  f"{st['between']:+.3f}  {st['gap']:+.3f}")

    print("[similarity] per-muscle decomposition …")
    S_mu = per_muscle_matrices(X["bio_emg"], X["sim_emg"])
    plot_per_muscle_distributions(S_mu, meta["animal"], out_dir / "fig_sim_muscle_hist")
    plot_per_muscle_per_animal(S_mu, meta["animal"], out_dir / "fig_sim_muscle_per_animal")
    print(f"  {'modality':<10s} {'muscle':<9s} within   between  gap")
    for modality in ["bio_emg", "sim_emg"]:
        for name in MUSCLES:
            st = block_stats(S_mu[(modality, name)], meta["animal"])
            print(f"  {modality:<10s} {name:<9s} {st['within']:+.3f}  "
                  f"{st['between']:+.3f}  {st['gap']:+.3f}")

    print("[similarity] cross-modal scatter (Bernstein probe + bio-vs-sim) …")
    cross = plot_cross_modal_scatter(S, meta["animal"], out_dir / "fig_sim_cross_modal")
    for k, v in cross.items():
        print(f"  {k}: Spearman={v['spearman']:+.3f}, Pearson={v['pearson']:+.3f}")

    print("[similarity] per-animal distributions …")
    plot_per_animal_distributions(S, meta["animal"], out_dir / "fig_sim_distributions_per_animal")
    # Per-animal medians table.
    print(f"  {'modality':<10s}  {'animal':<6s}  within_med  between_med  gap")
    for modality in ("bio_kin", "bio_emg", "sim_kin", "sim_emg"):
        dists = per_animal_distributions(S[modality], meta["animal"])
        for a in ANIMALS:
            wm = float(np.median(dists[a]["within"]))
            bm = float(np.median(dists[a]["between"]))
            print(f"  {modality:<10s}  {a:<6s}  {wm:+.3f}      {bm:+.3f}       {wm - bm:+.3f}")

    print("[similarity] fPCA (bio EMG vs sim EMG) …")
    V_bio, k_bio, mean_bio, cum_bio = fit_fpca(X["bio_emg"], var_target=args.var_target)
    V_sim, k_sim, mean_sim, cum_sim = fit_fpca(X["sim_emg"], var_target=args.var_target)
    print(f"  k_bio = {k_bio} (cumvar @ k = {cum_bio[k_bio-1]:.3f})")
    print(f"  k_sim = {k_sim} (cumvar @ k = {cum_sim[k_sim-1]:.3f})")
    k_common = min(k_bio, k_sim)
    angles = subspace_angles(V_bio[:, :k_common], V_sim[:, :k_common])
    print(f"  principal angles (rad): {angles}")
    print(f"  cos(angles):            {np.cos(angles)}")
    print(f"  mean(angles) deg = {np.degrees(angles).mean():.2f}")

    k_show = min(3, k_common)
    plot_fpca_modes(V_bio[:, :k_show], V_sim[:, :k_show], k_show, out_dir / "fig_fpca_modes")
    plot_fpca_scatter(
        X["bio_emg"], X["sim_emg"], V_bio[:, :max(2, k_show)],
        mean_bio, meta["animal"], out_dir / "fig_fpca_bio_basis_scatter",
    )

    print("[similarity] rankings …")
    rankings = {}
    for k in S:
        idx, cos, nbr = topk_rankings(S[k], meta["animal"])
        rankings[f"{k}_idx"] = idx
        rankings[f"{k}_cos"] = cos
        rankings[f"{k}_nbr_animal"] = nbr
    np.savez(
        out_dir / "similarity_rankings.npz",
        animal=meta["animal"],
        trial_num=meta["trial_num"],
        rollout_row=meta["rollout_row"],
        **rankings,
    )
    print(f"[similarity] wrote {out_dir/'similarity_rankings.npz'}")

    # Sanity check: bio_kin within > between (the weakest of our predictions).
    gap = stats["bio_kin"]["gap"]
    if gap < 0.02:
        print(f"[similarity] WARN: bio_kin gap = {gap:+.3f} < 0.02 — check sorting / labels.")

    print("[similarity] done.")


if __name__ == "__main__":
    main()
