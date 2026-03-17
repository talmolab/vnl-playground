"""Psychometric curve analysis for RunGap corridor data.

Computes success rate, head kinematics metrics, and approach behavior as
functions of gap distance, following Parker et al. (eLife 2022).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from vnl_playground.tasks.rodent.analysis.head_kinematics import ApproachWindow
from vnl_playground.tasks.rodent.analysis.motion_parallax_analysis import (
    compute_approach_duration,
    compute_head_pitch_stats,
    compute_movement_amplitude,
    compute_total_head_distance,
    count_vertical_movements,
)

# ---------------------------------------------------------------------------
# Condition colour mapping (Parker et al. convention)
# ---------------------------------------------------------------------------

CONDITION_COLORS: dict[str, str] = {
    "binocular": "tab:blue",
    "monocular_left": "tab:pink",
    "monocular_right": "tab:red",
}

# Numeric metric keys used throughout the module.
_NUMERIC_METRICS = [
    "gap_length",
    "n_vertical_movements",
    "movement_amplitude",
    "mean_pitch",
    "std_pitch",
    "range_pitch",
    "total_head_distance",
    "approach_duration",
    "mean_forward_velocity",
]


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_gap_metrics(
    approach_windows: list[ApproachWindow],
    skull_z_traces: list[np.ndarray],
    pitch_traces: list[np.ndarray],
    skull_position_traces: list[np.ndarray],
    torso_velocity_traces: list[np.ndarray],
    dt: float = 0.01,
) -> list[dict]:
    """Compute per-gap metrics from approach kinematics.

    Each element of the input lists corresponds to one gap approach.

    Args:
        approach_windows: List of :class:`ApproachWindow` instances.
        skull_z_traces: Per-gap 1-D arrays of skull z-position.
        pitch_traces: Per-gap 1-D arrays of head pitch in degrees.
        skull_position_traces: Per-gap arrays of shape ``(T, 3)`` with skull
            positions.
        torso_velocity_traces: Per-gap arrays of shape ``(T, D)`` with torso
            velocities.  The first column (index 0) is forward velocity.
        dt: Simulation timestep in seconds.

    Returns:
        List of metric dicts, one per gap.
    """
    metrics_list: list[dict] = []

    for aw, skull_z, pitch, skull_pos, torso_vel in zip(
        approach_windows,
        skull_z_traces,
        pitch_traces,
        skull_position_traces,
        torso_velocity_traces,
    ):
        pitch_stats = compute_head_pitch_stats(pitch)
        n_vert = count_vertical_movements(skull_z, dt=dt)
        amp = compute_movement_amplitude(skull_z)
        head_dist = compute_total_head_distance(skull_pos)
        duration = compute_approach_duration(torso_vel, dt=dt)

        torso_vel = np.asarray(torso_vel)
        if torso_vel.ndim == 1:
            mean_fwd_vel = float(np.mean(torso_vel))
        else:
            mean_fwd_vel = float(np.mean(torso_vel[:, 0]))

        metrics_list.append(
            {
                "gap_index": aw.gap_index,
                "gap_length": aw.gap_length,
                "crossed": aw.crossed_successfully,
                "n_vertical_movements": n_vert,
                "movement_amplitude": amp,
                "mean_pitch": pitch_stats["mean_pitch"],
                "std_pitch": pitch_stats["std_pitch"],
                "range_pitch": pitch_stats["range_pitch"],
                "total_head_distance": head_dist,
                "approach_duration": duration,
                "mean_forward_velocity": mean_fwd_vel,
            }
        )

    return metrics_list


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def bin_metrics_by_gap_distance(
    metrics_list: list[dict],
    n_bins: int = 5,
    gap_range: tuple[float, float] = (0.03, 0.12),
) -> dict:
    """Bin per-gap metrics by gap distance.

    Args:
        metrics_list: Output of :func:`compute_gap_metrics`.
        n_bins: Number of evenly spaced bins across *gap_range*.
        gap_range: ``(min, max)`` gap distance defining the bin edges.

    Returns:
        Dict mapping *bin_center* (float, rounded to 6 decimals) to a dict
        containing ``"success_rate"``, ``"n_gaps"``, and for each numeric
        metric a sub-dict ``{"mean": float, "sem": float}``.
    """
    edges = np.linspace(gap_range[0], gap_range[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Pre-assign each metric entry to a bin.
    bins: dict[int, list[dict]] = {i: [] for i in range(n_bins)}

    for m in metrics_list:
        gl = m["gap_length"]
        idx = int(np.searchsorted(edges[1:], gl, side="right"))
        idx = min(idx, n_bins - 1)
        bins[idx].append(m)

    result: dict[float, dict] = {}
    for i, center in enumerate(centers):
        entries = bins[i]
        n_gaps = len(entries)
        key = round(float(center), 6)

        if n_gaps == 0:
            bin_result: dict = {"n_gaps": 0, "success_rate": np.nan}
            for metric_name in _NUMERIC_METRICS:
                bin_result[metric_name] = {"mean": np.nan, "sem": np.nan}
            result[key] = bin_result
            continue

        success_rate = float(np.mean([e["crossed"] for e in entries]))

        bin_result = {"n_gaps": n_gaps, "success_rate": success_rate}

        for metric_name in _NUMERIC_METRICS:
            values = np.array([e[metric_name] for e in entries], dtype=float)
            mean_val = float(np.mean(values))
            sem_val = float(np.std(values, ddof=1) / np.sqrt(n_gaps)) if n_gaps > 1 else 0.0
            bin_result[metric_name] = {"mean": mean_val, "sem": sem_val}

        result[key] = bin_result

    return result


# ---------------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------------


def compare_conditions(condition_data: dict[str, list[dict]]) -> dict:
    """Bin metrics for each viewing condition.

    Args:
        condition_data: Mapping from condition name (e.g. ``"binocular"``)
            to the corresponding list of metric dicts.

    Returns:
        Nested dict: ``condition -> binned_results`` as returned by
        :func:`bin_metrics_by_gap_distance`.
    """
    return {
        condition: bin_metrics_by_gap_distance(metrics)
        for condition, metrics in condition_data.items()
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _get_color(condition: str) -> str:
    """Return the colour for a condition, falling back to grey."""
    return CONDITION_COLORS.get(condition, "tab:gray")


def _sorted_bin_items(binned: dict):
    """Yield (center, data) pairs sorted by bin center."""
    for k in sorted(binned.keys()):
        yield k, binned[k]


# ---------------------------------------------------------------------------
# Psychometric curve: success rate vs gap distance
# ---------------------------------------------------------------------------


def plot_psychometric_curves(
    comparison_data: dict[str, dict],
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Plot success rate vs gap distance for each condition.

    Args:
        comparison_data: Output of :func:`compare_conditions`.
        ax: Optional axes to draw on.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    for condition, binned in comparison_data.items():
        centers = []
        rates = []
        for c, data in _sorted_bin_items(binned):
            centers.append(c)
            rates.append(data["success_rate"])

        color = _get_color(condition)
        ax.plot(centers, rates, "o-", color=color, label=condition)

    ax.set_xlabel("Gap distance (m)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Psychometric curve")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Head movement comparison (bar chart across conditions)
# ---------------------------------------------------------------------------


def plot_head_movement_comparison(
    comparison_data: dict[str, dict],
    metric_name: str,
    ylabel: str,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Bar chart of a metric averaged across all gap distances per condition.

    Args:
        comparison_data: Output of :func:`compare_conditions`.
        metric_name: Key into the binned metric dicts (e.g.
            ``"n_vertical_movements"``).
        ylabel: Label for the y-axis.
        ax: Optional axes to draw on.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    conditions = list(comparison_data.keys())
    means = []
    sems = []

    for cond in conditions:
        binned = comparison_data[cond]
        all_means = []
        all_ns = []
        for _, data in _sorted_bin_items(binned):
            n = data["n_gaps"]
            if n == 0:
                continue
            all_means.append(data[metric_name]["mean"])
            all_ns.append(n)

        if len(all_means) == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            weights = np.array(all_ns, dtype=float)
            vals = np.array(all_means, dtype=float)
            grand_mean = float(np.average(vals, weights=weights))
            # SEM across bins (unweighted, treating bins as samples)
            grand_sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            means.append(grand_mean)
            sems.append(grand_sem)

    x = np.arange(len(conditions))
    colors = [_get_color(c) for c in conditions]
    ax.bar(x, means, yerr=sems, color=colors, capsize=4, edgecolor="black", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(metric_name.replace("_", " ").title())
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metric vs gap distance (line plot)
# ---------------------------------------------------------------------------


def plot_metric_vs_gap_distance(
    comparison_data: dict[str, dict],
    metric_name: str,
    ylabel: str,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Line plot of a metric vs gap distance for each condition.

    Args:
        comparison_data: Output of :func:`compare_conditions`.
        metric_name: Key into the binned metric dicts.
        ylabel: Label for the y-axis.
        ax: Optional axes to draw on.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    for condition, binned in comparison_data.items():
        centers = []
        vals = []
        errs = []
        for c, data in _sorted_bin_items(binned):
            if data["n_gaps"] == 0:
                continue
            centers.append(c)
            vals.append(data[metric_name]["mean"])
            errs.append(data[metric_name]["sem"])

        color = _get_color(condition)
        ax.errorbar(centers, vals, yerr=errs, fmt="o-", color=color, label=condition, capsize=3)

    ax.set_xlabel("Gap distance (m)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(metric_name.replace("_", " ").title())
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Parker et al. multi-panel comparison figure
# ---------------------------------------------------------------------------


def plot_parker_comparison_panel(
    comparison_data: dict[str, dict],
    output_path: Optional[str] = None,
) -> Figure:
    """Multi-panel figure replicating key Parker et al. analyses.

    Layout (2 rows x 4 cols):
        [0,0] Success rate vs gap distance (Fig 2B top)
        [0,1] Mean forward velocity vs gap distance (Fig 2B bottom analog)
        [0,2] Vertical movement frequency per condition (Fig 3D)
        [0,3] Movement amplitude per condition (Fig 3E)
        [1,0] Mean head pitch per condition (Fig 3I)
        [1,1] Pitch range per condition
        [1,2] Total head distance per condition (Fig 3H)
        [1,3] Approach duration per condition (Fig 3G)

    Args:
        comparison_data: Output of :func:`compare_conditions`.
        output_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # Row 0, Col 0: Psychometric curve
    plot_psychometric_curves(comparison_data, ax=axes[0, 0])

    # Row 0, Col 1: Mean forward velocity vs gap distance
    plot_metric_vs_gap_distance(
        comparison_data,
        metric_name="mean_forward_velocity",
        ylabel="Forward velocity (m/s)",
        ax=axes[0, 1],
    )

    # Row 0, Col 2: Vertical movements (bar)
    plot_head_movement_comparison(
        comparison_data,
        metric_name="n_vertical_movements",
        ylabel="Count",
        ax=axes[0, 2],
    )

    # Row 0, Col 3: Movement amplitude (bar)
    plot_head_movement_comparison(
        comparison_data,
        metric_name="movement_amplitude",
        ylabel="Amplitude (m)",
        ax=axes[0, 3],
    )

    # Row 1, Col 0: Mean head pitch (bar)
    plot_head_movement_comparison(
        comparison_data,
        metric_name="mean_pitch",
        ylabel="Pitch (deg)",
        ax=axes[1, 0],
    )

    # Row 1, Col 1: Pitch range (bar)
    plot_head_movement_comparison(
        comparison_data,
        metric_name="range_pitch",
        ylabel="Pitch range (deg)",
        ax=axes[1, 1],
    )

    # Row 1, Col 2: Total head distance (bar)
    plot_head_movement_comparison(
        comparison_data,
        metric_name="total_head_distance",
        ylabel="Distance (m)",
        ax=axes[1, 2],
    )

    # Row 1, Col 3: Approach duration (bar)
    plot_head_movement_comparison(
        comparison_data,
        metric_name="approach_duration",
        ylabel="Duration (s)",
        ax=axes[1, 3],
    )

    fig.suptitle("Parker et al. comparison panel", fontsize=14, y=1.01)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
