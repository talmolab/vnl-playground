"""Neural and behavioral analysis pipeline for gap-jump experiments.

Provides tools for analyzing CNN ("Virtual V1") and RNN ("Virtual Decision
Circuit") representations, replicating the analysis from Liska et al.

Analysis components:
1. Psychometric curves (success rate vs. gap distance)
2. Decision time analysis (decision duration vs. gap distance)
3. Head movement analysis (kinematics during DECISION phase)
4. RNN hidden state analysis (confidence accumulation in GRU)
5. CNN feature analysis (distance encoding in V1)
6. ARHMM behavioral motif analysis
7. Latent intention trajectory analysis
"""

from collections import defaultdict
from typing import Optional

import numpy as np
from scipy import stats

# Use TYPE_CHECKING to avoid hard dependency on matplotlib at import time
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import matplotlib
    from matplotlib.figure import Figure


# ============================================================
# 1. Psychometric Curves
# ============================================================

def compute_psychometric_data(
    trial_data: list,
    conditions: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Compute psychometric curve data (success rate vs gap distance).

    Args:
        trial_data: List of TrialData objects.
        conditions: Condition names to include (None = all).

    Returns:
        Dict mapping condition -> {
            "distances": array of gap distances,
            "success_rates": array of success rates,
            "n_trials": array of trial counts per distance,
            "ci_lower": array of 95% CI lower bounds,
            "ci_upper": array of 95% CI upper bounds,
        }
    """
    by_condition = defaultdict(list)
    for t in trial_data:
        if conditions is None or t.condition in conditions:
            by_condition[t.condition].append(t)

    results = {}
    for condition, trials in by_condition.items():
        # Group by gap distance
        by_distance = defaultdict(lambda: {"success": 0, "total": 0})
        for t in trials:
            by_distance[t.gap_distance]["total"] += 1
            if t.outcome == "success":
                by_distance[t.gap_distance]["success"] += 1

        distances = sorted(by_distance.keys())
        success_rates = []
        n_trials_arr = []
        ci_lower = []
        ci_upper = []

        for d in distances:
            n = by_distance[d]["total"]
            k = by_distance[d]["success"]
            rate = k / max(n, 1)
            success_rates.append(rate)
            n_trials_arr.append(n)

            # Wilson score interval for binomial proportion
            if n > 0:
                lo, hi = _wilson_ci(k, n, alpha=0.05)
            else:
                lo, hi = 0.0, 1.0
            ci_lower.append(lo)
            ci_upper.append(hi)

        results[condition] = {
            "distances": np.array(distances),
            "success_rates": np.array(success_rates),
            "n_trials": np.array(n_trials_arr),
            "ci_lower": np.array(ci_lower),
            "ci_upper": np.array(ci_upper),
        }

    return results


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def fit_psychometric_sigmoid(
    distances: np.ndarray,
    success_rates: np.ndarray,
) -> dict:
    """Fit logistic sigmoid to psychometric data.

    Model: p(success) = 1 / (1 + exp(-(a + b * distance)))

    Returns:
        Dict with "a", "b", "threshold_50" (distance at 50% success),
        and "fitted_curve" (fine-grained predictions).
    """
    from scipy.optimize import curve_fit

    def sigmoid(x, a, b):
        return 1.0 / (1.0 + np.exp(-(a + b * x)))

    try:
        popt, pcov = curve_fit(sigmoid, distances, success_rates, p0=[5.0, -50.0])
        a, b = popt
        threshold_50 = -a / b if abs(b) > 1e-8 else np.nan
        x_fine = np.linspace(distances.min(), distances.max(), 100)
        fitted = sigmoid(x_fine, a, b)
        return {
            "a": a, "b": b,
            "threshold_50": threshold_50,
            "x_fine": x_fine,
            "fitted_curve": fitted,
        }
    except RuntimeError:
        return {"a": np.nan, "b": np.nan, "threshold_50": np.nan,
                "x_fine": distances, "fitted_curve": success_rates}


def plot_psychometric_curves(
    psychometric_data: dict[str, dict],
    title: str = "Psychometric Curves",
    figsize: tuple = (8, 5),
) -> "Figure":
    """Plot success rate vs gap distance for each condition.

    Matches paper Fig. 2 style.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    colors = {"binocular": "black", "monocular_left": "blue",
              "monocular_right": "cyan", "v1_suppression": "red",
              "v1_suppression_50": "orange", "monocular_left_v1": "purple"}

    for condition, data in psychometric_data.items():
        color = colors.get(condition, "gray")
        ax.errorbar(
            data["distances"] * 100,  # Convert to cm
            data["success_rates"],
            yerr=[data["success_rates"] - data["ci_lower"],
                  data["ci_upper"] - data["success_rates"]],
            fmt="o-", color=color, label=condition, capsize=3, markersize=6,
        )

    ax.set_xlabel("Gap Distance (cm)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# 2. Decision Time Analysis
# ============================================================

def compute_decision_time_data(
    trial_data: list,
    conditions: Optional[list[str]] = None,
    success_only: bool = True,
) -> dict[str, dict]:
    """Compute decision time statistics per gap distance and condition.

    Returns:
        Dict mapping condition -> {
            "distances": gap distances,
            "mean_times": mean decision times (seconds),
            "std_times": std of decision times,
            "sem_times": standard error of means,
            "all_times": dict of distance -> list of times,
        }
    """
    by_condition = defaultdict(list)
    for t in trial_data:
        if conditions is None or t.condition in conditions:
            if success_only and t.outcome != "success":
                continue
            if t.decision_time_steps > 0:
                by_condition[t.condition].append(t)

    results = {}
    for condition, trials in by_condition.items():
        by_dist = defaultdict(list)
        for t in trials:
            by_dist[t.gap_distance].append(t.decision_time_seconds)

        distances = sorted(by_dist.keys())
        mean_times = [np.mean(by_dist[d]) for d in distances]
        std_times = [np.std(by_dist[d]) for d in distances]
        sem_times = [np.std(by_dist[d]) / np.sqrt(len(by_dist[d]))
                     for d in distances]

        results[condition] = {
            "distances": np.array(distances),
            "mean_times": np.array(mean_times),
            "std_times": np.array(std_times),
            "sem_times": np.array(sem_times),
            "all_times": dict(by_dist),
        }

    return results


def plot_decision_times(
    decision_data: dict[str, dict],
    title: str = "Decision Time vs Gap Distance",
    figsize: tuple = (8, 5),
) -> "Figure":
    """Plot decision time vs gap distance. Paper Fig. 3 style."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    colors = {"binocular": "black", "monocular_left": "blue",
              "monocular_right": "cyan", "v1_suppression": "red"}

    for condition, data in decision_data.items():
        color = colors.get(condition, "gray")
        ax.errorbar(
            data["distances"] * 100,
            data["mean_times"],
            yerr=data["sem_times"],
            fmt="o-", color=color, label=condition, capsize=3,
        )

    ax.set_xlabel("Gap Distance (cm)")
    ax.set_ylabel("Decision Time (s)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# 3. Head Movement Analysis
# ============================================================

def analyze_head_movements(
    trial_data: list,
    dt: float = 0.02,
) -> dict:
    """Analyze head kinematics during DECISION phase.

    Extracts pitch, yaw, and roll velocities from skull tracking.
    Paper found more vertical movements under monocular conditions.

    Returns:
        Dict with per-condition head movement statistics.
    """
    by_condition = defaultdict(list)
    for t in trial_data:
        if t.head_positions is not None and len(t.head_positions) > 1:
            positions = t.head_positions  # [T, 3]
            velocities = np.diff(positions, axis=0) / dt
            speed = np.linalg.norm(velocities, axis=1)

            # Classify movement components
            vertical_vel = np.abs(velocities[:, 2])  # z-component
            horizontal_vel = np.linalg.norm(velocities[:, :2], axis=1)

            by_condition[t.condition].append({
                "mean_speed": np.mean(speed),
                "mean_vertical": np.mean(vertical_vel),
                "mean_horizontal": np.mean(horizontal_vel),
                "vertical_ratio": np.mean(vertical_vel) / (np.mean(speed) + 1e-8),
                "n_movements": _count_zero_crossings(velocities[:, 2]),
                "gap_distance": t.gap_distance,
            })

    results = {}
    for condition, movements in by_condition.items():
        results[condition] = {
            "mean_speed": np.mean([m["mean_speed"] for m in movements]),
            "mean_vertical": np.mean([m["mean_vertical"] for m in movements]),
            "mean_horizontal": np.mean([m["mean_horizontal"] for m in movements]),
            "vertical_ratio": np.mean([m["vertical_ratio"] for m in movements]),
            "mean_n_movements": np.mean([m["n_movements"] for m in movements]),
            "raw": movements,
        }

    return results


def _count_zero_crossings(signal: np.ndarray) -> int:
    """Count zero crossings in a signal."""
    return int(np.sum(np.diff(np.sign(signal)) != 0))


# ============================================================
# 4. RNN Hidden State Analysis ("Confidence Accumulation")
# ============================================================

def analyze_rnn_confidence(
    trial_data: list,
    n_components: int = 10,
    condition: str = "binocular",
) -> dict:
    """Analyze GRU hidden states for evidence of confidence accumulation.

    Methods:
    1. PCA on GRU hidden states across all DECISION timesteps
    2. Identify PCs correlated with time-to-jump (confidence dimension)
    3. Check for ramping activity

    Args:
        trial_data: List of TrialData.
        n_components: Number of PCA components.
        condition: Which condition to analyze.

    Returns:
        Dict with PCA results, ramping analysis, and distance encoding.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    # Collect all GRU hidden states from successful trials
    all_hidden = []
    all_gap_dists = []
    all_time_to_jump = []  # normalized time within decision period
    all_trial_ids = []

    trial_id = 0
    for t in trial_data:
        if (t.condition == condition and t.gru_hidden_states is not None
                and t.outcome == "success" and len(t.gru_hidden_states) > 0):
            T = len(t.gru_hidden_states)
            for step_idx in range(T):
                all_hidden.append(t.gru_hidden_states[step_idx])
                all_gap_dists.append(t.gap_distance)
                all_time_to_jump.append(1.0 - step_idx / max(T - 1, 1))
                all_trial_ids.append(trial_id)
            trial_id += 1

    if len(all_hidden) < n_components:
        return {"error": "Not enough data for PCA"}

    X = np.array(all_hidden)
    gap_dists = np.array(all_gap_dists)
    time_to_jump = np.array(all_time_to_jump)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Find "confidence dimension": PC most correlated with time-to-jump
    confidence_correlations = []
    for i in range(n_components):
        r, p = stats.pearsonr(X_pca[:, i], time_to_jump)
        confidence_correlations.append({"pc": i, "r": r, "p": p})
    confidence_dim = max(confidence_correlations, key=lambda x: abs(x["r"]))

    # Find "distance dimension": PC most correlated with gap distance
    distance_correlations = []
    for i in range(n_components):
        r, p = stats.pearsonr(X_pca[:, i], gap_dists)
        distance_correlations.append({"pc": i, "r": r, "p": p})
    distance_dim = max(distance_correlations, key=lambda x: abs(x["r"]))

    # Ramping analysis: linear regression of top PC over time
    ramping_results = {}
    for i in range(min(3, n_components)):
        reg = LinearRegression().fit(time_to_jump.reshape(-1, 1), X_pca[:, i])
        ramping_results[f"pc{i}"] = {
            "slope": float(reg.coef_[0]),
            "r2": float(reg.score(time_to_jump.reshape(-1, 1), X_pca[:, i])),
        }

    # Distance decoding: linear regression from hidden states to gap distance
    dist_reg = LinearRegression().fit(X, gap_dists)
    dist_r2 = dist_reg.score(X, gap_dists)

    return {
        "pca": pca,
        "X_pca": X_pca,
        "gap_distances": gap_dists,
        "time_to_jump": time_to_jump,
        "trial_ids": np.array(all_trial_ids),
        "explained_variance": pca.explained_variance_ratio_,
        "confidence_dim": confidence_dim,
        "distance_dim": distance_dim,
        "confidence_correlations": confidence_correlations,
        "distance_correlations": distance_correlations,
        "ramping": ramping_results,
        "distance_decoding_r2": dist_r2,
    }


def plot_rnn_trajectories(
    rnn_analysis: dict,
    pc_x: int = 0,
    pc_y: int = 1,
    color_by: str = "gap_distance",
    figsize: tuple = (8, 6),
) -> "Figure":
    """Plot PCA trajectories of RNN hidden states colored by gap distance or time."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    X_pca = rnn_analysis["X_pca"]
    trial_ids = rnn_analysis["trial_ids"]

    if color_by == "gap_distance":
        colors = rnn_analysis["gap_distances"]
        cmap = "viridis"
        label = "Gap Distance (m)"
    else:
        colors = rnn_analysis["time_to_jump"]
        cmap = "coolwarm"
        label = "Time to Jump (normalized)"

    unique_trials = np.unique(trial_ids)
    for tid in unique_trials:
        mask = trial_ids == tid
        trial_pca = X_pca[mask]
        trial_colors = colors[mask]
        sc = ax.scatter(trial_pca[:, pc_x], trial_pca[:, pc_y],
                       c=trial_colors, cmap=cmap, s=5, alpha=0.5)
        ax.plot(trial_pca[:, pc_x], trial_pca[:, pc_y],
               alpha=0.2, linewidth=0.5, color="gray")

    plt.colorbar(sc, ax=ax, label=label)
    ev = rnn_analysis["explained_variance"]
    ax.set_xlabel(f"PC{pc_x+1} ({ev[pc_x]*100:.1f}% var)")
    ax.set_ylabel(f"PC{pc_y+1} ({ev[pc_y]*100:.1f}% var)")
    ax.set_title("GRU Hidden State Trajectories")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# 5. CNN Feature Analysis ("Virtual V1")
# ============================================================

def analyze_cnn_features(
    trial_data: list,
    condition: str = "binocular",
) -> dict:
    """Analyze CNN features for distance encoding.

    Methods:
    1. Linear regression: CNN features -> gap distance
    2. Representational similarity analysis (RSA)

    Returns:
        Dict with distance decoding R2 and RSA results.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    all_features = []
    all_distances = []

    for t in trial_data:
        if (t.condition == condition and t.cnn_features is not None
                and t.outcome == "success"):
            # Use mean CNN features across decision period
            mean_feat = np.mean(t.cnn_features, axis=0)
            all_features.append(mean_feat)
            all_distances.append(t.gap_distance)

    if len(all_features) < 10:
        return {"error": "Not enough data"}

    X = np.array(all_features)
    y = np.array(all_distances)

    # Linear distance decoding
    reg = LinearRegression()
    cv_scores = cross_val_score(reg, X, y, cv=5, scoring="r2")
    reg.fit(X, y)

    # RSA: compute representational dissimilarity matrix
    unique_dists = np.unique(y)
    mean_patterns = {}
    for d in unique_dists:
        mask = y == d
        mean_patterns[d] = np.mean(X[mask], axis=0)

    n_dists = len(unique_dists)
    rdm = np.zeros((n_dists, n_dists))
    for i, d1 in enumerate(unique_dists):
        for j, d2 in enumerate(unique_dists):
            rdm[i, j] = 1.0 - np.corrcoef(mean_patterns[d1], mean_patterns[d2])[0, 1]

    # Physical distance matrix
    phys_rdm = np.abs(unique_dists[:, None] - unique_dists[None, :])
    # RSA: correlation between neural RDM and physical RDM
    mask = np.triu_indices(n_dists, k=1)
    rsa_r, rsa_p = stats.pearsonr(rdm[mask], phys_rdm[mask]) if len(mask[0]) > 2 else (0, 1)

    return {
        "distance_decoding_r2": float(np.mean(cv_scores)),
        "cv_scores": cv_scores,
        "rsa_correlation": float(rsa_r),
        "rsa_p_value": float(rsa_p),
        "rdm": rdm,
        "unique_distances": unique_dists,
    }


# ============================================================
# 6. Latent Intention Analysis
# ============================================================

def analyze_latent_intentions(
    trial_data: list,
    condition: str = "binocular",
    n_components: int = 5,
) -> dict:
    """Analyze latent intention (z) trajectories during decision period.

    Key question: Does z converge to a consistent "jump intention" before
    the jump, and does this intention vary with gap distance?

    Returns:
        Dict with convergence analysis and distance-dependent patterns.
    """
    from sklearn.decomposition import PCA

    # Collect final latent z (just before jump) for each successful trial
    final_z_by_dist = defaultdict(list)
    all_z_trajectories = []
    all_dists = []

    for t in trial_data:
        if (t.condition == condition and t.latent_z is not None
                and t.outcome == "success" and len(t.latent_z) > 0):
            final_z_by_dist[t.gap_distance].append(t.latent_z[-1])
            all_z_trajectories.append(t.latent_z)
            all_dists.append(t.gap_distance)

    if not all_z_trajectories:
        return {"error": "No data"}

    # Convergence: variance of final z across trials (per distance)
    convergence = {}
    for dist, z_list in final_z_by_dist.items():
        Z = np.array(z_list)
        convergence[dist] = {
            "mean_variance": float(np.mean(np.var(Z, axis=0))),
            "n_trials": len(z_list),
        }

    # Distance dependence: can we predict gap distance from final z?
    from sklearn.linear_model import LinearRegression
    all_final_z = np.array([t[-1] for t in all_z_trajectories])
    all_dists_arr = np.array(all_dists)

    if len(all_final_z) > 5:
        reg = LinearRegression().fit(all_final_z, all_dists_arr)
        dist_r2 = reg.score(all_final_z, all_dists_arr)
    else:
        dist_r2 = 0.0

    # PCA on final z
    if len(all_final_z) > n_components:
        pca = PCA(n_components=n_components)
        z_pca = pca.fit_transform(all_final_z)
    else:
        pca = None
        z_pca = all_final_z

    return {
        "convergence": convergence,
        "distance_prediction_r2": dist_r2,
        "pca": pca,
        "z_pca": z_pca,
        "distances": all_dists_arr,
    }


# ============================================================
# 7. Summary report
# ============================================================

def generate_analysis_report(
    trial_data: list,
    conditions: Optional[list[str]] = None,
) -> dict:
    """Generate a complete analysis report across all components.

    Returns:
        Dict with all analysis results.
    """
    report = {}

    # Psychometric curves
    report["psychometric"] = compute_psychometric_data(trial_data, conditions)

    # Decision times
    report["decision_times"] = compute_decision_time_data(trial_data, conditions)

    # Head movements
    report["head_movements"] = analyze_head_movements(trial_data)

    # RNN analysis (binocular baseline)
    report["rnn_confidence"] = analyze_rnn_confidence(
        trial_data, condition="binocular",
    )

    # CNN analysis
    report["cnn_features"] = analyze_cnn_features(trial_data, condition="binocular")

    # Latent intentions
    report["latent_intentions"] = analyze_latent_intentions(
        trial_data, condition="binocular",
    )

    return report


def print_report_summary(report: dict):
    """Print a human-readable summary of the analysis report."""
    print("\n" + "=" * 70)
    print("GAP-JUMP NEURAL ANALYSIS REPORT")
    print("=" * 70)

    # Psychometric curves
    if "psychometric" in report:
        print("\n--- Psychometric Curves ---")
        for cond, data in report["psychometric"].items():
            mean_sr = np.mean(data["success_rates"])
            print(f"  {cond}: mean success rate = {mean_sr:.2%}")

    # Decision times
    if "decision_times" in report:
        print("\n--- Decision Times ---")
        for cond, data in report["decision_times"].items():
            mean_dt = np.mean(data["mean_times"])
            print(f"  {cond}: mean decision time = {mean_dt:.3f}s")

    # RNN confidence
    if "rnn_confidence" in report and "error" not in report["rnn_confidence"]:
        rnn = report["rnn_confidence"]
        print("\n--- RNN Hidden State Analysis ---")
        print(f"  Distance decoding R2: {rnn['distance_decoding_r2']:.3f}")
        print(f"  Confidence dimension: PC{rnn['confidence_dim']['pc']+1} "
              f"(r={rnn['confidence_dim']['r']:.3f})")
        print(f"  Distance dimension: PC{rnn['distance_dim']['pc']+1} "
              f"(r={rnn['distance_dim']['r']:.3f})")

    # CNN features
    if "cnn_features" in report and "error" not in report["cnn_features"]:
        cnn = report["cnn_features"]
        print("\n--- CNN Feature Analysis ---")
        print(f"  Distance decoding R2: {cnn['distance_decoding_r2']:.3f}")
        print(f"  RSA correlation: {cnn['rsa_correlation']:.3f} (p={cnn['rsa_p_value']:.4f})")

    # Latent intentions
    if "latent_intentions" in report and "error" not in report["latent_intentions"]:
        lat = report["latent_intentions"]
        print("\n--- Latent Intention Analysis ---")
        print(f"  Distance prediction R2: {lat['distance_prediction_r2']:.3f}")

    print("\n" + "=" * 70)
