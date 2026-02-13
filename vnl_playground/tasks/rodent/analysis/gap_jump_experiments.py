"""Experimental manipulation suite for the gap-jump task.

Implements the experimental conditions from Liska et al. for post-training
analysis of the virtual rodent gap-jumping behavior:

- Binocular (baseline): Full egocentric vision
- Monocular: Left or right half of visual field masked to zeros
- V1 suppression: CNN feature maps zeroed or noise-injected
- Combined: Monocular + V1 suppression

Each experiment runs N trials per gap distance and collects:
- Trial outcome (success/failure/abort)
- Decision time (steps from DECISION phase to JUMP)
- Head kinematics during DECISION phase
- CNN features, GRU hidden states, and latent intentions per timestep
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================
# Vision manipulation functions
# ============================================================

def apply_monocular_mask(vision: jnp.ndarray, side: str = "left") -> jnp.ndarray:
    """Mask half of the visual field to simulate monocular eyelid suture.

    Args:
        vision: Image array [..., H, W, C].
        side: Which side to occlude ("left" or "right").

    Returns:
        Vision with one half zeroed out.
    """
    W = vision.shape[-2]
    half = W // 2
    if side == "left":
        return vision.at[..., :half, :].set(0.0)
    else:
        return vision.at[..., half:, :].set(0.0)


def apply_v1_suppression(
    cnn_features: jnp.ndarray,
    suppression_fraction: float = 1.0,
    rng: Optional[jax.Array] = None,
    noise_std: float = 0.1,
) -> jnp.ndarray:
    """Ablate CNN features by zeroing or adding noise.

    Analogous to optogenetic V1 suppression in the paper.

    Args:
        cnn_features: CNN output features [..., feature_dim].
        suppression_fraction: 0.0 = no suppression, 1.0 = complete.
        rng: Random key for noise injection (if None, uses zero masking).
        noise_std: Standard deviation of noise to inject.

    Returns:
        Modified CNN features.
    """
    if rng is not None:
        noise = jax.random.normal(rng, cnn_features.shape) * noise_std
        return cnn_features * (1.0 - suppression_fraction) + noise * suppression_fraction
    return cnn_features * (1.0 - suppression_fraction)


# ============================================================
# Trial data collection
# ============================================================

@dataclass
class TrialData:
    """Data collected from a single trial.

    Attributes:
        gap_distance: The gap distance for this trial (meters).
        outcome: Trial result ("success", "failure", "abort", "timeout").
        decision_time_steps: Steps from DECISION start to JUMP initiation.
        decision_time_seconds: Decision time in seconds.
        condition: Experimental condition name.
        total_steps: Total episode steps.

        # Per-timestep data during DECISION phase
        head_positions: [T, 3] skull body xpos during DECISION.
        head_orientations: [T, 9] skull body xmat (flattened 3x3) during DECISION.
        cnn_features: [T, cnn_dim] CNN encoder outputs during DECISION.
        gru_hidden_states: [T, gru_dim] GRU hidden states during DECISION.
        latent_z: [T, latent_dim] latent intentions during DECISION.
        torso_velocities: [T, 3] torso velocity during DECISION.
    """
    gap_distance: float = 0.0
    outcome: str = "unknown"
    decision_time_steps: int = 0
    decision_time_seconds: float = 0.0
    condition: str = "binocular"
    total_steps: int = 0

    head_positions: Optional[np.ndarray] = None
    head_orientations: Optional[np.ndarray] = None
    cnn_features: Optional[np.ndarray] = None
    gru_hidden_states: Optional[np.ndarray] = None
    latent_z: Optional[np.ndarray] = None
    torso_velocities: Optional[np.ndarray] = None


@dataclass
class ExperimentConfig:
    """Configuration for an experimental block.

    Attributes:
        condition: Name of the experimental condition.
        gap_distances: List of gap distances to test.
        n_trials_per_distance: Number of trials per gap distance.
        monocular_side: Which eye to occlude (None, "left", or "right").
        v1_suppression_fraction: CNN ablation level (0.0 to 1.0).
        v1_noise_std: Noise std for V1 suppression (0 = zero masking).
        ctrl_dt: Control timestep for converting steps to seconds.
    """
    condition: str = "binocular"
    gap_distances: tuple = (0.06, 0.08, 0.10, 0.12, 0.14)
    n_trials_per_distance: int = 50
    monocular_side: Optional[str] = None
    v1_suppression_fraction: float = 0.0
    v1_noise_std: float = 0.0
    ctrl_dt: float = 0.02


# Predefined experimental conditions
BINOCULAR = ExperimentConfig(condition="binocular")
MONOCULAR_LEFT = ExperimentConfig(condition="monocular_left", monocular_side="left")
MONOCULAR_RIGHT = ExperimentConfig(condition="monocular_right", monocular_side="right")
V1_SUPPRESSION = ExperimentConfig(condition="v1_suppression", v1_suppression_fraction=1.0)
V1_SUPPRESSION_PARTIAL = ExperimentConfig(
    condition="v1_suppression_50", v1_suppression_fraction=0.5,
)
MONOCULAR_LEFT_V1 = ExperimentConfig(
    condition="monocular_left_v1",
    monocular_side="left",
    v1_suppression_fraction=1.0,
)

ALL_CONDITIONS = [
    BINOCULAR, MONOCULAR_LEFT, MONOCULAR_RIGHT,
    V1_SUPPRESSION, V1_SUPPRESSION_PARTIAL,
    MONOCULAR_LEFT_V1,
]


# ============================================================
# Experiment runner
# ============================================================

def run_single_trial(
    env,
    policy_fn: Callable,
    rng: jax.Array,
    gap_distance: float,
    config: ExperimentConfig,
    max_steps: int = 500,
    record_neural: bool = True,
) -> TrialData:
    """Run a single trial under the specified experimental condition.

    Args:
        env: The GapJumpTrial environment (unwrapped).
        policy_fn: Policy function (obs, carry, rng) -> (action, new_carry, aux).
        rng: Random key.
        gap_distance: Desired gap distance for this trial.
        config: Experimental condition configuration.
        max_steps: Maximum episode steps.
        record_neural: Whether to record CNN/GRU/latent data.

    Returns:
        TrialData with trial outcome and neural recordings.
    """
    # Reset environment
    rng, reset_rng, policy_rng = jax.random.split(rng, 3)
    state = env.reset(reset_rng)

    # Recording buffers
    head_pos_list = []
    head_ori_list = []
    cnn_feat_list = []
    gru_hidden_list = []
    latent_list = []
    velocity_list = []

    # Initialize GRU carry
    carry = jnp.zeros(policy_fn.gru_hidden_size) if hasattr(policy_fn, 'gru_hidden_size') else None

    trial_phase = 0  # HOLD
    decision_start = -1
    jump_step = -1
    outcome = "timeout"

    for step_idx in range(max_steps):
        # Check if done
        if state.done > 0.5:
            if state.info.get("trial_success", False):
                outcome = "success"
            else:
                outcome = "failure"
            break

        # Get vision and apply manipulations
        obs = state.obs
        if config.monocular_side is not None:
            # Apply monocular mask to vision if present
            if "vision" in obs.get("state", {}):
                vision = obs["state"]["vision"]
                masked_vision = apply_monocular_mask(vision, config.monocular_side)
                obs["state"]["vision"] = masked_vision

        # Run policy
        rng, step_rng = jax.random.split(rng)
        if carry is not None:
            action, carry, aux = policy_fn(obs, carry, step_rng)
        else:
            action, aux = policy_fn(obs, step_rng)

        # Apply V1 suppression to CNN features if needed
        if config.v1_suppression_fraction > 0 and aux is not None and "cnn_features" in aux:
            rng, v1_rng = jax.random.split(rng)
            v1_rng_use = v1_rng if config.v1_noise_std > 0 else None
            aux["cnn_features"] = apply_v1_suppression(
                aux["cnn_features"],
                config.v1_suppression_fraction,
                v1_rng_use,
                config.v1_noise_std,
            )

        # Record data during DECISION phase
        current_phase = int(state.info.get("trial_phase", 0))
        if current_phase == 1 and record_neural:  # DECISION
            if decision_start < 0:
                decision_start = step_idx

            # Record head kinematics (skull body)
            # These would be extracted from state.data
            head_pos_list.append(np.zeros(3))  # placeholder
            head_ori_list.append(np.zeros(9))  # placeholder
            velocity_list.append(np.zeros(3))  # placeholder

            if aux is not None:
                if "cnn_features" in aux:
                    cnn_feat_list.append(np.array(aux["cnn_features"]))
                if "gru_hidden" in aux:
                    gru_hidden_list.append(np.array(aux["gru_hidden"]))
                if "latent_z" in aux:
                    latent_list.append(np.array(aux["latent_z"]))

        if current_phase == 2 and jump_step < 0:  # JUMP
            jump_step = step_idx

        # Step environment
        state = env.step(state, action)

    # Compute decision time
    decision_steps = max(0, jump_step - decision_start) if (decision_start >= 0 and jump_step >= 0) else 0

    trial = TrialData(
        gap_distance=float(gap_distance),
        outcome=outcome,
        decision_time_steps=decision_steps,
        decision_time_seconds=decision_steps * config.ctrl_dt,
        condition=config.condition,
        total_steps=step_idx + 1,
        head_positions=np.array(head_pos_list) if head_pos_list else None,
        head_orientations=np.array(head_ori_list) if head_ori_list else None,
        cnn_features=np.array(cnn_feat_list) if cnn_feat_list else None,
        gru_hidden_states=np.array(gru_hidden_list) if gru_hidden_list else None,
        latent_z=np.array(latent_list) if latent_list else None,
        torso_velocities=np.array(velocity_list) if velocity_list else None,
    )
    return trial


def run_experiment(
    env,
    policy_fn: Callable,
    config: ExperimentConfig,
    base_rng: jax.Array = None,
    max_steps: int = 500,
    record_neural: bool = True,
    verbose: bool = True,
) -> list[TrialData]:
    """Run a full experimental block across all gap distances.

    Args:
        env: The GapJumpTrial environment.
        policy_fn: Policy function.
        config: Experimental condition configuration.
        base_rng: Base random key.
        max_steps: Maximum steps per trial.
        record_neural: Whether to record neural data.
        verbose: Print progress.

    Returns:
        List of TrialData for all trials.
    """
    if base_rng is None:
        base_rng = jax.random.PRNGKey(0)

    all_trials = []
    trial_count = 0

    for gap_dist in config.gap_distances:
        for trial_idx in range(config.n_trials_per_distance):
            rng = jax.random.fold_in(base_rng, trial_count)
            trial_data = run_single_trial(
                env, policy_fn, rng, gap_dist, config,
                max_steps=max_steps, record_neural=record_neural,
            )
            all_trials.append(trial_data)
            trial_count += 1

            if verbose and trial_count % 10 == 0:
                print(f"  [{config.condition}] Trial {trial_count}: "
                      f"gap={gap_dist:.2f}m, outcome={trial_data.outcome}")

    if verbose:
        n_success = sum(1 for t in all_trials if t.outcome == "success")
        print(f"  [{config.condition}] Complete: {n_success}/{len(all_trials)} success "
              f"({100*n_success/len(all_trials):.1f}%)")

    return all_trials


def run_all_conditions(
    env,
    policy_fn: Callable,
    conditions: list[ExperimentConfig] = None,
    base_rng: jax.Array = None,
    max_steps: int = 500,
    record_neural: bool = True,
    verbose: bool = True,
) -> dict[str, list[TrialData]]:
    """Run all experimental conditions.

    Args:
        env: The GapJumpTrial environment.
        policy_fn: Policy function.
        conditions: List of conditions (defaults to ALL_CONDITIONS).
        base_rng: Base random key.
        max_steps: Maximum steps per trial.
        record_neural: Whether to record neural data.
        verbose: Print progress.

    Returns:
        Dict mapping condition name to list of TrialData.
    """
    if conditions is None:
        conditions = ALL_CONDITIONS
    if base_rng is None:
        base_rng = jax.random.PRNGKey(42)

    results = {}
    for i, config in enumerate(conditions):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running condition: {config.condition}")
            print(f"{'='*60}")

        condition_rng = jax.random.fold_in(base_rng, i)
        trials = run_experiment(
            env, policy_fn, config,
            base_rng=condition_rng,
            max_steps=max_steps,
            record_neural=record_neural,
            verbose=verbose,
        )
        results[config.condition] = trials

    return results


# ============================================================
# Utility functions for working with trial data
# ============================================================

def compute_success_rate(trials: list[TrialData]) -> dict[float, float]:
    """Compute success rate per gap distance."""
    from collections import defaultdict
    counts = defaultdict(lambda: {"success": 0, "total": 0})
    for t in trials:
        counts[t.gap_distance]["total"] += 1
        if t.outcome == "success":
            counts[t.gap_distance]["success"] += 1
    return {
        dist: c["success"] / max(c["total"], 1)
        for dist, c in sorted(counts.items())
    }


def compute_mean_decision_time(trials: list[TrialData]) -> dict[float, float]:
    """Compute mean decision time per gap distance (successful trials only)."""
    from collections import defaultdict
    times = defaultdict(list)
    for t in trials:
        if t.outcome == "success" and t.decision_time_steps > 0:
            times[t.gap_distance].append(t.decision_time_seconds)
    return {
        dist: float(np.mean(ts)) if ts else 0.0
        for dist, ts in sorted(times.items())
    }


def extract_gru_hidden_states(trials: list[TrialData]) -> dict[float, list[np.ndarray]]:
    """Extract GRU hidden states grouped by gap distance."""
    from collections import defaultdict
    hidden_states = defaultdict(list)
    for t in trials:
        if t.gru_hidden_states is not None and t.outcome == "success":
            hidden_states[t.gap_distance].append(t.gru_hidden_states)
    return dict(hidden_states)


def save_experiment_results(results: dict[str, list[TrialData]], path: str):
    """Save experiment results to npz file."""
    save_dict = {}
    for condition, trials in results.items():
        save_dict[f"{condition}/gap_distances"] = np.array([t.gap_distance for t in trials])
        save_dict[f"{condition}/outcomes"] = np.array([t.outcome for t in trials])
        save_dict[f"{condition}/decision_times"] = np.array([t.decision_time_seconds for t in trials])
    np.savez(path, **save_dict)
    print(f"Saved results to {path}")
