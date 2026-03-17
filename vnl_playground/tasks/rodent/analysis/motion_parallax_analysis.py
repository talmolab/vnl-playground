"""Motion parallax analysis for virtual rodent gap-crossing behavior.

Analyzes head kinematics during gap approach to detect motion parallax
and other monocular depth cues, following Parker et al. (eLife 2022).
"""

import numpy as np
from scipy.signal import find_peaks


def count_vertical_movements(
    z_positions: np.ndarray,
    min_amplitude: float = 0.001,
    dt: float = 0.01,
) -> int:
    """Count vertical head movements (bobs) that exceed a minimum amplitude.

    Computes velocity from z_positions, finds zero crossings, and counts
    segments between consecutive crossings whose peak-to-trough amplitude
    meets the threshold.

    Args:
        z_positions: 1-D array of vertical head positions over time.
        min_amplitude: Minimum peak-to-trough amplitude to count as a
            movement (meters). Default 0.001.
        dt: Timestep in seconds (default 0.01, i.e. 100 Hz).

    Returns:
        Number of qualifying vertical movements.
    """
    if len(z_positions) < 3:
        return 0

    velocity = np.diff(z_positions) / dt

    # Find zero crossings: indices where sign changes between consecutive samples
    sign_changes = np.diff(np.sign(velocity))
    crossing_indices = np.where(sign_changes != 0)[0]

    if len(crossing_indices) < 2:
        return 0

    count = 0
    for i in range(len(crossing_indices) - 1):
        start = crossing_indices[i]
        end = crossing_indices[i + 1] + 1  # +1 to include the endpoint
        segment = z_positions[start : end + 1]
        amplitude = np.max(segment) - np.min(segment)
        if amplitude >= min_amplitude:
            count += 1

    return count


def compute_movement_amplitude(z_positions: np.ndarray) -> float:
    """Compute the mean amplitude of vertical oscillations.

    Identifies peaks and troughs, pairs consecutive alternating extrema,
    and returns the mean absolute difference between them.

    Args:
        z_positions: 1-D array of vertical head positions.

    Returns:
        Mean amplitude of oscillations, or 0.0 if no extrema found.
    """
    if len(z_positions) < 3:
        return 0.0

    peak_indices, _ = find_peaks(z_positions)
    trough_indices, _ = find_peaks(-z_positions)

    if len(peak_indices) == 0 or len(trough_indices) == 0:
        return 0.0

    # Merge peaks and troughs into a sorted list of (index, type) tuples
    extrema = []
    for idx in peak_indices:
        extrema.append((idx, "peak"))
    for idx in trough_indices:
        extrema.append((idx, "trough"))
    extrema.sort(key=lambda x: x[0])

    # Pair consecutive alternating extrema and compute semi-amplitudes
    amplitudes = []
    for i in range(len(extrema) - 1):
        idx_a, type_a = extrema[i]
        idx_b, type_b = extrema[i + 1]
        if type_a != type_b:
            amplitudes.append(abs(z_positions[idx_a] - z_positions[idx_b]) / 2.0)

    if len(amplitudes) == 0:
        return 0.0

    return float(np.mean(amplitudes))


def compute_head_pitch_stats(pitch_degrees: np.ndarray) -> dict:
    """Compute summary statistics for head pitch angle.

    Args:
        pitch_degrees: 1-D array of head pitch angles in degrees.

    Returns:
        Dict with keys: mean_pitch, std_pitch, min_pitch, max_pitch,
        range_pitch.
    """
    pitch_degrees = np.asarray(pitch_degrees, dtype=float)
    return {
        "mean_pitch": float(np.mean(pitch_degrees)),
        "std_pitch": float(np.std(pitch_degrees)),
        "min_pitch": float(np.min(pitch_degrees)),
        "max_pitch": float(np.max(pitch_degrees)),
        "range_pitch": float(np.max(pitch_degrees) - np.min(pitch_degrees)),
    }


def compute_total_head_distance(positions: np.ndarray) -> float:
    """Compute total distance traveled by the head (skull).

    Args:
        positions: Array of shape (T, 3) with skull positions over time.

    Returns:
        Total Euclidean distance traveled.
    """
    if len(positions) < 2:
        return 0.0

    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


def compute_approach_duration(
    torso_velocities: np.ndarray,
    dt: float = 0.01,
) -> float:
    """Compute the duration of the approach phase.

    Args:
        torso_velocities: Array of torso velocity samples during approach.
        dt: Timestep in seconds (default 0.01, i.e. 100 Hz).

    Returns:
        Duration in seconds.
    """
    return len(torso_velocities) * dt
