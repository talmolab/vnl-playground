"""Head kinematics extraction and gap-approach window detection.

Provides utilities to compute head pose (pitch, yaw, roll) from MuJoCo
body transforms and to detect temporal windows preceding gap crossings
for downstream analysis of approach behavior.

MuJoCo xmat convention (3x3 rotation matrix, columns = local axes in world):
    col 0 = forward (x-axis)
    col 1 = left    (y-axis)
    col 2 = up      (z-axis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jp
import numpy as np


# ---------------------------------------------------------------------------
# Head pose
# ---------------------------------------------------------------------------


@dataclass
class HeadPose:
    """Instantaneous head orientation and position."""

    position: np.ndarray  # (3,) skull xpos in world frame
    pitch_deg: float  # head pitch (up/down) in degrees; positive = up
    yaw_deg: float  # head yaw (left/right) relative to torso, degrees
    roll_deg: float = 0.0
    eye_left_pos: Optional[np.ndarray] = None
    eye_right_pos: Optional[np.ndarray] = None


def extract_head_pose(
    skull_xpos: np.ndarray,
    skull_xmat: np.ndarray,
    torso_xmat: np.ndarray,
    eye_left_pos: Optional[np.ndarray] = None,
    eye_right_pos: Optional[np.ndarray] = None,
) -> HeadPose:
    """Compute head pitch, yaw, and roll from MuJoCo body transforms.

    Args:
        skull_xpos: (3,) world-frame position of the skull body.
        skull_xmat: (3, 3) rotation matrix for the skull body.
            Columns are [forward, left, up] in world frame.
        torso_xmat: (3, 3) rotation matrix for the torso body.
            Columns are [forward, left, up] in world frame.
        eye_left_pos: Optional (3,) world-frame position of the left eye.
        eye_right_pos: Optional (3,) world-frame position of the right eye.

    Returns:
        A :class:`HeadPose` with pitch, yaw, and roll in degrees.
    """
    # Convert to plain numpy for the return dataclass (inputs may be JAX).
    skull_xpos = np.asarray(skull_xpos, dtype=np.float64)
    skull_xmat = np.asarray(skull_xmat, dtype=np.float64)
    torso_xmat = np.asarray(torso_xmat, dtype=np.float64)

    # --- Forward and up vectors (columns of xmat) ---
    skull_forward = skull_xmat[:, 0]  # local x in world frame
    skull_up = skull_xmat[:, 2]  # local z in world frame
    torso_forward = torso_xmat[:, 0]

    # --- Pitch: angle of skull forward relative to horizontal plane ---
    # Project skull forward onto horizontal (xy) plane and compute elevation.
    horizontal_len = np.sqrt(skull_forward[0] ** 2 + skull_forward[1] ** 2)
    pitch_rad = np.arctan2(skull_forward[2], horizontal_len)
    pitch_deg = float(np.degrees(pitch_rad))

    # --- Yaw: angle of skull forward in horizontal plane relative to torso forward ---
    # Project both forward vectors onto horizontal plane.
    skull_fwd_h = np.array([skull_forward[0], skull_forward[1], 0.0])
    torso_fwd_h = np.array([torso_forward[0], torso_forward[1], 0.0])

    skull_fwd_h_len = np.linalg.norm(skull_fwd_h)
    torso_fwd_h_len = np.linalg.norm(torso_fwd_h)

    if skull_fwd_h_len < 1e-8 or torso_fwd_h_len < 1e-8:
        yaw_deg = 0.0
    else:
        skull_fwd_h = skull_fwd_h / skull_fwd_h_len
        torso_fwd_h = torso_fwd_h / torso_fwd_h_len

        # Dot product gives cosine of unsigned angle.
        cos_yaw = np.clip(np.dot(skull_fwd_h, torso_fwd_h), -1.0, 1.0)
        yaw_unsigned = np.arccos(cos_yaw)

        # Sign via cross product z-component (positive = skull left of torso).
        cross_z = torso_fwd_h[0] * skull_fwd_h[1] - torso_fwd_h[1] * skull_fwd_h[0]
        yaw_rad = float(np.copysign(yaw_unsigned, cross_z))
        yaw_deg = float(np.degrees(yaw_rad))

    # --- Roll: rotation around the forward axis ---
    roll_rad = np.arctan2(skull_up[1], skull_up[2])
    roll_deg = float(np.degrees(roll_rad))

    return HeadPose(
        position=skull_xpos,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        roll_deg=roll_deg,
        eye_left_pos=np.asarray(eye_left_pos) if eye_left_pos is not None else None,
        eye_right_pos=np.asarray(eye_right_pos) if eye_right_pos is not None else None,
    )


# ---------------------------------------------------------------------------
# Approach windows
# ---------------------------------------------------------------------------


@dataclass
class ApproachWindow:
    """A temporal window just before the rodent reaches a gap leading edge."""

    gap_index: int
    gap_leading_edge: float
    gap_length: float
    timestep_indices: np.ndarray  # indices into the trajectory
    crossed_successfully: bool = True


def detect_approach_windows(
    torso_x: np.ndarray,
    gap_leading_edges: np.ndarray,
    gap_lengths: np.ndarray,
    window_steps: int = 25,
) -> list[ApproachWindow]:
    """Detect approach windows preceding each gap crossing.

    For each gap, finds the first timestep at which the torso x-position
    reaches (or exceeds) the gap leading edge, then takes the preceding
    *window_steps* timesteps as the approach window.

    Args:
        torso_x: (T,) array of torso x-positions over time.
        gap_leading_edges: (N,) x-positions of gap leading edges.
        gap_lengths: (N,) lengths of each gap.
        window_steps: Number of timesteps before arrival to include.

    Returns:
        List of :class:`ApproachWindow`, one per gap that the torso reached.
    """
    torso_x = np.asarray(torso_x)
    gap_leading_edges = np.asarray(gap_leading_edges)
    gap_lengths = np.asarray(gap_lengths)

    windows: list[ApproachWindow] = []

    for i, (edge, length) in enumerate(zip(gap_leading_edges, gap_lengths)):
        # First timestep where torso_x >= gap leading edge.
        reached = np.where(torso_x >= edge)[0]
        if len(reached) == 0:
            continue  # never reached this gap

        arrival_idx = int(reached[0])

        # Take window_steps before (and including) arrival.
        start = max(0, arrival_idx - window_steps + 1)
        indices = np.arange(start, arrival_idx + 1)

        # Pad from the left if we don't have enough steps.
        if len(indices) < window_steps:
            pad_len = window_steps - len(indices)
            indices = np.concatenate([np.full(pad_len, indices[0], dtype=int), indices])

        windows.append(
            ApproachWindow(
                gap_index=i,
                gap_leading_edge=float(edge),
                gap_length=float(length),
                timestep_indices=indices,
            )
        )

    return windows


# ---------------------------------------------------------------------------
# Full gap-approach episode (for downstream analysis)
# ---------------------------------------------------------------------------


@dataclass
class GapApproachEpisode:
    """All data for a single gap approach, suitable for batch analysis."""

    gap_index: int
    gap_distance: float  # gap length
    success: bool
    timesteps: np.ndarray  # timestep indices
    head_poses: list[HeadPose]
    torso_positions: np.ndarray  # (T, 3) or (T,) x-positions
    torso_velocities: np.ndarray  # (T,) or (T, 3) velocities
    condition: str = "binocular"
