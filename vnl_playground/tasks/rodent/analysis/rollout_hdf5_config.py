"""Configuration dataclasses for HDF5 rollout collection."""

from dataclasses import dataclass


@dataclass
class RolloutCollectionConfig:
    """Top-level configuration for rollout data collection.

    Attributes:
        checkpoint_path: Path to trained policy checkpoint directory.
        output_path: Path for the output HDF5 file.
        prior_checkpoint_path: Path to SCAMPER prior checkpoint. If None,
            loaded from checkpoint config.json transfer.prior_checkpoint_path.
        mode: "variable_gap" or "fixed_gap".
        n_episodes: Number of rollout episodes (variable mode).
        n_envs: Parallel environments per batch.
        episode_length: Maximum steps per episode.
        seed: Random seed.
        min_gaps_crossed: Minimum gap crossings to keep episode (variable mode).
        capture_vision: Store raw 32x32 binocular images.
        capture_activations: Store vision_features, intention, decoder layers.
        capture_cnn_maps: Store per-layer spatial CNN feature maps (large).
    """

    checkpoint_path: str
    output_path: str
    prior_checkpoint_path: str | None = None
    mode: str = "variable_gap"
    n_episodes: int = 50
    n_envs: int = 64
    episode_length: int = 1000
    seed: int = 42
    min_gaps_crossed: int = 8
    capture_vision: bool = True
    capture_activations: bool = True
    capture_cnn_maps: bool = False


@dataclass
class FixedGapConfig:
    """Configuration for fixed-gap sweep mode.

    Creates environments with n_gaps_per_env identical gaps at each
    gap length, sweeping from gap_min to gap_max in gap_step increments.

    Attributes:
        gap_min: Minimum gap length in meters.
        gap_max: Maximum gap length in meters.
        gap_step: Sweep step size in meters.
        n_gaps_per_env: Platforms per environment (literally N identical gaps).
        episodes_per_gap: Rollout episodes per gap length.
        min_gaps_crossed: Minimum gaps crossed to keep episode (0 = no filter).
    """

    gap_min: float = 0.03
    gap_max: float = 0.20
    gap_step: float = 0.01
    n_gaps_per_env: int = 5
    episodes_per_gap: int = 20
    min_gaps_crossed: int = 0
