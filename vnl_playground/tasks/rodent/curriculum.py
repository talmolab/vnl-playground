"""Auto-curriculum trainer for GapJumpTrial.

Automatically graduates between curriculum phases when the agent's trial
success rate exceeds a threshold.  Each phase rebuilds the environment with
harder gap distances and calls ``ppo.train()`` (or ``ff_ppo.train()``) from
scratch, loading the previous phase's policy params as initialisation.

The ``progress_fn`` callback from Brax PPO receives
``eval/episode_trial/success`` at each eval epoch.  When this metric exceeds
``graduation_threshold`` for ``graduation_patience`` consecutive evaluations
the current training run is stopped and the next phase begins.

Usage::

    trainer = CurriculumTrainer(cfg)
    trainer.train()

Or from the CLI::

    python -m vnl_playground.train_highlvl \\
        --config-name=gap_jump_trial_vision/curriculum_auto
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurriculumPhase:
    """Configuration for one curriculum phase."""

    name: str
    gap_distances: tuple[float, ...]
    hold_duration: int = 0
    episode_length: int = 500
    max_decision_steps: int = 300
    # Reward overrides (merged on top of base sparse config)
    extra_reward_terms: dict[str, dict[str, float]] = field(default_factory=dict)
    # Termination overrides
    termination_overrides: dict[str, dict] = field(default_factory=dict)
    # Training hyperparameters
    learning_rate: float = 3e-4
    unroll_length: int = 10
    num_timesteps: int = 100_000_000
    # Graduation criteria
    graduation_threshold: float = 0.8
    graduation_patience: int = 3


class GraduationMonitor:
    """Monitors success rate from progress_fn and signals graduation.

    Thread-safe: ``progress_fn`` may be called from a Brax training thread
    while the main thread polls ``should_graduate``.
    """

    def __init__(self, threshold: float, patience: int):
        self.threshold = threshold
        self.patience = patience
        self._consecutive_above = 0
        self._graduated = False
        self._lock = threading.Lock()
        self._latest_success_rate = 0.0

    def update(self, metrics: dict[str, Any]) -> None:
        """Called inside progress_fn with eval metrics."""
        success_rate = None
        for key in ("eval/episode_trial/success", "eval/episode_episode_trial/success"):
            if key in metrics:
                val = metrics[key]
                success_rate = float(val) if hasattr(val, "dtype") else val
                break

        if success_rate is None:
            return

        with self._lock:
            self._latest_success_rate = success_rate
            if success_rate >= self.threshold:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0

            if self._consecutive_above >= self.patience:
                self._graduated = True

    @property
    def should_graduate(self) -> bool:
        with self._lock:
            return self._graduated

    @property
    def latest_success_rate(self) -> float:
        with self._lock:
            return self._latest_success_rate

    def reset(self) -> None:
        with self._lock:
            self._consecutive_above = 0
            self._graduated = False
            self._latest_success_rate = 0.0


def build_phases_from_config(curriculum_cfg: dict) -> list[CurriculumPhase]:
    """Build CurriculumPhase list from a Hydra/OmegaConf curriculum dict.

    Expected structure::

        curriculum:
          graduation_threshold: 0.8
          graduation_patience: 3
          phases:
            - name: "Phase 1: Stepping"
              gap_distances: [0.0, 0.005, ...]
              hold_duration: 0
              ...
    """
    default_threshold = curriculum_cfg.get("graduation_threshold", 0.8)
    default_patience = curriculum_cfg.get("graduation_patience", 3)
    phases = []

    for i, phase_dict in enumerate(curriculum_cfg["phases"]):
        phase = CurriculumPhase(
            name=phase_dict.get("name", f"Phase {i + 1}"),
            gap_distances=tuple(phase_dict["gap_distances"]),
            hold_duration=phase_dict.get("hold_duration", 0),
            episode_length=phase_dict.get("episode_length", 500),
            max_decision_steps=phase_dict.get("max_decision_steps", 300),
            extra_reward_terms=phase_dict.get("extra_reward_terms", {}),
            termination_overrides=phase_dict.get("termination_overrides", {}),
            learning_rate=phase_dict.get("learning_rate", 3e-4),
            unroll_length=phase_dict.get("unroll_length", 10),
            num_timesteps=phase_dict.get("num_timesteps", 100_000_000),
            graduation_threshold=phase_dict.get(
                "graduation_threshold", default_threshold
            ),
            graduation_patience=phase_dict.get("graduation_patience", default_patience),
        )
        phases.append(phase)

    return phases


def apply_phase_to_env_config(base_env_args: dict, phase: CurriculumPhase) -> dict:
    """Produce env_args dict with phase-specific overrides applied.

    Merges the phase's gap_distances, hold_duration, episode_length,
    and any extra_reward_terms on top of the base config.
    """
    env_args = dict(base_env_args)
    env_args["gap_distances"] = list(phase.gap_distances)
    env_args["hold_duration"] = phase.hold_duration
    env_args["episode_length"] = phase.episode_length
    env_args["max_decision_steps"] = phase.max_decision_steps

    # Merge extra reward terms
    if phase.extra_reward_terms:
        reward_terms = dict(env_args.get("reward_terms", {}))
        for name, params in phase.extra_reward_terms.items():
            reward_terms[name] = dict(params)
        env_args["reward_terms"] = reward_terms

    # Merge termination overrides
    if phase.termination_overrides:
        term_criteria = dict(env_args.get("termination_criteria", {}))
        for name, params in phase.termination_overrides.items():
            term_criteria[name] = dict(params)
        env_args["termination_criteria"] = term_criteria

    # Update trial_timeout to match episode_length
    if "termination_criteria" in env_args:
        term = env_args["termination_criteria"]
        if "trial_timeout" in term:
            term["trial_timeout"]["max_steps"] = phase.episode_length

    return env_args


def apply_phase_to_train_config(
    base_train_config: dict, phase: CurriculumPhase
) -> dict:
    """Produce train_config dict with phase-specific hyperparameter overrides."""
    train_config = dict(base_train_config)
    train_config["learning_rate"] = phase.learning_rate
    train_config["unroll_length"] = phase.unroll_length
    train_config["num_timesteps"] = phase.num_timesteps
    train_config["episode_length"] = phase.episode_length
    return train_config


def make_curriculum_progress_fn(
    monitor: GraduationMonitor,
    base_progress_fn,
    phase_idx: int,
    phase_name: str,
):
    """Wrap the base wandb progress_fn to also check graduation criteria.

    Adds ``curriculum/phase``, ``curriculum/phase_name``, and
    ``curriculum/success_rate`` to the logged metrics.
    """

    def progress_fn(num_steps, metrics):
        # Add curriculum tracking metrics
        metrics["curriculum/phase"] = phase_idx
        metrics["curriculum/phase_name"] = phase_name
        monitor.update(metrics)
        metrics["curriculum/success_rate"] = monitor.latest_success_rate
        metrics["curriculum/graduated"] = float(monitor.should_graduate)

        # Call the base progress_fn (wandb logging)
        base_progress_fn(num_steps, metrics)

        if monitor.should_graduate:
            logging.info(
                f"CURRICULUM: Phase {phase_idx} ({phase_name}) graduated! "
                f"Success rate: {monitor.latest_success_rate:.3f} >= "
                f"{monitor.threshold} for {monitor.patience} consecutive evals."
            )

    return progress_fn
