"""Bayesian optimization driver for moving-shoulder training.

See docs/superpowers/specs/2026-04-20-bo-shape-constrained-sweep-design.md.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import optuna
import pandas as pd
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
)
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.samplers import NSGAIISampler
from optuna.trial import FrozenTrial, TrialState


OBJECTIVE_DIRECTIONS = ["maximize", "maximize", "minimize", "minimize", "minimize", "minimize"]
OBJECTIVE_KEYS = (
    "eval/emg_biceps_corr",
    "eval/emg_triceps_corr",
    "eval/emg_biceps_mae",
    "eval/emg_triceps_mae",
    "eval/emg_biceps_trial_mae",
    "eval/emg_triceps_trial_mae",
)
REWARD_KEY = "eval/episode_reward"
CONSTRAINT_REWARD_FLOOR = 380.0
WINNER_REWARD_FLOOR = 400.0
WANDB_PROJECT = "vnl-mjx-rl"
TRAINING_SCRIPT = "train_mouse_janelia_sigmoid_moving_shoulder.py"


def load_warmstart(csv_path: Path) -> list[FrozenTrial]:
    raise NotImplementedError


def make_study(study_name: str, journal_path: Path) -> optuna.Study:
    raise NotImplementedError


def launch_training(params: dict, tag: str, log_dir: Path) -> int:
    raise NotImplementedError


def read_metrics(tag: str, retries: int = 3, backoff_s: float = 30.0) -> Optional[dict]:
    raise NotImplementedError


def report_frontier(study: optuna.Study, out_csv: Path) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
