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

SEARCH_SPACE_DISTRIBUTIONS = {
    "fs": FloatDistribution(0.1, 1.5),
    "damp": FloatDistribution(1e-8, 1e-6, log=True),
    "cc": FloatDistribution(0.0, 0.2),
    "cdc": FloatDistribution(0.0, 0.2),
    "qvel_init": CategoricalDistribution(["zeros", "reference"]),
}

_MS_REF_SUFFIX = "reference_data_moving_shoulder"


def _row_is_moving_shoulder_and_standard(row: pd.Series) -> bool:
    tags = str(row.get("Tags", ""))
    if "moving-shoulder" not in tags:
        return False
    if "tau-extra" in tags:
        return False
    if row.get("State") != "finished":
        return False
    ref = str(row.get("_fields.reference_data_path", ""))
    if not ref.endswith(_MS_REF_SUFFIX):
        return False
    return True


def load_warmstart(csv_path: Path) -> list[FrozenTrial]:
    df = pd.read_csv(csv_path)
    trials: list[FrozenTrial] = []
    for i, row in df.iterrows():
        if not _row_is_moving_shoulder_and_standard(row):
            continue
        try:
            fs = float(row["_fields.force_scale"])
            damp = float(row["_fields.joint_damping"])
            cc = float(row["_fields.control_cost"])
            cdc = float(row["_fields.control_diff_cost"])
            qvel = str(row["_fields.qvel_init"])
            R = float(row["eval/episode_reward"])
            bcorr = float(row["eval/emg_biceps_corr"])
            tcorr = float(row["eval/emg_triceps_corr"])
            bmae = float(row["eval/emg_biceps_mae"])
            tmae = float(row["eval/emg_triceps_mae"])
            btrial = float(row["eval/emg_biceps_trial_mae"])
            ttrial = float(row["eval/emg_triceps_trial_mae"])
        except (ValueError, TypeError):
            continue
        if any(math.isnan(v) for v in (fs, damp, cc, cdc, R, bcorr, tcorr, bmae, tmae, btrial, ttrial)):
            continue
        if qvel not in ("zeros", "reference"):
            continue
        params = {"fs": fs, "damp": damp, "cc": cc, "cdc": cdc, "qvel_init": qvel}
        trials.append(
            optuna.trial.create_trial(
                params=params,
                distributions=SEARCH_SPACE_DISTRIBUTIONS,
                values=[bcorr, tcorr, bmae, tmae, btrial, ttrial],
                user_attrs={
                    "R": R,
                    "constraint": [CONSTRAINT_REWARD_FLOOR - R],
                    "source_name": str(row.get("Name", f"row-{i}")),
                },
            )
        )
    return trials


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
