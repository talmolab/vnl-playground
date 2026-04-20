"""Bayesian optimization driver for moving-shoulder training.

See docs/superpowers/specs/2026-04-20-bo-shape-constrained-sweep-design.md.
"""
from __future__ import annotations

import argparse
import csv as _csv
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


def _constraints_func(trial: FrozenTrial) -> list[float]:
    c = trial.user_attrs.get("constraint")
    if c is None:
        # Unknown feasibility -> treat as feasible; NSGA-II will still use objectives.
        return [-1.0]
    return list(c)


def make_study(study_name: str, journal_path: Path) -> optuna.Study:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    sampler = NSGAIISampler(constraints_func=_constraints_func)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=OBJECTIVE_DIRECTIONS,
        sampler=sampler,
        load_if_exists=True,
    )


FIXED_CLI_ARGS = (
    "--seed", "1",
    "--num-timesteps", "800000000",
    "--num-evals", "8",
    "--episode-length", "100",
    "--joint-armature", "4e-10",
    "--ctrl-dt", "0.0025",
    "--sim-dt", "0.00125",
    "--joints-weight", "5.0",
    "--joints-vel-weight", "0.5",
    "--wrist-pos-weight", "0.1",
    "--bodies-pos-weight", "0.1",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fmt_float(x: float) -> str:
    # Use a consistent format so tests can assert exact strings.
    if abs(x) < 1e-3 and x != 0:
        return f"{x:.0e}"
    return str(x)


def launch_training(params: dict, tag: str, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"
    cmd = [
        sys.executable,
        TRAINING_SCRIPT,
        "--force-scale", _fmt_float(params["fs"]),
        "--joint-damping", _fmt_float(params["damp"]),
        "--control-cost", _fmt_float(params["cc"]),
        "--control-diff-cost", _fmt_float(params["cdc"]),
        "--qvel-init", str(params["qvel_init"]),
        *FIXED_CLI_ARGS,
        "--wandb-tags", "bo-s13", tag,
    ]
    with open(log_path, "w") as fp:
        result = subprocess.run(
            cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    return result.returncode


def _wandb_api():
    import wandb  # imported lazily so tests can patch without wandb login
    return wandb.Api()


def read_metrics(tag: str, retries: int = 3, backoff_s: float = 30.0) -> Optional[dict]:
    for attempt in range(retries):
        api = _wandb_api()
        runs = list(api.runs(
            path=WANDB_PROJECT,
            filters={"tags": {"$in": [tag]}},
        ))
        if runs:
            summary = runs[0].summary
            try:
                R = float(summary[REWARD_KEY])
                bcorr = float(summary["eval/emg_biceps_corr"])
                tcorr = float(summary["eval/emg_triceps_corr"])
                bmae = float(summary["eval/emg_biceps_mae"])
                tmae = float(summary["eval/emg_triceps_mae"])
                btrial = float(summary["eval/emg_biceps_trial_mae"])
                ttrial = float(summary["eval/emg_triceps_trial_mae"])
            except (KeyError, TypeError, ValueError):
                R = bcorr = tcorr = bmae = tmae = btrial = ttrial = float("nan")
            if any(math.isnan(v) for v in (R, bcorr, tcorr, bmae, tmae, btrial, ttrial)):
                return None
            return {
                "R": R,
                "bcorr": bcorr,
                "tcorr": tcorr,
                "bmae": bmae,
                "tmae": tmae,
                "btrial": btrial,
                "ttrial": ttrial,
            }
        if attempt < retries - 1:
            time.sleep(backoff_s)
    return None


def report_frontier(study: optuna.Study, out_csv: Path) -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    winners = []
    for t in completed:
        R = t.user_attrs.get("R")
        if R is None or R < WINNER_REWARD_FLOOR:
            continue
        # Require values to be present (completed multi-objective trials have values).
        if t.values is None:
            continue
        bcorr, tcorr, bmae, tmae, btrial, ttrial = t.values
        winners.append({
            "trial": t.number,
            "R": R,
            "bcorr": bcorr,
            "tcorr": tcorr,
            "bmae": bmae,
            "tmae": tmae,
            "btrial": btrial,
            "ttrial": ttrial,
            "fs": t.params.get("fs"),
            "damp": t.params.get("damp"),
            "cc": t.params.get("cc"),
            "cdc": t.params.get("cdc"),
            "qvel_init": t.params.get("qvel_init"),
            "source_name": t.user_attrs.get("source_name", ""),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial", "R", "bcorr", "tcorr", "bmae", "tmae", "btrial", "ttrial",
                  "fs", "damp", "cc", "cdc", "qvel_init", "source_name"]
    with open(out_csv, "w", newline="") as fp:
        w = _csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for row in winners:
            w.writerow(row)

    print(f"Wrote {len(winners)} winner(s) to {out_csv}")
    for row in sorted(winners, key=lambda r: -r["bcorr"])[:5]:
        print(f"  trial={row['trial']} R={row['R']:.0f} "
              f"bcorr={row['bcorr']:.2f} tcorr={row['tcorr']:.2f} "
              f"bmae={row['bmae']:.2f} tmae={row['tmae']:.2f} "
              f"btrial={row['btrial']:.2f} ttrial={row['ttrial']:.2f} "
              f"fs={row['fs']:.2f} damp={row['damp']:.1e} cc={row['cc']:.3f} cdc={row['cdc']:.3f}")


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
