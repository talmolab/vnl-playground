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
            cc = float(row["reward_weights/control_cost"])
            cdc = float(row["reward_weights/control_diff_cost"])
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
    import os
    env = {**os.environ, "PYOPENGL_PLATFORM": "egl"}
    with open(log_path, "w") as fp:
        result = subprocess.run(
            cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
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


CONSECUTIVE_FAIL_LIMIT = 5


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fp:
        fp.write(json.dumps(row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warmstart-csv", type=Path, required=True)
    p.add_argument("--study-name", type=str, required=True)
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--jsonl-log", type=Path, required=True)
    p.add_argument("--frontier-csv", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    args = p.parse_args()

    study = make_study(args.study_name, args.journal)

    # Warm-start only if the study is empty (avoid re-adding on resume).
    if len(study.trials) == 0:
        warmstarts = load_warmstart(args.warmstart_csv)
        print(f"Adding {len(warmstarts)} warm-start trials...")
        study.add_trials(warmstarts)
    else:
        print(f"Resuming study with {len(study.trials)} existing trials; skipping warm-start.")

    args.log_dir.mkdir(parents=True, exist_ok=True)

    consecutive_fail = 0
    for i in range(args.n_trials):
        trial = study.ask(SEARCH_SPACE_DISTRIBUTIONS)
        tag = f"trial-{trial.number:04d}"
        # NSGA-II crossover can extrapolate past distribution bounds with
        # out-of-bound warm-starts in the pool; clamp to declared ranges.
        params = {
            "fs": min(max(trial.params["fs"], 0.5), 1.0),
            "damp": min(max(trial.params["damp"], 1e-7), 1.5e-6),
            "cc": min(max(trial.params["cc"], 0.0), 0.1),
            "cdc": min(max(trial.params["cdc"], 0.0), 0.1),
            "qvel_init": trial.params["qvel_init"],
        }
        print(f"[{i+1}/{args.n_trials}] {tag}  params={params}")

        rc = launch_training(params, tag, args.log_dir)
        metrics = read_metrics(tag) if rc == 0 else None

        if metrics is None:
            print(f"  {tag} FAILED (rc={rc} or no metrics)")
            study.tell(trial, state=TrialState.FAIL)
            consecutive_fail += 1
            if consecutive_fail >= CONSECUTIVE_FAIL_LIMIT:
                print(f"ABORT: {CONSECUTIVE_FAIL_LIMIT} consecutive failures", file=sys.stderr)
                _append_jsonl(args.jsonl_log, {"event": "abort",
                                               "reason": "consecutive_fail_limit",
                                               "trial": trial.number})
                sys.exit(2)
            continue

        consecutive_fail = 0
        R = metrics["R"]
        trial.set_user_attr("constraint", [CONSTRAINT_REWARD_FLOOR - R])
        trial.set_user_attr("R", R)
        study.tell(trial, [metrics["bcorr"], metrics["tcorr"],
                           metrics["bmae"], metrics["tmae"],
                           metrics["btrial"], metrics["ttrial"]])
        _append_jsonl(args.jsonl_log, {
            "trial": trial.number,
            "tag": tag,
            **params,
            **metrics,
        })
        print(f"  {tag} OK  R={R:.0f} bcorr={metrics['bcorr']:.2f} "
              f"tcorr={metrics['tcorr']:.2f}")

    report_frontier(study, args.frontier_csv)


if __name__ == "__main__":
    main()
