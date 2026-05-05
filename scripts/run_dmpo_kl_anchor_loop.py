"""Autonomous experiment driver for DMPO + kl-anchor (B-aggressive)."""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "logs" / "dmpo_kl_anchor_experiments.tsv"
LOG_DIR = REPO / "logs" / "dmpo_kl_anchor"
CKPT_ROOT = REPO / "checkpoints"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TSV.parent.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# Per-run wall-time budget (seconds). Default: 1.5h.
RUN_BUDGET_S = int(os.environ.get("DMPO_RUN_BUDGET_S", str(90 * 60)))

# Maximum number of full-loop iterations before exit (safety net).
MAX_ITERS = int(os.environ.get("DMPO_MAX_ITERS", "200"))

CONFIG_NAME = "rodent_run_gap_dmpo/velocity_only_kl_anchor"

log = logging.getLogger("dmpo_kl_anchor_loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class RunSpec:
    name: str
    w_anchor: float
    alpha_anchor: float
    prior_lr_mult: float
    decoder_lr_mult: float
    epsilon_stddev: float
    num_timesteps: int

    def overrides(self):
        return [
            f"++checkpoint_dir=./checkpoints/{self.name}",
            f"++logging_config.exp_name={self.name}",
            f"++kl_anchor.w_anchor={self.w_anchor}",
            f"++kl_anchor.alpha_anchor={self.alpha_anchor}",
            f"++kl_anchor.prior_lr_mult={self.prior_lr_mult}",
            f"++kl_anchor.decoder_lr_mult={self.decoder_lr_mult}",
            f"++train_config.epsilon_stddev={self.epsilon_stddev}",
            f"++train_config.num_timesteps={self.num_timesteps}",
        ]


# alpha capped so per-step bonus * (1/(1-gamma)) stays under critic vmax=150.
SEED_SPECS = [
    RunSpec("kl_a_v1_baseline",    0.5,  3.0,  0.1, 1.0, 1e-7, 600_000_000),
    RunSpec("kl_a_v2_low_anchor",  0.1,  1.0,  0.1, 1.0, 1e-7, 600_000_000),
    RunSpec("kl_a_v3_strong",      1.0,  4.0,  0.1, 1.0, 1e-7, 600_000_000),
    RunSpec("kl_a_v4_eps_stddev",  0.5,  3.0,  0.1, 1.0, 1e-3, 600_000_000),
    RunSpec("kl_a_v5_full_thaw",   0.5,  3.0,  1.0, 1.0, 1e-3, 600_000_000),
    RunSpec("kl_a_v6_no_anchor",   0.0,  0.0,  1.0, 1.0, 1e-3, 600_000_000),
]


def _ensure_tsv_header():
    if not TSV.exists():
        with open(TSV, "w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([
                "name", "started_at", "ended_at", "duration_s", "status",
                "env_steps_at_exit", "peak_eval_reward",
                "w_anchor", "alpha_anchor", "prior_lr_mult", "decoder_lr_mult",
                "epsilon_stddev", "num_timesteps", "notes",
            ])


def _append_row(row):
    _ensure_tsv_header()
    with open(TSV, "a") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(row)


def _classify_crash(stderr: str) -> str:
    """Return 'easy' if stderr looks like a recoverable bug; else 'fundamental'."""
    s = stderr.lower()
    EASY = (
        "modulenotfounderror", "no module named", "syntaxerror",
        "name '.*' is not defined", "nameerror", "typeerror.*missing",
        "could not connect to wandb", "no such file or directory",
    )
    FUNDAMENTAL = (
        "out of memory", "cuda error: out of memory", "nan",
        "assertionerror", "shape mismatch",
        "Cannot find proprio key",
    )
    for p in FUNDAMENTAL:
        if re.search(p, s):
            return "fundamental"
    for p in EASY:
        if re.search(p, s):
            return "easy"
    return "fundamental"


def _peak_eval_reward_from_log(log_path: Path) -> float:
    if not log_path.exists():
        return float("nan")
    best = float("-inf")
    pat = re.compile(r"eval/cumulative_reward['\"]?\s*[:=]\s*([-\d.eE+]+)")
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return float("nan")
    for m in pat.finditer(text):
        try:
            val = float(m.group(1))
            if val > best:
                best = val
        except ValueError:
            continue
    return best if best > float("-inf") else float("nan")


def _env_steps_at_exit(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    pat = re.compile(r"env_steps=(\d+)")
    best = 0
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return 0
    for m in pat.finditer(text):
        try:
            v = int(m.group(1))
            if v > best:
                best = v
        except ValueError:
            continue
    return best


def run_one(spec: RunSpec) -> dict:
    log_path = LOG_DIR / f"{spec.name}.log"
    cmd = [
        sys.executable, "-m", "vnl_playground.train_highlvl_dmpo_kl_anchor",
        f"--config-name={CONFIG_NAME}",
        *spec.overrides(),
    ]
    log.info("Launching %s", spec.name)
    log.info("Command: %s", " ".join(cmd))
    started = time.time()
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=RUN_BUDGET_S)
            status = "completed" if rc == 0 else "error"
        except subprocess.TimeoutExpired:
            log.warning("Run %s exceeded budget %ds — terminating", spec.name, RUN_BUDGET_S)
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            status = "budget_exceeded"
            rc = -1
    ended = time.time()
    duration = ended - started

    env_steps = _env_steps_at_exit(log_path)
    peak_reward = _peak_eval_reward_from_log(log_path)

    notes = ""
    if status == "error":
        try:
            tail = log_path.read_text(errors="ignore")[-4000:]
        except Exception:
            tail = ""
        cls = _classify_crash(tail)
        notes = f"crash_class={cls}"
        if cls == "fundamental":
            status = "crash"

    row = [
        spec.name,
        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ended)),
        f"{duration:.1f}",
        status,
        env_steps,
        f"{peak_reward:.3f}" if peak_reward == peak_reward else "nan",
        spec.w_anchor, spec.alpha_anchor, spec.prior_lr_mult, spec.decoder_lr_mult,
        spec.epsilon_stddev, spec.num_timesteps,
        notes,
    ]
    _append_row(row)
    return {
        "name": spec.name, "status": status, "rc": rc,
        "env_steps": env_steps, "peak_reward": peak_reward, "notes": notes,
    }


def best_so_far():
    if not TSV.exists():
        return None
    best_row = None
    best_val = float("-inf")
    with open(TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            try:
                v = float(row["peak_eval_reward"])
            except (KeyError, ValueError):
                continue
            if v == v and v > best_val and row["status"] == "completed":
                best_val = v
                best_row = row
    if best_row is None:
        return None
    try:
        return RunSpec(
            name=f"kl_a_perturb_{int(time.time())}",
            w_anchor=float(best_row["w_anchor"]),
            alpha_anchor=float(best_row["alpha_anchor"]),
            prior_lr_mult=float(best_row["prior_lr_mult"]),
            decoder_lr_mult=float(best_row["decoder_lr_mult"]),
            epsilon_stddev=float(best_row["epsilon_stddev"]),
            num_timesteps=int(best_row["num_timesteps"]),
        )
    except (KeyError, ValueError):
        return None


def perturb(base: RunSpec) -> RunSpec:
    rng = random.Random()
    w = base.w_anchor * rng.choice([0.5, 1.0, 2.0])
    alpha = base.alpha_anchor * rng.choice([0.5, 1.0, 2.0])
    plr = base.prior_lr_mult * rng.choice([0.3, 1.0, 3.0])
    dlr = base.decoder_lr_mult * rng.choice([0.5, 1.0, 2.0])
    eps = rng.choice([1e-7, 1e-5, 1e-3, 1e-1])
    plr = max(min(plr, 1.0), 0.001)
    dlr = max(min(dlr, 5.0), 0.1)
    return RunSpec(
        name=f"kl_a_perturb_{int(time.time())}",
        w_anchor=float(w),
        alpha_anchor=float(alpha),
        prior_lr_mult=float(plr),
        decoder_lr_mult=float(dlr),
        epsilon_stddev=float(eps),
        num_timesteps=base.num_timesteps,
    )


def main():
    log.info("Starting kl-anchor experiment loop. TSV at %s", TSV)
    log.info("Per-run budget %ds, MAX_ITERS=%d", RUN_BUDGET_S, MAX_ITERS)
    queue = list(SEED_SPECS)
    seen_names = set()
    iteration = 0
    while iteration < MAX_ITERS:
        iteration += 1
        if queue:
            spec = queue.pop(0)
        else:
            base = best_so_far()
            if base is None:
                log.info("No completed run yet; replaying seed queue")
                queue = list(SEED_SPECS)
                continue
            spec = perturb(base)
        if spec.name in seen_names:
            spec = RunSpec(**{**asdict(spec), "name": f"{spec.name}_r{iteration}"})
        seen_names.add(spec.name)
        result = run_one(spec)
        log.info("Run %s finished: %s", spec.name, json.dumps(result, default=str))
        if result["status"] == "crash" and "easy" in (result.get("notes") or ""):
            log.warning("Easy-crash retry for %s", spec.name)
            retry = RunSpec(**{**asdict(spec), "name": f"{spec.name}_retry"})
            run_one(retry)
    log.info("MAX_ITERS=%d reached. Exiting.", MAX_ITERS)


if __name__ == "__main__":
    main()
