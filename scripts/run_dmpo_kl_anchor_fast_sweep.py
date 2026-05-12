"""Fast autonomous sweep for kl-anchor (post-warmstart-fix).

Designed for ~7-15 min/experiment iteration so we can map the
(alpha_anchor, w_anchor, prior_lr_mult, epsilon_stddev) hyperparameter
grid quickly now that the warm-start invariant is fixed.

Differences from `run_dmpo_kl_anchor_loop.py`:
- 50M timesteps per run (≈12 min on a 5090 vs 90 min for the original 600M)
- Tighter wall-clock budget (1500s = 25 min/run)
- Shorter eval cadence (every 10M = 5 evals per 50M run)
- Curated initial sweep grid (8 cells covering the most informative
  combinations of {alpha=0,0.5,2,5} × {prior_lr=0.1,1.0})
- After the seed grid, perturbs the best-so-far run with smaller deltas
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "logs" / "dmpo_kl_anchor_fast_sweep.tsv"
LOG_DIR = REPO / "logs" / "dmpo_kl_anchor_fast"
CKPT_ROOT = REPO / "checkpoints"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TSV.parent.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_BUDGET_S = int(os.environ.get("DMPO_FAST_BUDGET_S", str(15 * 60)))
MAX_ITERS = int(os.environ.get("DMPO_FAST_MAX_ITERS", "200"))
NUM_TIMESTEPS = int(os.environ.get("DMPO_FAST_NUM_TIMESTEPS", "25_000_000"))

CONFIG_NAME = "rodent_run_gap_dmpo/velocity_only_kl_anchor"

log = logging.getLogger("dmpo_kl_anchor_fast")
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
    num_timesteps: int = NUM_TIMESTEPS

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
            "++train_config.eval_every_steps=10000000",
        ]


# Seed grid: probe (alpha, prior_lr_mult) cross with informative w/eps
# Names start with `kw_` ("kl-anchor warm-fix") to distinguish from the
# previous-broken `kl_a_*` checkpoints in /checkpoints.
SEED_SPECS = [
    # No-anchor baseline (full thaw): is the warm-start alone enough?
    RunSpec("kw_a0_thaw",     0.0, 0.0, 1.0, 1.0, 1e-3),
    # Light anchor + thaw: imit teacher early, free exploration later
    RunSpec("kw_a05_thaw",    0.5, 0.5, 1.0, 1.0, 1e-3),
    RunSpec("kw_a2_thaw",     0.5, 2.0, 1.0, 1.0, 1e-3),
    # Strong anchor + thaw: lock close to imit
    RunSpec("kw_a5_thaw",     0.5, 5.0, 1.0, 1.0, 1e-3),
    # Same alpha grid but with protected prior (prior_lr_mult=0.1)
    RunSpec("kw_a0_prot",     0.0, 0.0, 0.1, 1.0, 1e-3),
    RunSpec("kw_a05_prot",    0.5, 0.5, 0.1, 1.0, 1e-3),
    RunSpec("kw_a2_prot",     0.5, 2.0, 0.1, 1.0, 1e-3),
    RunSpec("kw_a5_prot",     0.5, 5.0, 0.1, 1.0, 1e-3),
    # Tight epsilon_stddev variants (mirror plan's kl_a_v1 setup but with fix)
    RunSpec("kw_v1_replay",   0.5, 3.0, 0.1, 1.0, 1e-7),
    RunSpec("kw_v6_replay",   0.0, 0.0, 1.0, 1.0, 1e-7),
]


def _ensure_tsv_header():
    if not TSV.exists():
        with open(TSV, "w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([
                "name", "started_at", "ended_at", "duration_s", "status",
                "env_steps_at_exit", "peak_eval_reward",
                "first_probe_r_anchor", "final_policy_loss", "final_critic_loss",
                "w_anchor", "alpha_anchor", "prior_lr_mult", "decoder_lr_mult",
                "epsilon_stddev", "num_timesteps", "notes",
            ])


def _append_row(row):
    _ensure_tsv_header()
    with open(TSV, "a") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(row)


_PROBE_PAT = re.compile(r"anchor_invariant_probe r_anchor=([\d.]+) action_mse=([\d.]+)")
_CHUNK_PAT = re.compile(r"chunk env_steps=(\d+).*policy_loss=([-\d.eE+]+)\s+critic_loss=([-\d.eE+]+)")
_EVAL_PAT = re.compile(r"eval/cumulative_reward['\"]?\s*[:=\s]+\s*([-\d.eE+]+)")
_EVAL_LEN_PAT = re.compile(r"eval/mean_episode_length['\"]?\s*[:=\s]+\s*([-\d.eE+]+)")
_GAP_PAT = re.compile(r"eval/total_gap_crossings['\"]?\s*[:=\s]+\s*([-\d.eE+]+)")


def _parse_log(log_path: Path) -> dict:
    out = {
        "first_probe_r_anchor": float("nan"),
        "first_probe_mse": float("nan"),
        "env_steps": 0,
        "final_policy_loss": float("nan"),
        "final_critic_loss": float("nan"),
        "peak_eval_reward": float("-inf"),
        "final_eval_len": float("nan"),
        "total_gap_crossings": 0,
    }
    if not log_path.exists():
        return out
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return out
    m = _PROBE_PAT.search(text)
    if m:
        out["first_probe_r_anchor"] = float(m.group(1))
        out["first_probe_mse"] = float(m.group(2))
    last_chunk = None
    for m in _CHUNK_PAT.finditer(text):
        last_chunk = m
        try:
            v = int(m.group(1))
            if v > out["env_steps"]:
                out["env_steps"] = v
        except ValueError:
            pass
    if last_chunk is not None:
        try:
            out["final_policy_loss"] = float(last_chunk.group(2))
            out["final_critic_loss"] = float(last_chunk.group(3))
        except ValueError:
            pass
    for m in _EVAL_PAT.finditer(text):
        try:
            v = float(m.group(1))
            if v > out["peak_eval_reward"]:
                out["peak_eval_reward"] = v
        except ValueError:
            continue
    if out["peak_eval_reward"] == float("-inf"):
        out["peak_eval_reward"] = float("nan")
    # Track the last (most recent) episode-length value as final_eval_len.
    last_len = None
    for m in _EVAL_LEN_PAT.finditer(text):
        last_len = m
    if last_len is not None:
        try:
            out["final_eval_len"] = float(last_len.group(1))
        except ValueError:
            pass
    # Sum gap crossings across all eval reports (max value over a single eval).
    best_gap = 0
    for m in _GAP_PAT.finditer(text):
        try:
            v = int(float(m.group(1)))
            if v > best_gap:
                best_gap = v
        except ValueError:
            continue
    out["total_gap_crossings"] = best_gap
    return out


def _classify_status(rc: int, log_path: Path) -> tuple[str, str]:
    if rc == 0:
        return "completed", ""
    try:
        tail = log_path.read_text(errors="ignore")[-4000:]
    except Exception:
        tail = ""
    s = tail.lower()
    if "out of memory" in s or "cuda error: out of memory" in s:
        return "oom", "GPU OOM"
    if "nan" in s and "loss" in s:
        return "nan", "NaN in loss"
    if "traceback" in s:
        return "crash", tail[-500:]
    return "error", tail[-500:]


def run_one(spec: RunSpec) -> dict:
    log_path = LOG_DIR / f"{spec.name}.log"
    cmd = [
        sys.executable, "-m", "vnl_playground.train_highlvl_dmpo_kl_anchor",
        f"--config-name={CONFIG_NAME}",
        *spec.overrides(),
    ]
    log.info("Launching %s", spec.name)
    log.info("  config: w=%.2f alpha=%.2f plr=%.2f dlr=%.2f eps=%.0e steps=%dM",
             spec.w_anchor, spec.alpha_anchor, spec.prior_lr_mult,
             spec.decoder_lr_mult, spec.epsilon_stddev, spec.num_timesteps // 1_000_000)
    started = time.time()
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT,
                                env=env)
        try:
            rc = proc.wait(timeout=RUN_BUDGET_S)
            status_kind, notes = _classify_status(rc, log_path)
        except subprocess.TimeoutExpired:
            log.warning("Run %s exceeded budget %ds — terminating", spec.name, RUN_BUDGET_S)
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            status_kind = "budget_exceeded"
            notes = ""
            rc = -1
    ended = time.time()
    duration = ended - started

    parsed = _parse_log(log_path)

    row = [
        spec.name,
        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ended)),
        f"{duration:.1f}",
        status_kind,
        parsed["env_steps"],
        f"{parsed['peak_eval_reward']:.3f}" if parsed["peak_eval_reward"] == parsed["peak_eval_reward"] else "nan",
        f"{parsed['first_probe_r_anchor']:.4f}" if parsed["first_probe_r_anchor"] == parsed["first_probe_r_anchor"] else "nan",
        f"{parsed['final_policy_loss']:.3f}" if parsed["final_policy_loss"] == parsed["final_policy_loss"] else "nan",
        f"{parsed['final_critic_loss']:.3f}" if parsed["final_critic_loss"] == parsed["final_critic_loss"] else "nan",
        spec.w_anchor, spec.alpha_anchor, spec.prior_lr_mult, spec.decoder_lr_mult,
        spec.epsilon_stddev, spec.num_timesteps,
        notes,
    ]
    _append_row(row)

    log.info("  ✓ %s: status=%s steps=%dM probe=%s final_pl=%s final_cl=%s peak_eval=%s",
             spec.name, status_kind, parsed["env_steps"] // 1_000_000,
             f"{parsed['first_probe_r_anchor']:.3f}" if parsed["first_probe_r_anchor"] == parsed["first_probe_r_anchor"] else "nan",
             f"{parsed['final_policy_loss']:.2f}" if parsed["final_policy_loss"] == parsed["final_policy_loss"] else "nan",
             f"{parsed['final_critic_loss']:.2f}" if parsed["final_critic_loss"] == parsed["final_critic_loss"] else "nan",
             f"{parsed['peak_eval_reward']:.1f}" if parsed["peak_eval_reward"] == parsed["peak_eval_reward"] else "nan",
             )
    return {
        "name": spec.name, "status": status_kind, "rc": rc, "duration_s": duration,
        **parsed,
    }


def best_so_far() -> RunSpec | None:
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
            name=f"kw_perturb_{int(time.time())}",
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
    # Smaller delta steps than the original loop for finer search.
    w = base.w_anchor * rng.choice([0.7, 1.0, 1.4]) if base.w_anchor > 0 else rng.choice([0.0, 0.3, 0.5])
    alpha = base.alpha_anchor * rng.choice([0.7, 1.0, 1.4]) if base.alpha_anchor > 0 else rng.choice([0.0, 0.5, 1.5])
    plr = base.prior_lr_mult * rng.choice([0.5, 1.0, 2.0])
    dlr = base.decoder_lr_mult * rng.choice([0.7, 1.0, 1.4])
    eps = rng.choice([1e-7, 1e-5, 1e-3, 1e-1])
    plr = max(min(plr, 2.0), 0.01)
    dlr = max(min(dlr, 5.0), 0.1)
    return RunSpec(
        name=f"kw_perturb_{int(time.time())}",
        w_anchor=float(w),
        alpha_anchor=float(alpha),
        prior_lr_mult=float(plr),
        decoder_lr_mult=float(dlr),
        epsilon_stddev=float(eps),
        num_timesteps=base.num_timesteps,
    )


def main():
    log.info("Starting kl-anchor FAST sweep. TSV: %s", TSV)
    log.info("Per-run budget %ds (%.1f min), default num_timesteps=%dM, MAX_ITERS=%d",
             RUN_BUDGET_S, RUN_BUDGET_S/60, NUM_TIMESTEPS // 1_000_000, MAX_ITERS)
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
        log.info("Iteration %d/%d done: %s", iteration, MAX_ITERS,
                 json.dumps({k: v for k, v in result.items() if k in ("name","status","duration_s","first_probe_r_anchor","peak_eval_reward")}, default=str))
    log.info("MAX_ITERS=%d reached. Exiting.", MAX_ITERS)


if __name__ == "__main__":
    main()
