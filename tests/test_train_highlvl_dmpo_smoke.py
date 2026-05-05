"""Smoke test: train_highlvl_dmpo.main runs >=1 fused step without crashing.

Uses tiny num_envs / num_timesteps so the test completes on a workstation
in <2 min. WANDB_MODE=disabled to avoid network calls. eval rendering off.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_smoke_one_chunk(tmp_path):
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    # Keep XLA from preallocating so the test can share GPU with other procs.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    overrides = [
        "--config-name=rodent_run_gap_dmpo/velocity_only",
        # Tiny so the test finishes fast and survives small-VRAM CI machines.
        "train_config.num_envs=8",
        "train_config.num_timesteps=4096",
        "train_config.unroll_length=8",
        "train_config.batch_size=8",
        "train_config.sequence_length=8",
        "train_config.min_replay_size=64",
        "train_config.max_replay_size=512",
        "train_config.iters_per_chunk=1",
        "train_config.log_every_steps=64",
        "train_config.eval_every_steps=999_999_999",
        "env_config.episode_length=64",
        "env_config.naconmax=512",
        f"++checkpoint_dir={tmp_path}",
        "++eval_render_config.enable=false",
    ]
    cmd = [sys.executable, "-m", "vnl_playground.train_highlvl_dmpo", *overrides]
    result = subprocess.run(
        cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    assert result.returncode == 0, "train_highlvl_dmpo smoke run failed"
    # Sanity: at least one chunk metric was logged.
    assert "chunk env_steps=" in result.stdout or "chunk env_steps=" in result.stderr, (
        "No chunk metrics found - did training_loop.run actually iterate?"
    )
