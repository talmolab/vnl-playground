"""1-chunk subprocess smoke test for train_highlvl_dmpo_kl_anchor."""
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
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    overrides = [
        "--config-name=rodent_run_gap_dmpo/velocity_only_kl_anchor",
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
    cmd = [sys.executable, "-m", "vnl_playground.train_highlvl_dmpo_kl_anchor", *overrides]
    result = subprocess.run(
        cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    assert result.returncode == 0, "kl_anchor smoke run failed"
    assert (
        "chunk env_steps=" in result.stdout
        or "chunk env_steps=" in result.stderr
    ), "No chunk metrics found"
