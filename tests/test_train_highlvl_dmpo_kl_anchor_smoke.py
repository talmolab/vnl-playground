"""1-chunk subprocess smoke test for train_highlvl_dmpo_kl_anchor."""
import os
import re
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

    # Parse the startup invariant probe and assert r_anchor > 0.85.
    # The probe runs a single env.reset + policy.mode + bind + env.step cycle
    # right after warm-start setup, before the training loop. Because we use
    # `.mode()` (deterministic), the warm-started policy's bound action equals
    # tanh(mu_imit_pretanh) up to numerics — so r_anchor should be ~1.0.
    # The 0.85 threshold is conservative; even a small regression in any of:
    # normalizer seeding, sigma parameterization, or warm-start splice will
    # push r_anchor well below this.
    m = re.findall(
        r"anchor_invariant_probe r_anchor=([\d.]+) action_mse=([\d.]+)",
        result.stdout + result.stderr,
    )
    assert m, (
        "No anchor_invariant_probe line in subprocess output. Either training "
        "crashed before the probe fired, or the probe block was removed from "
        "train_highlvl_dmpo_kl_anchor.py. stdout:\n" + result.stdout[-2000:]
    )
    first_r, first_mse = m[0]
    first_r = float(first_r)
    assert first_r > 0.85, (
        f"Warm-start invariant broken: probe r_anchor={first_r:.4f} "
        f"(expected > 0.85, with .mode() actions ~1.0), action_mse={first_mse}. "
        "The trainable pipeline is not reproducing the imit pipeline at step 0. "
        "Check: (1) seed_proprio_from_imit is being called, (2) policy uses "
        "softplus(log_std)+1e-3 not exp, (3) warm-start params are spliced."
    )
