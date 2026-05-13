"""End-to-end smoke: load StickImitation, reset, step, verify reference data."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def stick_env():
    from vnl_playground import registry
    env = registry.load("StickImitation", flatten_obs=False)
    return env


def test_env_compiles_with_expected_shapes(stick_env):
    # Free joint (7) + 41 hinge joints = 48 qpos.
    assert stick_env.mj_model.nq == 48
    assert stick_env.mj_model.nu == 41
    # 6 floor<->claw pairs added by add_stick().
    pairs = [
        (stick_env.mj_model.pair(i).geom1, stick_env.mj_model.pair(i).geom2)
        for i in range(stick_env.mj_model.npair)
    ]
    assert len(pairs) == 6


def test_reset_and_step_returns_finite(stick_env):
    state = stick_env.reset(jax.random.PRNGKey(0))
    assert jp.all(jp.isfinite(state.data.qpos))
    next_state = stick_env.step(state, jp.zeros(stick_env.action_size))
    assert jp.all(jp.isfinite(next_state.data.qpos))
    assert jp.isfinite(next_state.reward)


def test_verify_reference_data_passes(stick_env):
    """The STAC fit's qpos must reproduce its xpos in the env (within atol).

    We load with `flatten_obs=False`, so the env returned by registry.load() is
    the raw Imitation instance (no wrapper) — call verify_reference_data
    directly.
    """
    ok = stick_env.verify_reference_data(atol=5e-3)
    assert ok, "reference_data verification reported failures (see warnings)"
