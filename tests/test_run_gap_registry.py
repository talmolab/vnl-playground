"""RodentRunGap is registered and builds on the CPU."""

import jax
import pytest

from vnl_playground import tasks
from vnl_playground.tasks.rodent import consts


def test_run_gap_registered_with_default_config():
    cfg = tasks.get_default_config("RodentRunGap")
    assert (
        "gap_length_range" in cfg or "gap_length" in cfg
    )  # one of the two forms the task uses
    assert consts.CORRIDOR_ARENA_XML_PATH.exists()


@pytest.mark.slow
def test_run_gap_resets_and_steps_on_cpu():
    cfg = tasks.get_default_config("RodentRunGap")
    # RunGap's platforms are box geoms colliding against the rodent's
    # ellipsoid collision primitives; the pure-jax mjx backend does not
    # implement (ellipsoid, box) collisions, so this task must keep the
    # "warp" backend (its default), which runs CPU-only when no CUDA
    # device is present.
    cfg.mujoco_impl = "warp"
    # flatten_obs=False to keep a dict-shaped observation for the assertion
    # below; tasks.load's default (flatten_obs=True) returns a flat array.
    env = tasks.load("RodentRunGap", config=cfg, flatten_obs=False)
    state = env.reset(jax.random.key(0))
    state = env.step(state, jax.numpy.zeros(env.action_size))
    assert "proprioception" in state.obs or "state" in state.obs


@pytest.mark.slow
def test_run_gap_flattened_obs_on_cpu():
    """Test the default flattened-observation path used by training.

    This covers the flatten_obs=True path (the training default), in contrast
    to test_run_gap_resets_and_steps_on_cpu which uses flatten_obs=False.
    """
    cfg = tasks.get_default_config("RodentRunGap")
    # Use the default mujoco_impl (warp) and default flatten_obs (True).
    env = tasks.load("RodentRunGap", config=cfg)
    state = env.reset(jax.random.key(0))
    state = env.step(state, jax.numpy.zeros(env.action_size))
    # Flattened observations should be a 1D array with positive size.
    assert state.obs.ndim == 1
    assert state.obs.shape[0] > 0
