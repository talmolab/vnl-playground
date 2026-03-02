"""Tests for sparse reward gap-jump trial environment."""

import jax
import jax.numpy as jp
import pytest

from vnl_playground.tasks.rodent.gap_jump_trial import (
    GapJumpTrial,
    default_config,
    dense_config,
    OUTCOME_ONGOING,
    OUTCOME_SUCCESS,
    OUTCOME_FAILURE,
    OUTCOME_ABORT,
    OUTCOME_TIMEOUT,
    PHASE_HOLD,
    PHASE_DECISION,
    PHASE_JUMP,
)


@pytest.fixture
def env():
    cfg = default_config()
    return GapJumpTrial(config=cfg)


def test_reset_has_outcome_tracking(env):
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    assert "trial_outcome" in state.info
    assert int(state.info["trial_outcome"]) == OUTCOME_ONGOING


def test_trial_success_termination_registered(env):
    assert "trial_success" in env._registry.terminations


def test_abort_dismount_termination_registered(env):
    assert "abort_dismount" in env._registry.terminations


def test_time_penalty_registered(env):
    assert "time_penalty" in env._registry.rewards


def test_sparse_config_has_no_dense_rewards():
    cfg = default_config()
    assert "forward_displacement" not in cfg.reward_terms
    assert "approach_velocity" not in cfg.reward_terms


def test_sparse_config_has_sparse_rewards():
    cfg = default_config()
    assert "jump_success" in cfg.reward_terms
    assert "fall_penalty" in cfg.reward_terms
    assert "abort_penalty" in cfg.reward_terms
    assert "time_penalty" in cfg.reward_terms
    assert "hold_stillness" in cfg.reward_terms
    assert cfg.reward_terms["jump_success"]["weight"] >= 100.0


def test_dense_config_exists():
    cfg = dense_config()
    assert "forward_displacement" in cfg.reward_terms
    assert "approach_velocity" in cfg.reward_terms


@pytest.mark.parametrize(
    "gap_distances",
    [
        (0.000, 0.005, 0.010, 0.015, 0.020),  # Phase 1
        (0.02, 0.03, 0.04, 0.05, 0.06),  # Phase 2
        (0.06, 0.08, 0.10, 0.12, 0.14),  # Phase 3
    ],
)
def test_sparse_env_reset_step_cycle(gap_distances):
    """Env should reset and step without errors for each curriculum phase."""
    cfg = default_config()
    cfg.gap_distances = gap_distances
    env = GapJumpTrial(config=cfg)

    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    assert state.reward.shape == ()
    assert state.done.shape == ()
    assert int(state.info["trial_outcome"]) == OUTCOME_ONGOING

    # Step 10 iterations with zero action
    for _ in range(10):
        action = jp.zeros(env.action_size)
        state = env.step(state, action)
        assert not jp.any(jp.isnan(state.reward))


def test_zero_gap_terminates_on_success():
    """With zero gap, walking forward should trigger success termination."""
    cfg = default_config()
    cfg.gap_distances = (0.00,)
    cfg.hold_duration = 0
    env = GapJumpTrial(config=cfg)

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    for _ in range(500):
        action = jp.zeros(env.action_size)
        state = env.step(state, action)
        if state.done > 0.5:
            break

    # Should have terminated
    assert state.done > 0.5


def test_vision_config_inherits_sparse():
    """Vision config should inherit sparse rewards from base."""
    from vnl_playground.tasks.rodent.gap_jump_trial_vision import (
        default_config as vision_default,
    )

    cfg = vision_default()
    assert "trial_success" in cfg.termination_criteria
    assert "abort_dismount" in cfg.termination_criteria
    assert "jump_success" in cfg.reward_terms
    assert cfg.reward_terms["jump_success"]["weight"] == 100.0
    assert "forward_displacement" not in cfg.reward_terms
