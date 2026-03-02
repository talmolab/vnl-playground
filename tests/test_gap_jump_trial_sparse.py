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
