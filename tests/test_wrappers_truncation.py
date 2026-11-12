"""TruncationStashWrapper keeps the time-limit flag through auto-reset.

Off-policy learners need to tell a time-out (bootstrap through it) from a real
termination (value 0). brax's ``EpisodeWrapper`` records ``info["truncation"]``,
but ``BraxAutoResetWrapper(full_reset=True)`` replaces ``info`` with the reset
state's on done, erasing it. The stash copies the post-step flag under
``info["AutoResetWrapper_preserve_info"]``, the one subtree that survives.

A toy env with a fixed ``episode_length=5`` is enough; no physics is needed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper as mp_wrapper

from vnl_playground.tasks.wrappers_truncation import (
    PRESERVE_KEY,
    TruncationStashWrapper,
    wrap_for_dmpo_training,
)

EPISODE_LENGTH = 5
BATCH = 3
STEPS = 12


class _Toy:
    """obs = steps taken in the episode; reward 1; optional self-termination.

    Reads the episode step counter that ``EpisodeWrapper`` keeps in
    ``info["steps"]`` (the pre-step count), so it terminates on the same step
    index whatever the reset mode.
    """

    action_size = 1
    observation_size = 1

    def __init__(self, terminate_at=None):
        self._terminate_at = terminate_at

    def reset(self, rng):
        del rng
        return mjx_env.State(
            data=None,
            obs=jnp.zeros(1),
            reward=jnp.float32(0),
            done=jnp.float32(0),
            metrics={},
            info={},
        )

    def step(self, state, action):
        del action
        t = state.info["steps"] + 1
        done = (
            jnp.float32(0)
            if self._terminate_at is None
            else jnp.float32(t >= self._terminate_at)
        )
        return state.replace(obs=jnp.reshape(t, (1,)), reward=jnp.float32(1), done=done)


def _rng():
    # Legacy uint32 keys: brax's wrappers read rng.shape[:-1] as the batch.
    return jax.random.split(jax.random.PRNGKey(0), BATCH)


def _rollout(env, steps=STEPS):
    state = jax.jit(env.reset)(_rng())
    step = jax.jit(env.step)
    out = []
    for _ in range(steps):
        state = step(state, jnp.zeros((BATCH, 1)))
        out.append(state)
    return state, out


def _preserved(state):
    return np.asarray(state.info[PRESERVE_KEY]["truncation"])


@pytest.mark.parametrize("full_reset", [True, False])
def test_truncation_survives_full_reset(full_reset):
    env = wrap_for_dmpo_training(
        _Toy(), episode_length=EPISODE_LENGTH, full_reset=full_reset
    )
    _, states = _rollout(env)
    flags = [float(_preserved(s)[0]) for s in states]
    # 1 on the step that hits the limit, 0 on the next step, then again at the
    # end of the following episode.
    assert flags == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    assert all(np.all(_preserved(s) == _preserved(s)[0]) for s in states)
    if full_reset:
        # The raw flag IS wiped by the auto-reset on that step: this is why the
        # stash exists.
        assert float(states[4].info["truncation"][0]) == 0.0
        assert float(states[4].done[0]) == 1.0


@pytest.mark.parametrize("full_reset", [True, False])
@pytest.mark.parametrize("terminate_at", [5, 3])
def test_termination_is_not_truncation(full_reset, terminate_at):
    env = wrap_for_dmpo_training(
        _Toy(terminate_at=terminate_at),
        episode_length=EPISODE_LENGTH,
        full_reset=full_reset,
    )
    _, states = _rollout(env)
    assert float(states[terminate_at - 1].done[0]) == 1.0
    assert all(float(_preserved(s)[0]) == 0.0 for s in states)


@pytest.mark.parametrize("full_reset", [True, False])
@pytest.mark.parametrize("terminate_at", [None, 3])
def test_composition_matches_brax_training_otherwise(full_reset, terminate_at):
    ours = wrap_for_dmpo_training(
        _Toy(terminate_at), episode_length=EPISODE_LENGTH, full_reset=full_reset
    )
    ref = mp_wrapper.wrap_for_brax_training(
        _Toy(terminate_at), episode_length=EPISODE_LENGTH, full_reset=full_reset
    )
    _, ours_states = _rollout(ours)
    _, ref_states = _rollout(ref)
    for a, b in zip(ours_states, ref_states):
        np.testing.assert_array_equal(np.asarray(a.obs), np.asarray(b.obs))
        np.testing.assert_array_equal(np.asarray(a.reward), np.asarray(b.reward))
        np.testing.assert_array_equal(np.asarray(a.done), np.asarray(b.done))
        np.testing.assert_array_equal(
            np.asarray(a.info["steps"]), np.asarray(b.info["steps"])
        )


def test_wrapper_chain_order():
    """EpisodeWrapper -> stash -> BraxAutoResetWrapper, outermost last."""
    env = wrap_for_dmpo_training(_Toy(), episode_length=EPISODE_LENGTH)
    chain = []
    while hasattr(env, "env"):
        chain.append(type(env).__name__)
        env = env.env
    assert chain[:3] == [
        "BraxAutoResetWrapper",
        "TruncationStashWrapper",
        "EpisodeWrapper",
    ]


def test_stash_requires_episode_wrapper_below():
    """Without brax's EpisodeWrapper underneath there is no flag to stash."""

    class _Bare(_Toy):
        def step(self, state, action):
            return state

    env = TruncationStashWrapper(_Bare())
    state = env.reset(_rng())
    assert float(state.info[PRESERVE_KEY]["truncation"]) == 0.0
    with pytest.raises(KeyError, match="EpisodeWrapper"):
        env.step(state, jnp.zeros((BATCH, 1)))


def test_other_preserved_info_is_kept():
    """Curriculum code may already use the preserve subtree; the stash adds to it."""

    class _Curriculum(_Toy):
        def reset(self, rng):
            state = super().reset(rng)
            state.info[PRESERVE_KEY] = {"level": jnp.float32(2)}
            return state

    env = wrap_for_dmpo_training(
        _Curriculum(), episode_length=EPISODE_LENGTH, full_reset=True
    )
    _, states = _rollout(env)
    for s in states:
        assert set(s.info[PRESERVE_KEY]) == {"level", "truncation"}
        assert np.all(np.asarray(s.info[PRESERVE_KEY]["level"]) == 2.0)
