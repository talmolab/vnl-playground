"""InfoResetOnDoneWrapper clears per-episode info keys when an episode ends.

The gap_crossing_bonus ratchet (BASELINE_w2) came from AutoResetWrapper
carrying state.info across episodes; this wrapper resets info on done.
"""

import jax
import jax.numpy as jnp
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.wrappers_info_reset import InfoResetOnDoneWrapper


class _Counting:
    """Minimal env: info['count'] increments every step; done at step 3."""

    action_size = 1
    observation_size = 1

    def reset(self, rng):
        return mjx_env.State(
            data=None,
            obs=jnp.zeros(1),
            reward=jnp.float32(0),
            done=jnp.float32(0),
            metrics={},
            info={"count": jnp.int32(0)},
        )

    def step(self, state, action):
        # Real envs carry forward info keys they don't touch (mutate in place
        # or merge); mimic that here so the wrapper's cache key survives.
        count = state.info["count"] + 1
        info = dict(state.info)
        info["count"] = count
        return state.replace(info=info, done=jnp.float32(count >= 3))


def test_info_is_reset_on_done():
    env = InfoResetOnDoneWrapper(_Counting(), keys=("count",))
    s = env.reset(jax.random.key(0))
    for _ in range(3):
        s = env.step(s, jnp.zeros(1))
    assert float(s.done) == 1.0
    s = env.step(s, jnp.zeros(1))
    assert int(s.info["count"]) == 1, "count must restart after the done step"
