"""Keep the time-limit flag through auto-reset, for off-policy training.

An off-policy learner must tell a time-out from a real termination: at a
time-out it bootstraps from the value of the next state, at a termination the
value is zero. brax's ``EpisodeWrapper`` records the distinction in
``info["truncation"]`` (1 only when the step limit is hit and the env did not
also terminate), but ``BraxAutoResetWrapper(full_reset=True)`` replaces
``state.info`` with the freshly reset state's on done, which erases the flag
exactly on the step that needs it. The one subtree it carries across is
``info["AutoResetWrapper_preserve_info"]``.

``TruncationStashWrapper`` sits between ``EpisodeWrapper`` and
``BraxAutoResetWrapper`` and copies the post-step flag into that subtree, so a
rollout can read ``state.info["AutoResetWrapper_preserve_info"]["truncation"]``
after the step, under either ``full_reset`` value. ``wrap_for_dmpo_training`` is
``mujoco_playground``'s ``wrap_for_brax_training`` with the stash inserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jp
from brax.envs.wrappers import training as brax_training
from mujoco_playground._src import wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

# BraxAutoResetWrapper passes this info subtree through a full_reset unchanged.
PRESERVE_KEY = "AutoResetWrapper_preserve_info"


class TruncationStashWrapper(wrapper.Wrapper):
    """Copies ``info["truncation"]`` into the auto-reset-preserved info subtree.

    Must wrap a brax ``EpisodeWrapper`` (which writes ``info["truncation"]``)
    and be wrapped by ``BraxAutoResetWrapper``. Other entries already under
    ``info[PRESERVE_KEY]`` are kept.
    """

    def reset(self, rng, **kwargs):
        state = self.env.reset(rng, **kwargs)
        preserved = dict(state.info.get(PRESERVE_KEY, {}))
        preserved["truncation"] = jp.zeros_like(state.done)
        state.info[PRESERVE_KEY] = preserved
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        if "truncation" not in state.info:
            raise KeyError(
                "TruncationStashWrapper: state.info has no 'truncation'; it "
                "must wrap brax's EpisodeWrapper, which writes that key. Present "
                f"keys: {sorted(state.info)}"
            )
        preserved = dict(state.info.get(PRESERVE_KEY, {}))
        preserved["truncation"] = state.info["truncation"]
        state.info[PRESERVE_KEY] = preserved
        return state


def wrap_for_dmpo_training(
    env: Any,
    episode_length: int,
    action_repeat: int = 1,
    full_reset: bool = False,
    randomization_fn: Callable | None = None,
) -> Any:
    """``wrap_for_brax_training`` with the truncation stash inserted.

    Same stack as ``mujoco_playground``'s ``wrap_for_brax_training`` (non-vision):
    vmap (domain-randomised when ``randomization_fn`` is given), then
    ``EpisodeWrapper``, then ``TruncationStashWrapper``, then
    ``BraxAutoResetWrapper``. The rollout contract is that after every step
    ``state.info[PRESERVE_KEY]["truncation"]`` holds the post-step truncation
    flag (1 only for a time-out, 0 for a termination), for both ``full_reset``
    values.
    """
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = TruncationStashWrapper(env)
    return wrapper.BraxAutoResetWrapper(env, full_reset=full_reset)
