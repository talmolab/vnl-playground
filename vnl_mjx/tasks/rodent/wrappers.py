from typing import Any, Callable, Mapping

from mujoco import mjx
from mujoco_playground._src import mjx_env

import jax
import jax.numpy as jp

from mujoco_playground import wrapper


class FlattenObsWrapper(wrapper.Wrapper):

    def __init__(self, env: wrapper.mjx_env.MjxEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> wrapper.mjx_env.State:
        state = self.env.reset(rng)
        return self._flatten(state)

    def step(
        self, state: wrapper.mjx_env.State, action: jax.Array
    ) -> wrapper.mjx_env.State:
        state = self.env.step(state, action)
        return self._flatten(state)

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        return jax.flatten_util.ravel_pytree(self.env._get_obs(data, info))[0]

    def _flatten(self, state: wrapper.mjx_env.State) -> wrapper.mjx_env.State:
        state = state.replace(
            obs=jax.flatten_util.ravel_pytree(state.obs)[0],
            metrics=self._flatten_metrics(state.metrics),
        )
        return state

    def _flatten_metrics(self, metrics: dict) -> dict:
        new_metrics = {}

        def rec(d: dict, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    rec(v, prefix + k + "/")
                else:
                    new_metrics[prefix + k] = v

        rec(metrics)
        return new_metrics


class HighLevelWrapper(wrapper.Wrapper):
    """Takes a decoder inference function and uses it to get the ctrl used in the sim step.

    The environment wrapped in this must use the same set of proprioceptive obs as the decoder.
    """

    def __init__(
        self,
        env: wrapper.mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        non_proprioceptive_obs_size: int,
    ):
        self._decoder_inference_fn = decoder_inference_fn
        self._latent_size = latent_size
        self.non_proprioceptive_obs_size = non_proprioceptive_obs_size
        super().__init__(env)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        obs = state.obs

        # Note: We assume the non proprioceptive obs are first indices in obs, followed by proprioceptive obs.
        ctrl, _ = self._decoder_inference_fn(
            jp.concatenate(
                [action, obs[..., self.non_proprioceptive_obs_size :]],
                axis=-1,
            ),
        )
        return self.env.step(state, ctrl)

    @property
    def action_size(self) -> int:
        return self._latent_size
