"""Common wrappers for MjxEnv-based environments.

These wrappers are generic and work with any MjxEnv-based environment,
including rodent, fruitfly, and future organisms.
"""

from typing import Any, Callable, Mapping

from mujoco import mjx
from mujoco_playground._src import mjx_env

import jax
import jax.numpy as jp

from mujoco_playground import wrapper


class FlattenObsWrapper(wrapper.Wrapper):
    """Wrapper that flattens hierarchical observations to 1D arrays.

    Converts nested observation dictionaries into flat JAX arrays,
    handling NaN values and flattening nested metrics dictionaries.
    """

    def __init__(self, env: wrapper.mjx_env.MjxEnv):
        super().__init__(env)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> wrapper.mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        return self._flatten(state)

    def step(
        self, state: wrapper.mjx_env.State, action: jax.Array
    ) -> wrapper.mjx_env.State:
        state = self.env.step(state, action)
        return self._flatten(state)

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = jax.flatten_util.ravel_pytree(self.env._get_obs(data, info))[0]
        obs = jp.nan_to_num(obs)
        return obs

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

    @property
    def unwrapped(self) -> mjx_env.MjxEnv:
        return self

    @property
    def _mjx_model(self):
        return self.env._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self.env._mjx_model = value


class HighLevelWrapper(wrapper.Wrapper):
    """Wrapper that uses a decoder to convert latent actions to control signals.

    Takes a decoder inference function and uses it to map high-level latent
    actions to low-level control signals for the environment.

    The environment wrapped in this must use the same set of proprioceptive
    observations as the decoder.

    The environment must return observations as a dict/OrderedDict with separate
    keys for task observations and proprioception. Only task observations are
    exposed to the high-level policy; proprioception is passed to the decoder.

    Args:
        env: The base environment to wrap.
        decoder_inference_fn: Function that maps (latent + proprioception) -> ctrl.
        latent_size: Size of the latent action space.
        highlvl_obs_key: Key for high-level policy observations (default: 'task_obs').
        decoder_obs_key: Key for decoder observations (default: 'proprioception').
    """

    def __init__(
        self,
        env: wrapper.mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        highlvl_obs_key: str = "task_obs",
        decoder_obs_key: str = "proprioception",
    ):
        self._decoder_inference_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._highlvl_obs_key = highlvl_obs_key
        self._decoder_obs_key = decoder_obs_key
        self._proprioceptive_obs_size = int(env.proprioceptive_obs_size)

        super().__init__(env)
        sample_state = env.reset(jax.random.PRNGKey(0))
        if not isinstance(sample_state.obs, Mapping):
            raise ValueError(
                "HighLevelWrapper requires dict observations. "
                f"Got {type(sample_state.obs).__name__} instead."
            )

        # Compute observation size from the task_obs key
        task_obs = sample_state.obs[self._highlvl_obs_key]
        self._task_obs_size = int(
            jax.flatten_util.ravel_pytree(task_obs)[0].shape[0]
        )

        _, self._dummy_decoder_extras = decoder_inference_fn(
            jp.zeros(self._latent_size + self._proprioceptive_obs_size)
        )

    def _flatten_obs_value(self, obs_value: Any) -> jax.Array:
        """Flatten an observation value (handles nested dicts)."""
        flat, _ = jax.flatten_util.ravel_pytree(obs_value)
        return jp.nan_to_num(flat)

    def _process_state(self, state: wrapper.mjx_env.State) -> wrapper.mjx_env.State:
        """Process state to extract task obs for high-level policy."""
        task_obs = self._flatten_obs_value(state.obs[self._highlvl_obs_key])
        # Store full dict obs in info for decoder access
        state.info["_full_obs"] = state.obs
        return state.replace(obs=task_obs)

    def _get_decoder_obs(self, state: wrapper.mjx_env.State) -> jax.Array:
        """Get proprioceptive observations for the decoder."""
        full_obs = state.info["_full_obs"]
        return self._flatten_obs_value(full_obs[self._decoder_obs_key])

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> wrapper.mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        state.info["decoder_extras"] = self._dummy_decoder_extras
        return self._process_state(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Get proprioceptive observations for decoder
        decoder_obs = self._get_decoder_obs(state)

        # Decode latent action to control signal
        ctrl, extras = self._decoder_inference_fn(
            jp.concatenate([action, decoder_obs], axis=-1),
        )

        # Step the underlying environment
        # Note: MjxEnv.step uses state.pipeline_state for physics, not state.obs,
        # so passing the processed state is safe.
        next_state = self.env.step(state, ctrl)
        next_state.info["decoder_extras"] = extras
        return self._process_state(next_state)

    @property
    def action_size(self) -> int:
        return self._latent_size

    @property
    def observation_size(self) -> int:
        """Return observation size for the high-level policy."""
        return self._task_obs_size
