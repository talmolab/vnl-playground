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


class BraxObsWrapper(wrapper.Wrapper):
    """Wrapper that flattens each top-level obs value into a single 1D array.

    Input:  {state: OrderedDict(task_obs=..., proprioception=...)}
    Output: {state: jax.Array}

    If privileged_state is present:
    Input:  {state: ..., privileged_state: ...}
    Output: {state: jax.Array, privileged_state: jax.Array}
    """

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> wrapper.mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        return state.replace(obs=self._flatten_obs(state.obs))

    def step(
        self, state: wrapper.mjx_env.State, action: jax.Array
    ) -> wrapper.mjx_env.State:
        state = self.env.step(state, action)
        return state.replace(obs=self._flatten_obs(state.obs))

    @staticmethod
    def _flatten_obs(obs):
        return {
            k: jp.nan_to_num(jax.flatten_util.ravel_pytree(v)[0])
            for k, v in obs.items()
        }


class TrackMjxObsWrapper(wrapper.Wrapper):
    """Wrapper that flattens each second-level obs value into a 1D array.

    Input:  {state: OrderedDict(task_obs=nested, proprioception=nested)}
    Output: {state: {task_obs: jax.Array, proprioception: jax.Array}}

    If privileged_state is present, it is flattened the same way.
    """

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> wrapper.mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        return state.replace(obs=self._flatten_obs(state.obs))

    def step(
        self, state: wrapper.mjx_env.State, action: jax.Array
    ) -> wrapper.mjx_env.State:
        state = self.env.step(state, action)
        return state.replace(obs=self._flatten_obs(state.obs))

    @staticmethod
    def _flatten_obs(obs):
        return {
            k: {
                k2: jp.nan_to_num(jax.flatten_util.ravel_pytree(v2)[0])
                for k2, v2 in v.items()
            }
            for k, v in obs.items()
        }


class LegacyObsWrapper(wrapper.Wrapper):
    """Wrapper that strips the state/privileged_state hierarchy from observations.

    Replaces obs with obs["state"], restoring the flat observation structure
    used by checkpoints trained before the asymmetric obs hierarchy was added.
    """

    def __init__(self, env: wrapper.mjx_env.MjxEnv, obs_key: str = "state"):
        super().__init__(env)
        self._obs_key = obs_key

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> wrapper.mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        return state.replace(obs=state.obs[self._obs_key])

    def step(
        self, state: wrapper.mjx_env.State, action: jax.Array
    ) -> wrapper.mjx_env.State:
        state = self.env.step(state, action)
        return state.replace(obs=state.obs[self._obs_key])

    @property
    def non_flattened_observation_size(self):
        return self.env.non_flattened_observation_size[self._obs_key]

    @property
    def observation_size(self):
        return jp.sum(
            jax.flatten_util.ravel_pytree(self.non_flattened_observation_size)[0]
        )

    @property
    def non_proprioceptive_obs_size(self):
        return self.observation_size - self.proprioceptive_obs_size


class HighLevelWrapper(wrapper.Wrapper):
    """Wrapper that uses a decoder to convert latent actions to control signals.

    Takes a decoder inference function and uses it to map high-level latent
    actions to low-level control signals for the environment.

    The environment wrapped in this must use the same set of proprioceptive
    observations as the decoder.

    The environment must return observations as a nested dict/OrderedDict with
    a top-level 'state' key containing 'task_obs' and 'proprioception'.
    For asymmetric actor-critic, set value_obs_key to a different top-level key
    (e.g. 'privileged_state') so the critic sees privileged information.

    Args:
        env: The base environment to wrap.
        decoder_inference_fn: Function that maps (latent + proprioception) -> ctrl.
        latent_size: Size of the latent action space.
        policy_obs_key: Top-level obs key for the policy/actor (default: 'state').
        value_obs_key: Top-level obs key for the value/critic (default: 'state').
        highlvl_obs_key: Key for high-level policy observations (default: 'task_obs').
        decoder_obs_key: Key for decoder observations (default: 'proprioception').
    """

    def __init__(
        self,
        env: wrapper.mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        policy_obs_key: str = "state",
        value_obs_key: str = "state",
        highlvl_obs_key: str = "task_obs",
        decoder_obs_key: str = "proprioception",
    ):
        super().__init__(env)
        self._decoder_inference_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._policy_obs_key = policy_obs_key
        self._value_obs_key = value_obs_key
        self._highlvl_obs_key = highlvl_obs_key
        self._decoder_obs_key = decoder_obs_key
        self._proprioceptive_obs_size = int(env.proprioceptive_obs_size)

        sample_state = env.reset(jax.random.PRNGKey(0))
        if not isinstance(sample_state.obs, Mapping):
            raise ValueError(
                f"HighLevelWrapper requires dict observations. Got {type(sample_state.obs).__name__}."
            )

        self._policy_obs_size = int(
            jax.flatten_util.ravel_pytree(
                sample_state.obs[policy_obs_key][highlvl_obs_key]
            )[0].shape[0]
        )
        self._value_obs_size = int(
            jax.flatten_util.ravel_pytree(
                sample_state.obs[value_obs_key][highlvl_obs_key]
            )[0].shape[0]
        )

        _, self._dummy_decoder_extras = decoder_inference_fn(
            jp.zeros(latent_size + self._proprioceptive_obs_size)
        )

    def _process_state(self, state: wrapper.mjx_env.State) -> wrapper.mjx_env.State:
        """Process state to extract task obs for high-level policy."""
        state.info["_full_obs"] = state.obs

        policy_obs = jp.nan_to_num(
            jax.flatten_util.ravel_pytree(
                state.obs[self._policy_obs_key][self._highlvl_obs_key]
            )[0]
        )
        value_obs = jp.nan_to_num(
            jax.flatten_util.ravel_pytree(
                state.obs[self._value_obs_key][self._highlvl_obs_key]
            )[0]
        )
        new_obs = {self._policy_obs_key: policy_obs}
        if self._value_obs_key != self._policy_obs_key:
            new_obs[self._value_obs_key] = value_obs
        return state.replace(obs=new_obs)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> wrapper.mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        state.info["decoder_extras"] = self._dummy_decoder_extras
        return self._process_state(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        decoder_obs = jp.nan_to_num(
            jax.flatten_util.ravel_pytree(
                state.info["_full_obs"][self._policy_obs_key][self._decoder_obs_key]
            )[0]
        )
        ctrl, extras = self._decoder_inference_fn(
            jp.concatenate([action, decoder_obs], axis=-1)
        )
        next_state = self.env.step(state, ctrl)
        next_state.info["decoder_extras"] = extras
        return self._process_state(next_state)

    @property
    def action_size(self) -> int:
        return self._latent_size

    @property
    def observation_size(self):
        """Return observation sizes for the high-level policy."""
        sizes = {self._policy_obs_key: self._policy_obs_size}
        if self._value_obs_key != self._policy_obs_key:
            sizes[self._value_obs_key] = self._value_obs_size
        return sizes
