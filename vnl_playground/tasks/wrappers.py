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

    @property
    def unwrapped(self) -> mjx_env.MjxEnv:
        return self

    @property
    def _mjx_model(self):
        return self.env._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self.env._mjx_model = value


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

    @property
    def unwrapped(self) -> mjx_env.MjxEnv:
        return self

    @property
    def _mjx_model(self):
        return self.env._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self.env._mjx_model = value


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
        lowlvl_obs_key: Key for decoder observations (default: 'proprioception').
    """

    def __init__(
        self,
        env: wrapper.mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        policy_obs_key: str = "state",
        value_obs_key: str = "state",
        highlvl_obs_key: str = "task_obs",
        lowlvl_obs_key: str = "proprioception",
    ):
        super().__init__(env)
        self._decoder_inference_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._policy_obs_key = policy_obs_key
        self._value_obs_key = value_obs_key
        self._highlvl_obs_key = highlvl_obs_key
        self._lowlvl_obs_key = lowlvl_obs_key
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
                state.info["_full_obs"][self._policy_obs_key][self._lowlvl_obs_key]
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

    @property
    def unwrapped(self) -> mjx_env.MjxEnv:
        return self

    @property
    def _mjx_model(self):
        return self.env._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self.env._mjx_model = value


def _default_reset_hidden_on_done(hidden: Any, done: jax.Array) -> Any:
    """Zero hidden state elements where ``done`` is true.

    Works for arbitrary pytrees of arrays, including tuples/lists used by LSTM
    carry states and stacked recurrent layers.
    """
    done = jp.asarray(done, dtype=jp.bool_)
    done_expanded = done[..., None]
    return jax.tree_util.tree_map(
        lambda x: jp.where(done_expanded, jp.zeros_like(x), x),
        hidden,
    )


class RecurrentHighLevelWrapper(wrapper.Wrapper):
    """High-level wrapper that drives a recurrent low-level decoder.

    Similar to :class:`HighLevelWrapper`, but the decoder receives and returns
    recurrent hidden state each step.

    Args:
        env: The base environment to wrap.
        decoder_step_fn: Callable mapping
            ``(latent_plus_proprio, hidden) -> (ctrl, extras, new_hidden)``.
        init_decoder_hidden_fn: Callable mapping ``batch_size -> hidden``.
        latent_size: Size of the high-level latent action space.
        policy_obs_key: Top-level obs key for actor observations.
        value_obs_key: Top-level obs key for critic observations.
        highlvl_obs_key: Observation key exposed to the high-level policy.
        lowlvl_obs_key: Observation key consumed by the low-level decoder.
        reset_decoder_hidden_fn: Optional custom hidden-reset function with
            signature ``(hidden, done) -> hidden``. If omitted, hidden state is
            zeroed where ``done`` is true.
    """

    def __init__(
        self,
        env: wrapper.mjx_env.MjxEnv,
        decoder_step_fn: Callable,
        init_decoder_hidden_fn: Callable,
        latent_size: int,
        policy_obs_key: str = "state",
        value_obs_key: str = "state",
        highlvl_obs_key: str = "task_obs",
        lowlvl_obs_key: str = "proprioception",
        reset_decoder_hidden_fn: Callable | None = None,
    ):
        super().__init__(env)
        self._decoder_step_fn = decoder_step_fn
        self._init_decoder_hidden_fn = init_decoder_hidden_fn
        self._reset_decoder_hidden_fn = (
            reset_decoder_hidden_fn or _default_reset_hidden_on_done
        )
        self._latent_size = latent_size
        self._policy_obs_key = policy_obs_key
        self._value_obs_key = value_obs_key
        self._highlvl_obs_key = highlvl_obs_key
        self._lowlvl_obs_key = lowlvl_obs_key
        self._proprioceptive_obs_size = int(env.proprioceptive_obs_size)

        sample_state = env.reset(jax.random.PRNGKey(0))
        if not isinstance(sample_state.obs, Mapping):
            raise ValueError(
                "RecurrentHighLevelWrapper requires dict observations. "
                f"Got {type(sample_state.obs).__name__}."
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

        # Lock decoder extras tree structure using one dummy recurrent forward pass.
        dummy_hidden = self._init_decoder_hidden_fn(1)
        dummy_input = jp.zeros(latent_size + self._proprioceptive_obs_size)
        _, self._dummy_decoder_extras, _ = self._decoder_step_fn(
            dummy_input,
            dummy_hidden,
        )
        self._decoder_extras_treedef = jax.tree_util.tree_structure(
            self._dummy_decoder_extras
        )

    @staticmethod
    def _infer_batch_size(obs_leaf: jax.Array) -> int:
        """Infer batch size from a flattened observation leaf."""
        return int(obs_leaf.shape[0]) if obs_leaf.ndim >= 2 else 1

    @staticmethod
    def _flatten_obs_tree(obs_tree: Any) -> jax.Array:
        """Flatten an observation tree, preserving batch dimension when present."""
        leaves = jax.tree_util.tree_leaves(obs_tree)
        if not leaves:
            raise ValueError("Cannot flatten empty observation tree.")

        first_leaf = leaves[0]
        if first_leaf.ndim >= 2:
            batch_size = first_leaf.shape[0]
            flat_leaves = [
                jp.nan_to_num(leaf).reshape((batch_size, -1)) for leaf in leaves
            ]
            return jp.concatenate(flat_leaves, axis=-1)

        flat_leaves = [jp.nan_to_num(leaf).reshape((-1,)) for leaf in leaves]
        return jp.concatenate(flat_leaves, axis=-1)

    def _process_state(self, state: wrapper.mjx_env.State) -> wrapper.mjx_env.State:
        """Flatten high-level observations and cache full observation in info."""
        info = dict(state.info)
        info["_full_obs"] = state.obs

        policy_obs = self._flatten_obs_tree(
            state.obs[self._policy_obs_key][self._highlvl_obs_key]
        )
        value_obs = self._flatten_obs_tree(
            state.obs[self._value_obs_key][self._highlvl_obs_key]
        )

        new_obs = {self._policy_obs_key: policy_obs}
        if self._value_obs_key != self._policy_obs_key:
            new_obs[self._value_obs_key] = value_obs

        return state.replace(obs=new_obs, info=info)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> wrapper.mjx_env.State:
        state = self.env.reset(rng, **kwargs)
        state = self._process_state(state)

        policy_leaf = state.obs[self._policy_obs_key]
        batch_size = self._infer_batch_size(policy_leaf)
        decoder_hidden = self._init_decoder_hidden_fn(batch_size)

        if batch_size == 1:
            dummy_input = jp.zeros(self._latent_size + self._proprioceptive_obs_size)
        else:
            dummy_input = jp.zeros(
                (batch_size, self._latent_size + self._proprioceptive_obs_size)
            )

        _, decoder_extras, _ = self._decoder_step_fn(dummy_input, decoder_hidden)
        if (
            jax.tree_util.tree_structure(decoder_extras)
            != self._decoder_extras_treedef
        ):
            raise ValueError(
                "decoder_step_fn returned extras with inconsistent pytree "
                "structure between init and reset."
            )

        info = dict(state.info)
        info["decoder_hidden"] = decoder_hidden
        info["decoder_extras"] = decoder_extras
        return state.replace(info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if "decoder_hidden" not in state.info:
            raise KeyError(
                "Missing state.info['decoder_hidden']; call reset() before step()."
            )
        if "_full_obs" not in state.info:
            raise KeyError(
                "Missing state.info['_full_obs']; wrapper state cache was not initialized."
            )

        decoder_obs = self._flatten_obs_tree(
            state.info["_full_obs"][self._policy_obs_key][self._lowlvl_obs_key]
        )
        decoder_input = jp.concatenate([action, decoder_obs], axis=-1)

        ctrl, extras, new_hidden = self._decoder_step_fn(
            decoder_input,
            state.info["decoder_hidden"],
        )

        next_state = self.env.step(state, ctrl)
        next_hidden = self._reset_decoder_hidden_fn(new_hidden, next_state.done)

        next_state = self._process_state(next_state)
        info = dict(next_state.info)
        info["decoder_hidden"] = next_hidden
        info["decoder_extras"] = extras
        return next_state.replace(info=info)

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
