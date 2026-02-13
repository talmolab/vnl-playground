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
    top-level keys 'state' and 'privileged_state', each containing 'task_obs'
    and 'proprioception'. Task observations are extracted and exposed with the
    state/privileged_state structure preserved for asymmetric actor-critic
    training. Proprioception is passed to the decoder.

    When ``pass_vision=True``, the wrapper produces vision-only observations
    for the high-level policy: ``{"proprioception": empty[0], "vision": ...}``.
    The agent must derive all task-relevant information from egocentric pixels.
    Body proprioception is routed exclusively to the frozen decoder.

    When ``pass_vision=True`` AND ``pass_task_obs=True``, the wrapper includes
    both vision pixels and the flattened task_obs (keyed as ``"imitation_target"``)
    in the observation dict, giving the high-level network both modalities.

    Args:
        env: The base environment to wrap.
        decoder_inference_fn: Function that maps (latent + proprioception) -> ctrl.
        latent_size: Size of the latent action space.
        obs_key: Top-level observation key to use for decoder (default: 'state').
        highlvl_obs_key: Key for high-level policy observations (default: 'task_obs').
        decoder_obs_key: Key for decoder observations (default: 'proprioception').
        pass_vision: If True, expose only vision to the high-level policy
            (no gap features, no proprioception). Requires env to have a
            ``vision`` key in its observations.
        pass_task_obs: If True (requires ``pass_vision=True``), also include the
            flattened ``highlvl_obs_key`` as ``"imitation_target"`` alongside
            vision in the observation dict.
    """

    def __init__(
        self,
        env: wrapper.mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        obs_key: str = "state",
        highlvl_obs_key: str = "task_obs",
        decoder_obs_key: str = "proprioception",
        pass_vision: bool = False,
        pass_task_obs: bool = False,
    ):
        super().__init__(env)
        if pass_task_obs and not pass_vision:
            raise ValueError(
                "pass_task_obs=True requires pass_vision=True. "
                "Task obs passthrough is only supported in vision mode."
            )
        self._decoder_inference_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._obs_key = obs_key
        self._highlvl_obs_key = highlvl_obs_key
        self._decoder_obs_key = decoder_obs_key
        self._pass_vision = pass_vision
        self._pass_task_obs = pass_task_obs
        self._proprioceptive_obs_size = int(env.proprioceptive_obs_size)

        sample_state = env.reset(jax.random.PRNGKey(0))
        if not isinstance(sample_state.obs, Mapping):
            raise ValueError(
                f"HighLevelWrapper requires dict observations. Got {type(sample_state.obs).__name__}."
            )

        self._state_obs_size = int(
            jax.flatten_util.ravel_pytree(sample_state.obs["state"][highlvl_obs_key])[
                0
            ].shape[0]
        )
        self._privileged_obs_size = int(
            jax.flatten_util.ravel_pytree(
                sample_state.obs["privileged_state"][highlvl_obs_key]
            )[0].shape[0]
        )

        if pass_vision and "vision" not in sample_state.obs.get("state", {}):
            raise ValueError(
                "pass_vision=True requires env observations to contain a 'vision' key "
                "inside 'state'. Use a vision-enabled environment (e.g. RunGapVision)."
            )
        if pass_vision:
            self._vision_shape = sample_state.obs["state"]["vision"].shape

        _, self._dummy_decoder_extras = decoder_inference_fn(
            jp.zeros(latent_size + self._proprioceptive_obs_size)
        )

    def _process_state(self, state: wrapper.mjx_env.State) -> wrapper.mjx_env.State:
        """Process state to extract obs for the high-level policy."""
        # Store full dict obs in info for decoder access
        state.info["_full_obs"] = state.obs

        if self._pass_vision and self._pass_task_obs:
            # Vision + task_obs mode: high-level policy sees pixels AND
            # flattened task observations (e.g. imitation targets).
            # Body proprioception is routed to the frozen decoder in step().
            flat_task_obs = jp.nan_to_num(
                jax.flatten_util.ravel_pytree(
                    state.obs["state"][self._highlvl_obs_key]
                )[0]
            )
            new_obs = {
                "imitation_target": flat_task_obs,
                "proprioception": jp.zeros(0),
                "vision": state.obs["state"]["vision"],
            }
        elif self._pass_vision:
            # Vision-only mode: high-level policy sees ONLY pixels.
            # No gap features, no proprioception — the agent must derive
            # all task-relevant information from the egocentric camera.
            # Body proprioception is routed to the frozen decoder in step().
            new_obs = {
                "proprioception": jp.zeros(0),
                "vision": state.obs["state"]["vision"],
            }
        else:
            # MLP mode: flat obs with state/privileged_state structure
            new_obs = {
                "state": jp.nan_to_num(
                    jax.flatten_util.ravel_pytree(
                        state.obs["state"][self._highlvl_obs_key]
                    )[0]
                ),
                "privileged_state": jp.nan_to_num(
                    jax.flatten_util.ravel_pytree(
                        state.obs["privileged_state"][self._highlvl_obs_key]
                    )[0]
                ),
            }
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
                state.info["_full_obs"][self._obs_key][self._decoder_obs_key]
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
    def observation_size(self) -> dict[str, int]:
        """Return observation sizes for the high-level policy."""
        if self._pass_vision and self._pass_task_obs:
            return {
                "imitation_target": self._state_obs_size,
                "proprioception": 0,
            }
        if self._pass_vision:
            return {"proprioception": 0}
        return {
            "state": self._state_obs_size,
            "privileged_state": self._privileged_obs_size,
        }

    @property
    def vision_shape(self):
        """Shape of the vision observation (H, W, C). Only valid when pass_vision=True."""
        if not self._pass_vision:
            raise AttributeError("vision_shape is only available when pass_vision=True")
        return self._vision_shape
