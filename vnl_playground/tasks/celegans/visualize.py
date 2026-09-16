"""Celegans environment for visualization."""

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.celegans import base as celegans_base


class CelegansRender(celegans_base.CelegansEnv):
    """Celegans environment for visualization."""

    def __init__(
        self,
        config: config_dict.ConfigDict = celegans_base.default_config(),
        config_override: dict[str, str | int | list[Any]] | None = None,
    ):
        super().__init__(config, config_override)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment and returns the initial state.

        Args:
            rng: A JAX random key for any stochasticity in the reset process.
        """
        data = mjx_env.init(self.mjx_model)
        reward, done, obs = jp.zeros(3)
        return mjx_env.State(data, obs, reward, done, {}, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Steps the environment forward by one time step.

        Args:
            state: The current state of the environment.
            action: The action to take.

        Returns:
            The next state of the environment.
        """
        n_steps = self._n_steps
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)
        state = state.replace(data=data)
        return state
