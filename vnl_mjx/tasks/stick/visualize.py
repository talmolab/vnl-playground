from vnl_mjx.tasks.stick import base as stick_base
from vnl_mjx.tasks.stick import consts

from mujoco_playground._src import mjx_env

import jax
import jax.numpy as jp


class StickRender(stick_base.StickEnv):

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        reward, done, obs = jp.zeros(3)
        return mjx_env.State(data, obs, reward, done, {}, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return state
