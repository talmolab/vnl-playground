from vnl_playground.tasks.fruitfly import base as fly_base
from vnl_playground.tasks.fruitfly import consts

from mujoco_playground._src import mjx_env

import jax
import jax.numpy as jp


class FlyRender(fly_base.FruitflyEnv):

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        reward, done, obs = jp.zeros(3)
        return mjx_env.State(data, obs, reward, done, {}, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return state
