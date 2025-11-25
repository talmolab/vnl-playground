from mujoco_playground._src import wrapper, mjx_env
from jax.numpy.flatten_util import ravel_pytree
import jax

class FlattenObservation(wrapper.Wrapper):
    def __init__(self, env: mjx_env.MjxEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array, start_frame: int = 0) -> mjx_env.State:
        state = self.env.reset(rng, start_frame)
        state = state.replace(obs=ravel_pytree(state.obs)[0])
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        state = state.replace(obs=ravel_pytree(state.obs)[0])
        return state