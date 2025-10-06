import jax
import jax.flatten_util
from mujoco_playground import wrapper

class FlattenObsWrapper(wrapper.Wrapper):

    def __init__(self, env: wrapper.mjx_env.MjxEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> wrapper.mjx_env.State:
        state = self.env.reset(rng)
        return self._flatten(state)
    
    def step(self, state: wrapper.mjx_env.State, action: jax.Array) -> wrapper.mjx_env.State:
        state = self.env.step(state, action)
        return self._flatten(state)
    
    def _flatten(self, state: wrapper.mjx_env.State) -> wrapper.mjx_env.State:
        state = state.replace(
            obs =jax.flatten_util.ravel_pytree(state.obs)[0],
            #metrics = self._flatten_metrics(state.metrics),
        )
        return state
    
    def _flatten_metrics(self, metrics: dict) -> dict:
        new_metrics = {}
        def rec(d: dict, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    rec(v, prefix + k + '/')
                else:
                    new_metrics[prefix + k] = v
        rec(metrics)
        return new_metrics
    
    @property
    def observation_size(self) -> int:
        rng_shape = jax.eval_shape(jax.random.key, 0)
        #flat_obs = lambda rng: self._flatten(self.env.reset(rng).obs)
        obs_size = len(jax.eval_shape(self.reset, rng_shape).obs)
        return obs_size