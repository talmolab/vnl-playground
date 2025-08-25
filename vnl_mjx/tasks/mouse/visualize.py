from vnl_mjx.tasks.mouse.mouse_reach import MouseEnv
from mujoco_playground._src import mjx_env
import jax
import jax.numpy as jp


class MouseRender(MouseEnv):
    """Simple render-only wrapper; uses base reset/step unless you need custom."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """
        Minimal reset that keeps the pipeline valid for viewer-only cases.

        Args:
            rng: JAX key.

        Returns:
            mjx_env.State with zeros and empty info (no targets).
        """
        data = mjx_env.init(self.mjx_model)
        reward, done, obs = jp.zeros(3, dtype=jp.float32)
        return mjx_env.State(data, obs, reward, done, {}, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """No-op step for static rendering."""
        return state
