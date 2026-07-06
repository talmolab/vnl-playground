"""Render-only fruitfly env for figure rendering.

Mirrors ``tasks.stick.visualize`` / ``tasks.rodent.visualize``: a thin,
concrete subclass of :class:`FruitflyEnv` with no-op ``reset``/``step``, meant
purely for compiling an ``MjModel`` and rendering static qpos (motion
sequences, montages, videos).

Beyond the bare render env it exposes an ``arena`` switch so figures can pick a
floor-less pure-white skybox (``consts.WHITE_ARENA_XML_PATH`` -- the same
"white aesthetic" as the stick / rodent white arenas) or the checkerboard-floor
``arena.xml`` used for video renders where a ground plane is wanted. The
fly-body helpers themselves (``add_fly`` / ``add_ghost_fly``, both of which take
an RGBA so callers can grade transparency) live on :class:`FruitflyEnv`.
"""

from vnl_playground.tasks.fruitfly import base as fly_base
from vnl_playground.tasks.fruitfly import consts

from mujoco_playground._src import mjx_env

import jax
import jax.numpy as jp


class FlyRender(fly_base.FruitflyEnv):
    """Render-only fruitfly env with a selectable arena background."""

    def __init__(self, arena: str = "white", **kwargs) -> None:
        """
        Args:
            arena: ``"white"`` (floor-less pure-white skybox, the default) or
                ``"grid"`` (the checkerboard-floor arena used for video renders
                where a ground plane is wanted).
            **kwargs: forwarded to :class:`FruitflyEnv` (e.g. ``config_overrides``).
        """
        cfg = fly_base.default_config()
        cfg.arena_xml_path = (
            consts.WHITE_ARENA_XML_PATH if arena == "white" else consts.ARENA_XML_PATH
        )
        super().__init__(cfg, **kwargs)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        reward, done, obs = jp.zeros(3)
        return mjx_env.State(data, obs, reward, done, {}, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return state
