"""Render-only stick-bug env for figure rendering.

Mirrors ``tasks.rodent.visualize`` / ``tasks.fruitfly.visualize``: a thin,
concrete subclass of :class:`StickBugEnv` with no-op ``reset``/``step``, meant
purely for compiling an ``MjModel`` and rendering static qpos (motion
sequences, montages, videos).

Beyond the bare render env it adds two figure helpers the training envs don't
need:

* a floor-less pure-white arena by default (``consts.WHITE_ARENA_XML_PATH``),
  the same "white aesthetic" as ``tasks.rodent.visualize``'s white arena;
* :meth:`StickRender.add_overlay_ghost`, which attaches a transparent reference
  "ghost" at the same pose as the rollout body, with collisions recursively
  disabled so the mesh ghost never interacts with the scene -- the
  ``tasks.fruitfly`` mesh-ghost trick.
"""

import jax
import jax.numpy as jp
import mujoco

from mujoco_playground._src import mjx_env

from vnl_playground.tasks.stick import consts
from vnl_playground.tasks.stick.base import StickBugEnv, default_config
from vnl_playground.tasks.utils import _recolour_tree, scale_spec


class StickRender(StickBugEnv):
    """Render-only stick env with a white background and overlay-ghost helper."""

    def __init__(self, model: str = "mesh", arena: str = "white", **kwargs) -> None:
        """
        Args:
            model: ``"mesh"`` (Sungaya triangle-mesh model, 48 qpos) or
                ``"box"`` (legacy primitive-geom model, 45 qpos).
            arena: ``"white"`` (floor-less pure-white skybox, the default) or
                ``"grid"`` (the checkerboard-floor arena used for video renders
                where a ground plane is wanted).
        """
        cfg = default_config()
        cfg.walker_xml_path = (
            consts.STICK_XML_PATH if model == "mesh" else consts.STICK_BOX_XML_PATH
        )
        cfg.arena_xml_path = (
            consts.WHITE_ARENA_XML_PATH if arena == "white" else consts.ARENA_XML_PATH
        )
        super().__init__(cfg, **kwargs)
        self.model = model
        self._suffix = ""  # add_stick sets this; default for ghost-only envs

    def add_stick(self, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=None, suffix="-stick"):
        """Attach a stick body for rendering (no floor-foot collision pairs).

        ``StickBugEnv.add_stick`` adds ``<pair>`` contacts that reference the
        arena "floor" geom and the mesh-only claw geoms. Those are irrelevant to
        static rendering and absent from the white arena / box model, so we skip
        them here.
        """
        stick = mujoco.MjSpec.from_file(self._walker_xml_path)
        if rgba is not None:
            for body in stick.worldbody.bodies:
                _recolour_tree(body, rgba=rgba)
        frame = self._spec.worldbody.add_frame(pos=pos, quat=quat)
        frame.attach_body(stick.body("reference_base"), "", suffix=suffix)
        self._suffix = suffix

    def add_overlay_ghost(
        self,
        pos=(0, 0, 0),
        ghost_rgba=(1.0, 1.0, 1.0, 0.2),
        rescale_factor: float = 1.0,
        suffix="-ghost",
    ):
        """Attach a transparent reference ghost overlaid at ``pos``.

        Recolours the whole body tree and recursively disables collisions on
        every ghost geom (body tree + worldbody) so the mesh ghost cannot
        interact with the scene -- the ``tasks.fruitfly`` mesh-ghost pattern.

        Call this *after* :meth:`add_stick`; ``qpos`` is then ordered
        ``[rollout..., ghost...]`` in body-attach order, so a montage that adds
        all rollout bodies first and all ghosts second can set
        ``data.qpos = concat(rollout_qpos, ghost_qpos)``.
        """
        ghost = mujoco.MjSpec.from_file(self._walker_xml_path)
        if rescale_factor != 1.0:
            ghost = scale_spec(ghost, rescale_factor, root_body="reference_base")

        def _disable_collisions(body):
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0
            for child in body.bodies:
                _disable_collisions(child)

        for geom in ghost.worldbody.geoms:
            geom.contype = 0
            geom.conaffinity = 0
        for body in ghost.worldbody.bodies:
            _recolour_tree(body, rgba=ghost_rgba)
            _disable_collisions(body)

        frame = self._spec.worldbody.add_frame(pos=pos, quat=[1, 0, 0, 0])
        frame.attach_body(ghost.body("reference_base"), "", suffix=suffix)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        reward, done, obs = jp.zeros(3)
        return mjx_env.State(data, obs, reward, done, {}, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return state
