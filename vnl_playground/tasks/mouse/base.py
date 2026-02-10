"""Base classes for mouse (arena-first, add walker later)."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from tqdm import tqdm

from mujoco_playground._src import mjx_env
from vnl_playground.tasks.mouse import consts


def get_assets() -> Dict[str, bytes]:
    """Collect XML + assets into a dict (for bundling/remote)."""
    assets = {}
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    """Default sim + XML config for mouse tasks."""
    return config_dict.create(
        walker_xml_path=consts.MOUSE_XML_PATH,
        arena_xml_path=consts.MOUSE_ARENA_XML_PATH,  # separate empty arena
        ctrl_dt=0.0025,  # physics_steps_per_control_step=2 -> 0.00125*2
        sim_dt=0.00125,  # mj_model_timestep from imitation settings
        solver="cg",  # CG solver as in imitation settings
        iterations=6,
        ls_iterations=6,
        noslip_iterations=0,
        Kp=35.0,
        Kd=0.5,
        episode_length=150,
        mujoco_impl="jax",
    )


class MouseBaseEnv(mjx_env.MjxEnv):
    """Arena-first base for mouse environments with add_mouse() then compile."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize with arena-only MjSpec; add mouse(s) later via add_mouse().

        Args:
            config: Configuration dictionary (expects walker_xml_path, arena_xml_path).
            config_overrides: Optional overrides for fields in `config`.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._arena_xml_path = str(config.arena_xml_path)

        # Build an arena-only spec; walker gets attached on demand.
        self._spec = mujoco.MjSpec.from_file(self._arena_xml_path)
        self._compiled = False

    def add_mouse(
        self,
        freejoint: bool = False,
        pos: Union[tuple[float, float, float], list[float]] = (0.0, 0.0, 0.02),
        suffix: str = "-mouse",
        rgba: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Attach a mouse model to the arena at the given position.

        Args:
            freejoint: If True, add a freejoint on the attached root body.
            pos: Spawn position (x, y, z) in arena frame.
            suffix: Name suffix to avoid collisions for multiple mice.
            rgba: Optional per-geom RGBA override for the attached mouse.

        Returns:
            None
        """
        mouse_spec = mujoco.MjSpec.from_file(self._walker_xml_path)

        # Attach using a frame (like rodent) instead of site for better positioning
        spawn_frame = self._spec.worldbody.add_frame(
            pos=list(pos),
            quat=[1, 0, 0, 0],
        )
        # Attach the clavicle body (root of arm hierarchy), not entire worldbody
        body = spawn_frame.attach_body(mouse_spec.body("clavicle"), "", suffix)
        if freejoint:
            body.add_freejoint()
        if rgba is not None:
            for g in getattr(body, "geom", []):
                g.rgba = list(rgba)

    def add_ghost_mouse(
        self,
        pos: Union[tuple[float, float, float], list[float]] = (0.2, 0.0, 0.02),
        suffix: str = "-ghost",
        ghost_rgba: tuple[float, float, float, float] = (
            65 / 256,
            181 / 256,
            225 / 256,
            0.54,
        ),
        no_collision: bool = True,
    ) -> None:
        """
        Attach a ghost/reference mouse (no freejoint, translucent, non-colliding).

        Args:
            pos: Spawn position (x, y, z) in arena frame.
            suffix: Name suffix to avoid collisions for multiple ghosts.
            ghost_rgba: RGBA to tint all geoms of the ghost mouse.
            no_collision: If True, set contype=conaffinity=0 on all geoms.

        Returns:
            None
        """
        mouse_spec = mujoco.MjSpec.from_file(self._walker_xml_path)

        # Attach using a frame for consistent positioning
        spawn_frame = self._spec.worldbody.add_frame(
            pos=list(pos),
            quat=[1, 0, 0, 0],
        )
        # Attach the clavicle body (root of arm hierarchy), not entire worldbody
        body = spawn_frame.attach_body(mouse_spec.body("clavicle"), "", suffix)
        # Intentionally NO freejoint: kinematically tied through the attached tree.

        for g in getattr(body, "geom", []):
            g.rgba = list(ghost_rgba)
            if no_collision:
                g.contype = 0
                g.conaffinity = 0

    def add_multiple_mice(
        self,
        n: int,
        spacing: float = 0.05,
        base_pos: tuple[float, float, float] = (0.0, 0.0, 0.02),
        freejoint: bool = True,
    ) -> None:
        """
        Convenience: spawn `n` mice along +Y with `tqdm` progress.

        Args:
            n: Number of mice to spawn.
            spacing: Y-axis spacing between mice.
            base_pos: Base position for the first mouse (x, y, z).
            freejoint: Whether each spawned mouse gets a freejoint.

        Returns:
            None
        """
        x0, y0, z0 = base_pos
        for i in tqdm(range(n), desc="Spawning mice"):
            self.add_mouse(
                freejoint=freejoint,
                pos=(x0, y0 + i * spacing, z0),
                suffix=f"-{i}",
            )

    def compile(self) -> None:
        """
        Compile the current spec into mjModel/mjx.Model.

        Args:
            None

        Returns:
            None
        """
        if not self._compiled:
            self._mj_model = self._spec.compile()

            # Set timestep
            self._mj_model.opt.timestep = self._config.sim_dt

            # Set solver type and iterations (critical for performance!)
            self._mj_model.opt.solver = {
                "cg": mujoco.mjtSolver.mjSOL_CG,
                "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            }[self._config.solver.lower()]
            self._mj_model.opt.iterations = self._config.iterations
            self._mj_model.opt.ls_iterations = self._config.ls_iterations
            self._mj_model.opt.noslip_iterations = self._config.noslip_iterations

            # High-res offscreen buffer for nice renders
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160

            # Use configured implementation (warp/jax)
            self._mjx_model = mjx.put_model(
                self._mj_model, impl=self._config.mujoco_impl
            )
            self._compiled = True

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def xml_path(self) -> str:
        return self._walker_xml_path

    @property
    def walker_xml_path(self) -> str:
        return self._walker_xml_path

    @property
    def arena_xml_path(self) -> str:
        return self._arena_xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def dt(self) -> float:
        """Control timestep (ctrl_dt)."""
        return self._config.ctrl_dt
