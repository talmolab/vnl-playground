"""Base classes for fruitfly"""

from typing import Any, Dict, Optional, Union

from etils import epath
import logging
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.fruitfly import consts
from vnl_mjx.tasks.utils import _scale_body_tree, _recolour_tree, dm_scale_spec


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.FRUITFLY_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.FRUITFLY_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.FRUITFLY_XML_PATH,
        arena_xml_path=consts.WHITE_ARENA_XML_PATH,
        sim_dt=0.001,
        ctrl_dt=0.002,
        solver="cg",
        iterations=4,
        ls_iterations=4,
        noslip_iterations=0,
    )


class FruitflyEnv(mjx_env.MjxEnv):
    """Base class for fruitfly environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize the FruitflyEnv class with only arena

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for the environment.
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Optional overrides for the configuration. Defaults to None.
            compile_spec (bool, optional): Whether to compile the model. Defaults to False.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._arena_xml_path = str(config.arena_xml_path)
        self._spec = mujoco.MjSpec.from_file(str(config.arena_xml_path))
        self._compiled = False

    def add_fly(
        self,
        torque_actuators: bool,
        rescale_factor: float = 1.0,
        pos: tuple[float, float, float] = (0, 0, 0.05),
        quat: tuple[float, float, float, float] = (1, 0, 0, 0),
        rgba: Optional[tuple[float, float, float, float]] = None,
        suffix: str = "-fly",
    ) -> None:
        """Adds the fly model to the environment.

        Args:
            torque_actuators: Whether to convert motors to torque-mode actuators.
            rescale_factor: Factor to rescale the fly body. Defaults to 1.0.
            pos: Position (x, y, z) to spawn the fly. Defaults to (0, 0, 0.05).
            quat: Quaternion (w, x, y, z) for fly orientation. Defaults to (1, 0, 0, 0).
            rgba: RGBA color values (red, green, blue, alpha) for recoloring the body.
                If None, no recoloring is applied. Defaults to None.
            suffix: Suffix to append to body names. Defaults to "-fly".
        """
        fly = mujoco.MjSpec.from_file(self._walker_xml_path)

        # a) Convert motors to torque‑mode if requested
        if torque_actuators and hasattr(fly, "actuator"):
            logging.info("Converting to torque actuators")
            for actuator in fly.actuators:  # type: ignore[attr-defined]
                # Set gain to max force; remove bias terms if present
                if actuator.forcerange.size >= 2:
                    actuator.gainprm[0] = actuator.forcerange[1]
                # reset custom bias terms
                actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
                actuator.biasprm = np.zeros((10, 1))

        if rescale_factor != 1.0:
            logging.info(f"Rescaling body tree with scale factor {rescale_factor}")
            fly = dm_scale_spec(fly, rescale_factor)

        # Recolor the body if rgba is specified
        if rgba is not None:
            for body in fly.worldbody.bodies:
                _recolour_tree(body, rgba=rgba)

        spawn_frame = self._spec.worldbody.add_frame(
            pos=pos,
            quat=quat,
        )

        spawn_body = spawn_frame.attach_body(fly.body("thorax"), "", suffix=suffix)
        self._suffix = suffix

    def add_ghost_fly(
        self,
        rescale_factor: float = 1.0,
        pos=(0, 0, 0.05),
        ghost_rgba=(0.8, 0.8, 0.8, 0.3),
        suffix="-ghost",
    ):
        """Adds a ghost fly model to the environment."""
        fly_spec = mujoco.MjSpec.from_file(self._walker_xml_path)
        # Scale and recolor the ghost body
        for body in fly_spec.worldbody.bodies:
            _scale_body_tree(body, rescale_factor)
            _recolour_tree(body, rgba=ghost_rgba)
        # Attach as ghost at the offset frame
        spawn_frame = self._spec.worldbody.add_frame(pos=pos, quat=[1, 0, 0, 0])
        spawn_body = spawn_frame.attach_body(fly_spec.body("thorax"), "", suffix=suffix)

    def compile(self, forced=False, mjx_model=False) -> None:
        """Compiles the model from the mj_spec and put models to mjx"""
        if not self._compiled or forced:
            self._spec.option.noslip_iterations = self._config.noslip_iterations
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            # Increase offscreen framebuffer size to render at higher resolutions.
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mj_model.opt.iterations = self._config.iterations
            self._mj_model.opt.ls_iterations = self._config.ls_iterations
            if mjx_model:
                self._mjx_model = mjx.put_model(self._mj_model)
        self._compiled = True

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

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
