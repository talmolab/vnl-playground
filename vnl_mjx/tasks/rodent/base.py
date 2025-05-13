"""Base classes for rodent"""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.rodent import consts


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.RODENT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.RODENT_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        mj_model_timestep=0.002,
        solver="cg",
        iterations=4,
        ls_iterations=4,
    )


class RodentEnv(mjx_env.MjxEnv):
    """Base class for rodent environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize the RodentEnv class with only arena

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for the environment.
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Optional overrides for the configuration. Defaults to None.
            compile_spec (bool, optional): Whether to compile the model. Defaults to False.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._arena_xml_path = str(config.arena_xml_path)
        self._spec = mujoco.MjSpec.from_string(config.arena_xml_path.read_text())
        self._compiled = False

    def add_rodent(self, pos=(0, 0, 0.05), quat=(1, 0, 0, 0)) -> None:
        """Adds the rodent model to the environment."""
        rodent = mujoco.MjSpec.from_string(
            epath.Path(self._walker_xml_path).read_text()
        )
        spawn_site = self._spec.worldbody.add_site(
            name="rodent_spawn",
            pos=pos,
            quat=quat,
        )
        spawn_body = spawn_site.attach_body(rodent.worldbody, "", "-rodent")
        spawn_body.add_freejoint()

    def compile(self) -> None:
        """Compiles the model from the mj_spec and put models to mjx"""
        if not self._compiled:
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            # Increase offscreen framebuffer size to render at higher resolutions.
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mj_model.opt.iterations = self._config.iterations
            self._mj_model.opt.ls_iterations = self._config.ls_iterations
            self._mjx_model = mjx.put_model(self._mj_model)
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
