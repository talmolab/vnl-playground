"""Base classes for C. Elegans"""

from typing import Any, Dict, Optional, Union

from etils import epath
import logging
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.celegans import consts
from vnl_mjx.tasks.utils import _scale_body_tree, _recolour_tree, dm_scale_spec


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.CELEGANS_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.CELEGANS_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.CELEGANS_XML_PATH,
        arena_xml_path=consts.WHITE_ARENA_XML_PATH,
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="cg",
        iterations=4,
        ls_iterations=4,
        noslip_iterations=0,
    )

END_EFFECTORS = ["lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"]
TOUCH_SENSORS = ["palm_L", "palm_R", "sole_L", "sole_R"]

class CelegansEnv(mjx_env.MjxEnv):
    """Base class for C. Elegans environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize the CelegansEnv class with only arena

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for the environment.
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Optional overrides for the configuration. Defaults to None.
            compile_spec (bool, optional): Whether to compile the model. Defaults to False.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._arena_xml_path = str(config.arena_xml_path)
        self._spec = mujoco.MjSpec.from_file(self._arena_xml_path)
        self._compiled = False

    def add_worm(
        self,
        torque_actuators: bool,
        rescale_factor: float = 1.0,
        pos: tuple[float, float, float] = (0, 0, 0.05),
        quat: tuple[float, float, float, float] = (1, 0, 0, 0),
        rgba: Optional[tuple[float, float, float, float]] = None,
        suffix: str = "-worm",
    ) -> None:
        """Adds the c. elegans model to the environment.
        
        Args:
            torque_actuators: Whether to convert motors to torque-mode actuators.
            rescale_factor: Factor to rescale the c. elegans body. Defaults to 1.0.
            pos: Position (x, y, z) to spawn the c. elegans. Defaults to (0, 0, 0.05).
            quat: Quaternion (w, x, y, z) for c. elegans orientation. Defaults to (1, 0, 0, 0).
            rgba: RGBA color values (red, green, blue, alpha) for recoloring the body.
                If None, no recoloring is applied. Defaults to None.
            suffix: Suffix to append to body names. Defaults to "-worm".
        """
        worm = mujoco.MjSpec.from_file(self._walker_xml_path)

        # a) Convert motors to torque‑mode if requested
        if torque_actuators and hasattr(worm, "actuator"):
            logging.info("Converting to torque actuators")
            for actuator in worm.actuators:  # type: ignore[attr-defined]
                # Set gain to max force; remove bias terms if present
                if actuator.forcerange.size >= 2:
                    actuator.gainprm[0] = actuator.forcerange[1]
                # reset custom bias terms
                actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
                actuator.biasprm = np.zeros((10, 1))

        if rescale_factor != 1.0:
            logging.info(f"Rescaling body tree with scale factor {rescale_factor}")
            worm = dm_scale_spec(worm, rescale_factor)

        # Recolor the body if rgba is specified
        if rgba is not None:
            for body in worm.worldbody.bodies:
                _recolour_tree(body, rgba=rgba)

        spawn_site = self._spec.worldbody.add_frame(
            pos=pos,
            quat=quat,
        )
        spawn_body = spawn_site.attach_body(worm.worldbody, "torso1_body", suffix=suffix)
        self._suffix = suffix
        spawn_body.add_freejoint()

    def add_ghost_worm(
        self,
        rescale_factor: float = 1.0,
        pos=(0, 0, 0.05),
        ghost_rgba=(0.8, 0.8, 0.8, 0.3),
        suffix="-ghost",
    ):
        """Adds a ghost worm model to the environment."""
        walker_spec = mujoco.MjSpec.from_string(
            epath.Path(self._walker_xml_path).read_text()
        )
        # Scale and recolor the ghost body
        for body in walker_spec.worldbody.bodies:
            _scale_body_tree(body, rescale_factor)
            _recolour_tree(body, rgba=ghost_rgba)
        # Attach as ghost at the offset frame
        frame = self._spec.worldbody.add_frame(pos=pos, quat=[1, 0, 0, 0])
        spawn_body = frame.attach_body(walker_spec.body("torso1_body"), "", suffix=suffix)
        spawn_body.add_freejoint()

    def compile(self, forced=False) -> None:
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
            self._mjx_model = mjx.put_model(self._mj_model)
            self._compiled = True

    def _get_appendages_pos(self, data: mjx.Data) -> jp.ndarray:
        """Get appendages positions from the environment."""
        torso = data.bind(self.mjx_model, self._spec.body("torso-worm"))
        positions = jp.vstack(
            [
                data.bind(self.mjx_model, self._spec.body(f"{name}{self._suffix}")).xpos
                for name in END_EFFECTORS
            ]
        )
        # get relative pos in egocentric frame
        egocentric_pos = jp.dot(positions - torso.xpos, torso.xmat)
        return egocentric_pos.flatten()

    def _get_proprioception(self, data: mjx.Data) -> jp.ndarray:
        """Get proprioception data from the environment."""
        qpos = data.qpos[7:] # skip the root joint
        qvel = data.qvel[6:] # skip the root joint velocity
        actuator_ctrl = data.qfrc_actuator
        _, body_height, _ = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).xpos
        world_zaxis = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}")).xmat.flatten()[6:]
        appendages_pos = self._get_appendages_pos(data)
        proprioception = jp.concatenate(
            [
                qpos,
                qvel,
                actuator_ctrl,
                jp.array([body_height]),
                world_zaxis,
                appendages_pos,
            ]
        )
        return proprioception

    def _get_kinematic_sensors(self, data: mjx.Data) -> jp.ndarray:
        """Get kinematic sensors data from the environment."""
        accelerometer = data.bind(self.mjx_model, self._spec.sensor("accelerometer-worm")).sensordata
        velocimeter = data.bind(self.mjx_model, self._spec.sensor("velocimeter-worm")).sensordata
        gyro = data.bind(self.mjx_model, self._spec.sensor("gyro-worm")).sensordata
        sensors = jp.concatenate(
            [
                accelerometer,
                velocimeter,
                gyro,
            ]
        )
        return sensors

    def _get_touch_sensors(self, data: mjx.Data) -> jp.ndarray:
        """Get touch sensors data from the environment."""
        touches = [data.bind(self.mjx_model, self._spec.sensor(f"{name}{self._suffix}")).sensordata for name in TOUCH_SENSORS]
        return jp.array(touches)

    def _get_origin(self, data: mjx.Data) -> jp.ndarray:
        """Get origin position in the torso frame."""
        torso = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        torso_frame = torso.xmat
        torso_pos = torso.xpos
        return jp.dot(-torso_pos, torso_frame)

    def _get_egocentric_camera(self, data: mjx.Data):
        """Get egocentric camera data from the environment."""
        raise NotImplementedError(
            "Egocentric camera is not implemented for this environment."
        )

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
