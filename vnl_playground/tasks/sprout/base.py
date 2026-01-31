"""Base classes for Fauna Robotics Sprout humanoid.

Follows the same pattern as the rodent walker (rodent/base.py),
adapted for the Sprout humanoid's body structure and sensors.
"""

from typing import Any, Dict, Optional, Union, Mapping
import collections

from etils import epath
import logging
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_playground.tasks.sprout import consts


def get_assets() -> Dict[str, bytes]:
    """Load all XML and mesh assets for the Sprout model."""
    assets = {}
    mjx_env.update_assets(assets, consts.SPROUT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.SPROUT_PATH / "xmls" / "meshes" / "collision")
    mjx_env.update_assets(assets, consts.SPROUT_PATH / "xmls" / "meshes" / "visual")
    mjx_env.update_assets(assets, consts.SPROUT_PATH / "xmls" / "textures")
    return assets


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for Sprout environments."""
    return config_dict.create(
        walker_xml_path=consts.SPROUT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="newton",
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        mujoco_impl="jax",
    )


class SproutEnv(mjx_env.MjxEnv):
    """Base class for Sprout humanoid environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the SproutEnv class with arena.

        Args:
            config: Configuration dictionary for the environment.
            config_overrides: Optional overrides for the configuration.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._arena_xml_path = str(config.arena_xml_path)
        self._spec = mujoco.MjSpec.from_file(self._arena_xml_path)
        self._compiled = False

    def add_sprout(
        self,
        torque_actuators: bool,
        pos: tuple[float, float, float] = (0, 0, 0),
        quat: tuple[float, float, float, float] = (1, 0, 0, 0),
        suffix: str = "-sprout",
    ) -> None:
        """Adds the Sprout model to the environment.

        Args:
            torque_actuators: Whether to convert motors to torque-mode actuators.
            pos: Position (x, y, z) to spawn the robot.
            quat: Quaternion (w, x, y, z) for orientation.
            suffix: Suffix to append to body names.
        """
        sprout = mujoco.MjSpec.from_file(self._walker_xml_path)

        # Convert motors to torque-mode if requested
        if torque_actuators and hasattr(sprout, "actuator"):
            logging.info("Converting to torque actuators")
            for actuator in sprout.actuators:
                # Set gain to effort limit for the corresponding motor group
                joint_name = actuator.name
                effort_limit = self._get_effort_limit(joint_name)
                actuator.gainprm[0] = effort_limit
                actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
                actuator.biasprm = np.zeros((10, 1))

        spawn_site = self._spec.worldbody.add_frame(
            pos=pos,
            quat=quat,
        )
        spawn_body = spawn_site.attach_body(
            sprout.body("torso_link"), "", suffix=suffix
        )
        self._suffix = suffix
        spawn_body.add_freejoint(name="root")

    def _get_effort_limit(self, joint_name: str) -> float:
        """Get the effort limit for a joint from the motor parameters."""
        for motor_type, params in consts.MOTOR_PARAMETERS.items():
            if joint_name in params["joints"]:
                return params["effort_limit"]
        # Default fallback
        return 30.0

    def compile(self, forced=False) -> None:
        """Compiles the model from the mj_spec and puts model to mjx."""
        if not self._compiled or forced:
            self._spec.option.noslip_iterations = self._config.noslip_iterations
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            # Increase offscreen framebuffer size for higher resolution rendering
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mj_model.opt.iterations = self._config.iterations
            self._mj_model.opt.ls_iterations = self._config.ls_iterations
            self._mj_model.opt.solver = {
                "cg": mujoco.mjtSolver.mjSOL_CG,
                "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            }[self._config.solver.lower()]
            self._mjx_model = mjx.put_model(
                self._mj_model, impl=self._config.mujoco_impl
            )
            self._compiled = True

    def _get_appendages_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[dict[str, jp.ndarray], jp.ndarray]:
        """Get egocentric position of end effectors."""
        torso = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        appendages_pos = collections.OrderedDict()
        for appendage_name in consts.END_EFFECTORS:
            global_xpos = data.bind(
                self.mjx_model,
                self._spec.body(f"{appendage_name}{self._suffix}"),
            ).xpos
            egocentric_xpos = jp.dot(global_xpos - torso.xpos, torso.xmat)
            appendages_pos[appendage_name] = egocentric_xpos
        if flatten:
            appendages_pos, _ = jax.flatten_util.ravel_pytree(appendages_pos)
        return appendages_pos

    def _get_bodies_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[dict[str, jp.ndarray], jp.ndarray]:
        """Get global positions of body parts."""
        bodies_pos = collections.OrderedDict()
        for body_name in consts.BODIES:
            global_xpos = data.bind(
                self.mjx_model,
                self._spec.body(f"{body_name}{self._suffix}"),
            ).xpos
            bodies_pos[body_name] = global_xpos
        if flatten:
            bodies_pos, _ = jax.flatten_util.ravel_pytree(bodies_pos)
        return bodies_pos

    def _get_joint_angles(self, data: mjx.Data) -> jp.ndarray:
        return data.qpos[7:]

    def _get_joint_ang_vels(self, data: mjx.Data) -> jp.ndarray:
        return data.qvel[6:]

    def _get_actuator_ctrl(self, data: mjx.Data) -> jp.ndarray:
        return data.qfrc_actuator

    def _get_body_height(self, data: mjx.Data) -> jp.ndarray:
        torso_pos = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        ).xpos
        return torso_pos[2]

    def _get_world_zaxis(self, data: mjx.Data) -> jp.ndarray:
        return self.root_body(data).xmat.flatten()[6:]

    def _get_proprioception(
        self, data: mjx.Data, info: Mapping[str, Any], flatten: bool = True
    ) -> Union[jp.ndarray, Mapping[str, jp.ndarray]]:
        """Get proprioception data from the environment."""
        proprioception = collections.OrderedDict(
            joint_angles=self._get_joint_angles(data),
            joint_ang_vels=self._get_joint_ang_vels(data),
            actuator_ctrl=self._get_actuator_ctrl(data),
            body_height=self._get_body_height(data).reshape(1),
            world_zaxis=self._get_world_zaxis(data),
            appendages_pos=self._get_appendages_pos(data, flatten=flatten),
            kinematic_sensors=self._get_kinematic_sensors(data, flatten=flatten),
            prev_action=info["prev_action"],
        )
        if flatten:
            proprioception, _ = jax.flatten_util.ravel_pytree(proprioception)
        return proprioception

    def _get_kinematic_sensors(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[Mapping[str, jp.ndarray], jp.ndarray]:
        """Get kinematic sensors data from the Sprout's IMU.

        Uses the torso_link_site sensors defined in the MJCF:
        - torso_link_site_pos (framepos)
        - torso_link_site_quat (framequat)
        - torso_link_site_linvel (framelinvel)
        - torso_link_site_angvel (frameangvel)
        - torso_link_site_vel (velocimeter)
        """
        linvel = data.bind(
            self.mjx_model,
            self._spec.sensor(f"torso_link_site_linvel{self._suffix}"),
        ).sensordata
        angvel = data.bind(
            self.mjx_model,
            self._spec.sensor(f"torso_link_site_angvel{self._suffix}"),
        ).sensordata
        vel = data.bind(
            self.mjx_model,
            self._spec.sensor(f"torso_link_site_vel{self._suffix}"),
        ).sensordata
        sensors = collections.OrderedDict(
            linvel=linvel,
            angvel=angvel,
            velocimeter=vel,
        )
        if flatten:
            sensors, _ = jax.flatten_util.ravel_pytree(sensors)
        return sensors

    def _get_origin(self, data: mjx.Data) -> jp.ndarray:
        """Get origin position in the torso frame."""
        torso = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        torso_frame = torso.xmat
        torso_pos = torso.xpos
        return jp.dot(-torso_pos, torso_frame)

    def get_joint_names(self):
        """Get names of all actuated joints (skip the freejoint)."""
        return [j.name for j in self._spec.joints[1:]]

    def root_body(self, data):
        """Get the root body binding."""
        return data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
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
