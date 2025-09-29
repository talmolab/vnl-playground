"""Base classes for C. elegans environments.

This module provides the core CelegansEnv class and utility functions for
creating and managing C. elegans simulation environments using MuJoCo.
"""

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.celegans import consts
from vnl_mjx.tasks.utils import _recolour_tree, _scale_body_tree, dm_scale_spec


def get_assets() -> Dict[str, bytes]:
    """Get asset files for C. elegans environment.

    Returns:
        Dictionary mapping asset file names to their byte content.
    """
    assets = {}
    mjx_env.update_assets(assets, consts.CELEGANS_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.CELEGANS_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    """Create default configuration for C. elegans environment.

    Returns:
        Configuration dictionary with default parameters for the environment.
    """
    return config_dict.create(
        walker_xml_path=consts.CELEGANS_XML_PATH,
        arena_xml_path=consts.WHITE_ARENA_XML_PATH,
        root_body=consts.ROOT,
        joints=consts.JOINTS,
        bodies=consts.BODIES,
        end_effectors=consts.END_EFFECTORS,
        touch_sensors=consts.TOUCH_SENSORS,
        sensors=consts.SENSORS,
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="cg",
        iterations=4,
        ls_iterations=4,
        noslip_iterations=0,
        nconmax=256,
        njmax=256,
        mujoco_impl="jax",
    )


class CelegansEnv(mjx_env.MjxEnv):
    """Base class for C. Elegans environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, List[Any]]]] = None,
    ) -> None:
        """Initialize the CelegansEnv class with only arena.

        Args:
            config: Configuration dictionary for the environment.
            config_overrides: Optional overrides for the configuration.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._arena_xml_path = str(config.arena_xml_path)
        self._spec = mujoco.MjSpec.from_file(self._arena_xml_path)
        self._compiled = False
        self._n_worms = 0

    def add_worm(
        self,
        torque_actuators: bool,
        rescale_factor: float = 1.0,
        dim: int = 3,
        pos: Tuple[float, float, float] = (0, 0, 0.05),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        friction: Tuple[float, ...] = (1, 1, 0.005, 0.0001, 0.0001),
        solimp: Tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2),
        rgba: Optional[Tuple[float, float, float, float]] = None,
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
        print(f"Loading worm from {self._walker_xml_path}")
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
        root = worm.worldbody
        spawn_body = spawn_site.attach_body(root, "", suffix=suffix)
        self._suffix = suffix
        self._n_worms += 1

        if dim == 3:
            spawn_body.add_freejoint()
        elif dim == 2:
            spawn_body.add_joint(
                axis=(1, 0, 0),
                name="rootx" + suffix,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
            )
            spawn_body.add_joint(
                axis=(0, 1, 0),
                name="rooty" + suffix,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
            )
            spawn_body.add_joint(
                axis=(0, 0, 1),
                name="rootz" + suffix,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
            )
            spawn_body.add_joint(
                axis=(0, 0, 1),
                name="free_body_rot" + suffix,
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=[0, 0, 0],
            )

        for body_name in self.body_names:
            body = worm.body(f"{body_name}{suffix}")
            for geom in body.geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    self._spec.add_pair(
                        name=f"{body_name}_floor",
                        geomname1=geom.name,
                        geomname2="floor",
                        condim=3,
                        friction=friction,
                    )  # , solimp=solimp)

    def add_ghost(
        self,
        rescale_factor: float = 1.0,
        pos: Tuple[float, float, float] = (0, 0, 0.05),
        ghost_rgba: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.3),
        suffix: str = "-ghost",
        dim: int = 3,
        inplace: bool = False,
    ) -> Optional[Tuple[mujoco.MjSpec, mujoco.MjModel]]:
        """Add a ghost worm model to the environment.

        Args:
            rescale_factor: Factor to rescale the ghost body.
            pos: Position to spawn the ghost worm.
            ghost_rgba: RGBA color values for the ghost.
            suffix: Suffix to append to ghost body names.
            dim: Dimensionality of the environment (2 or 3).
            inplace: Whether to modify the spec in place.

        Returns:
            If not inplace, returns tuple of (spec, compiled_model).
            Otherwise returns None.
        """
        print(f"Loading ghost worm from {self._walker_xml_path}")

        if not inplace:
            spec = self._spec.copy()
        else:
            spec = self._spec

        walker_spec = mujoco.MjSpec.from_string(
            epath.Path(self._walker_xml_path).read_text()
        )
        # Scale and recolor the ghost body
        for body in walker_spec.worldbody.bodies:
            _scale_body_tree(body, rescale_factor)
            _recolour_tree(body, rgba=ghost_rgba)
        # Attach as ghost at the offset frame
        frame = spec.worldbody.add_frame(pos=pos, quat=[1, 0, 0, 0])
        root = walker_spec.worldbody
        spawn_body = frame.attach_body(root, "", suffix=suffix)

        if dim == 3:
            spawn_body.add_freejoint()
        elif dim == 2:
            spawn_body.add_joint(
                axis=(1, 0, 0),
                name="rootx" + suffix,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
            )
            spawn_body.add_joint(
                axis=(0, 1, 0),
                name="rooty" + suffix,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
            )
            spawn_body.add_joint(
                axis=(0, 0, 1),
                name="rootz" + suffix,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
            )
            spawn_body.add_joint(
                axis=(0, 0, 1),
                name="free_body_rot" + suffix,
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=[0, 0, 0],
            )

        if not inplace:
            return spec, spec.compile()

    def compile(self, forced: bool = False) -> None:
        """Compile the model from the mj_spec and put models to mjx.

        Args:
            forced: Whether to force recompilation even if already compiled.
        """
        if not self._compiled or forced:
            self._spec.option.noslip_iterations = self.config.noslip_iterations
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self.config.sim_dt
            # Increase offscreen framebuffer size to render at higher resolutions.
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mj_model.opt.iterations = self.config.iterations
            self._mj_model.opt.ls_iterations = self.config.ls_iterations
            self._mjx_model = mjx.put_model(
                self._mj_model
            )  # , impl=self.config.mujoco_impl)
            self._compiled = True

    def _get_root_pos(self, data: mjx.Data) -> jp.ndarray:
        """Get root position from the environment.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Root body position in global coordinates.
        """
        return self.root_body(data).xpos

    def _get_root_quat(self, data: mjx.Data) -> jp.ndarray:
        """Get root quaternion from the environment.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Root body quaternion (w, x, y, z).
        """
        return data.bind(
            self.mjx_model, self._spec.body(f"{self.root_name}{self.suffix}")
        ).xquat

    def _get_appendages_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[jp.ndarray, Dict[str, jp.ndarray]]:
        """Get appendages positions from the environment.

        Args:
            data: MuJoCo simulation data.
            flatten: Whether to flatten the output into a single array.

        Returns:
            If flatten=True, returns flattened array of all appendage positions.
            If flatten=False, returns dict mapping appendage names to positions.
        """
        root = data.bind(
            self.mjx_model, self._spec.body(f"{self.root_name}{self.suffix}")
        )
        appendages_pos = collections.OrderedDict()
        for apppendage_name in self.end_eff_names:
            global_xpos = data.bind(
                self.mjx_model, self.spec.body(f"{apppendage_name}{self.suffix}")
            ).xpos
            egocentric_xpos = jp.dot(global_xpos - root.xpos, root.xmat)
            appendages_pos[apppendage_name] = egocentric_xpos
        if flatten:
            appendages_pos, _ = jax.flatten_util.ravel_pytree(appendages_pos)
        return appendages_pos

    def _get_bodies_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[Dict[str, jp.ndarray], jp.ndarray]:
        """Get global positions of the body parts.

        Args:
            data: MuJoCo simulation data.
            flatten: Whether to flatten the output into a single array.

        Returns:
            If flatten=True, returns flattened array of all body positions.
            If flatten=False, returns dict mapping body names to positions.
        """
        bodies_pos = collections.OrderedDict()
        for body_name in self.body_names:
            global_xpos = data.bind(
                self.mjx_model, self.spec.body(f"{body_name}{self.suffix}")
            ).xpos
            bodies_pos[body_name] = global_xpos
        if flatten:
            bodies_pos, _ = jax.flatten_util.ravel_pytree(bodies_pos)
        return bodies_pos

    def _get_joint_angles(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[jp.ndarray, Dict[str, jp.ndarray]]:
        """Get joint angles of the body parts.

        Args:
            data: MuJoCo simulation data.
            flatten: Whether to flatten the output into a single array.

        Returns:
            If flatten=True, returns flattened array of all joint angles.
            If flatten=False, returns dict mapping joint names to angles.
        """
        joint_angles = collections.OrderedDict()
        for joint_name in self.joint_names:
            try:
                joint_angles[joint_name] = data.bind(
                    self.mjx_model, self.spec.joint(f"{joint_name}{self.suffix}")
                ).qpos
            except Exception as e:
                print(e)
                raise ValueError(
                    f"Joint {joint_name}{self._suffix} not found in the environment.\nAvailable joints: {[joint.name for joint in self._spec.joints]}"
                )
        if flatten:
            joint_angles, _ = jax.flatten_util.ravel_pytree(joint_angles)
        return joint_angles

    def _get_joint_ang_vels(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[jp.ndarray, Dict[str, jp.ndarray]]:
        """Get joint angular velocities of the body parts.

        Args:
            data: MuJoCo simulation data.
            flatten: Whether to flatten the output into a single array.

        Returns:
            If flatten=True, returns flattened array of all joint velocities.
            If flatten=False, returns dict mapping joint names to velocities.
        """
        joint_ang_vels = collections.OrderedDict()
        for joint_name in self.joint_names:
            joint_ang_vels[joint_name] = data.bind(
                self.mjx_model, self.spec.joint(f"{joint_name}{self.suffix}")
            ).qvel
        if flatten:
            joint_ang_vels, _ = jax.flatten_util.ravel_pytree(joint_ang_vels)
        return joint_ang_vels

    def _get_actuator_ctrl(self, data: mjx.Data) -> jp.ndarray:
        """Get actuator control forces.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Array of actuator forces.
        """
        return data.qfrc_actuator

    def _get_body_height(self, data: mjx.Data) -> jp.ndarray:
        """Get the height (z-coordinate) of the root body.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Z-coordinate of the root body position.
        """
        torso_pos = data.bind(
            self.mjx_model, self.spec.body(f"{self.root_name}{self.suffix}")
        ).xpos
        torso_z = torso_pos[2]
        return torso_z  # self.root_body(data).xpos[1]

    def _get_world_zaxis(self, data: mjx.Data) -> jp.ndarray:
        """Get the world z-axis in the root body frame.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Z-axis vector in the root body's coordinate frame.
        """
        return self.root_body(data).xmat.flatten()[6:]

    def _get_proprioception(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[jp.ndarray, Dict[str, Any]]:
        """Get proprioception data from the environment.

        Args:
            data: MuJoCo simulation data.
            flatten: Whether to flatten the output into a single array.

        Returns:
            If flatten=True, returns flattened array of all proprioceptive data.
            If flatten=False, returns dict with proprioceptive components.
        """
        proprioception = collections.OrderedDict(
            joint_angles=self._get_joint_angles(data),
            joint_ang_vels=self._get_joint_ang_vels(data),
            actuator_ctrl=self._get_actuator_ctrl(data),
            body_height=self._get_body_height(data),
            world_zaxis=self._get_world_zaxis(data),
            appendages_pos=self._get_appendages_pos(data, flatten=flatten),
            kinematic_sensors=self._get_kinematic_sensors(data, flatten=flatten),
        )
        if flatten:
            proprioception, _ = jax.flatten_util.ravel_pytree(proprioception)
        return proprioception

    def _get_kinematic_sensors(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[jp.ndarray, Dict[str, jp.ndarray]]:
        """Get kinematic sensors data from the environment.

        Args:
            data: MuJoCo simulation data.
            flatten: Whether to flatten the output into a single array.

        Returns:
            If flatten=True, returns flattened array of sensor data.
            If flatten=False, returns dict mapping sensor names to data.
        """
        try:
            accelerometer = data.bind(
                self.mjx_model, self._spec.sensor(f"accelerometer{self._suffix}")
            ).sensordata
            velocimeter = data.bind(
                self.mjx_model, self._spec.sensor(f"velocimeter{self._suffix}")
            ).sensordata
            gyro = data.bind(
                self.mjx_model, self._spec.sensor(f"gyro{self._suffix}")
            ).sensordata
        except TypeError as e:
            print(f"Kinematic sensors not found for {self._suffix}")
            print(f"Available sensors: {[s.name for s in self._spec.sensors]}")
            raise e
        sensors = collections.OrderedDict(
            accelerometer=accelerometer,
            velocimeter=velocimeter,
            gyro=gyro,
        )
        if flatten:
            sensors, _ = jax.flatten_util.ravel_pytree(sensors)
        return sensors

    def _get_touch_sensors(self, data: mjx.Data) -> jp.ndarray:
        """Get touch sensors data from the environment.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Array of touch sensor readings.
        """
        touches = [
            data.bind(
                self.mjx_model, self._spec.sensor(f"{name}{self._suffix}")
            ).sensordata
            for name in self.config.touch_sensors
        ]
        return jp.array(touches)

    def _get_origin(self, data: mjx.Data) -> jp.ndarray:
        """Get origin position in the torso frame.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Origin position relative to the torso coordinate frame.
        """
        torso = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        torso_frame = torso.xmat
        torso_pos = torso.xpos
        return jp.dot(-torso_pos, torso_frame)

    def _get_egocentric_camera(self, data: mjx.Data) -> None:
        """Get egocentric camera data from the environment.

        Args:
            data: MuJoCo simulation data.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError(
            "Egocentric camera is not implemented for this environment."
        )

    def root_body(self, data: mjx.Data) -> mjx.Data:
        """Get the root body from the simulation data.

        Args:
            data: MuJoCo simulation data.

        Returns:
            Root body data.
        """
        # TODO: Double-check which body should be considered the root (walker or torso)
        return data.bind(
            self.mjx_model, self._spec.body(f"{consts.ROOT}{self._suffix}")
        )

    def save_spec(self, path: str, return_str: bool = False) -> Optional[str]:
        """Save the spec to a file.

        Args:
            path: Path where to save the XML specification.
            return_str: Whether to return the XML string.

        Returns:
            XML string if return_str is True, otherwise None.
        """
        xml_str = self._spec.to_xml()
        root = ET.fromstring(xml_str)
        with open(path, "wb") as f:
            f.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        print(f"Saved spec to {path}")
        if return_str:
            return xml_str
        else:
            return None

    @property
    def action_size(self) -> int:
        """Number of actuated degrees of freedom.

        Returns:
            Dimension of the action space.
        """
        return self._mjx_model.nu

    @property
    def xml_path(self) -> str:
        """Path to the walker XML file.

        Returns:
            String path to the walker XML specification.
        """
        return self._walker_xml_path

    @property
    def walker_xml_path(self) -> str:
        """Path to the walker XML file.

        Returns:
            String path to the walker XML specification.
        """
        return self._walker_xml_path

    @property
    def arena_xml_path(self) -> str:
        """Path to the arena XML file.

        Returns:
            String path to the arena XML specification.
        """
        return self._arena_xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Compiled MuJoCo model.

        Returns:
            The compiled MuJoCo model instance.
        """
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """MuJoCo XLA (JAX) model.

        Returns:
            The MJX model instance for JAX-based simulation.
        """
        return self._mjx_model

    @property
    def spec(self) -> mujoco.MjSpec:
        """Get the MuJoCo specification.

        Returns:
            The MuJoCo specification object.
        """
        return self._spec

    @property
    def is_compiled(self) -> bool:
        """Check if the model has been compiled.

        Returns:
            True if the model has been compiled.
        """
        return self._compiled

    @property
    def suffix(self) -> str:
        """Get the suffix used for body/joint names.

        Returns:
            Suffix string (e.g., "-worm").
        """
        return getattr(self, "_suffix", "")

    @property
    def n_worms(self) -> int:
        """Get the number of worms in the environment.

        Returns:
            Number of worms.
        """
        return self._n_worms

    @property
    def config(self) -> Any:
        """Get the environment configuration.

        Returns:
            The configuration object.
        """
        return self._config

    @property
    def root_name(self) -> str:
        """Get the root body name.

        Returns:
            Name of the root body.
        """
        return self.config.root_body

    @property
    def joint_names(self) -> List[str]:
        """Get the list of joint names in the configuration.

        Returns:
            List of joint names.
        """
        return self.config.joints

    @property
    def body_names(self) -> List[str]:
        """Get the list of body names in the configuration.

        Returns:
            List of body names.
        """
        return self.config.bodies

    @property
    def end_eff_names(self) -> List[str]:
        """Get the list of end effector names in the configuration.

        Returns:
            List of end effector names.
        """
        return self.config.end_effectors

    @property
    def n_bodies(self) -> int:
        """Get number of bodies in the configuration.

        Returns:
            Number of body parts.
        """
        return len(self.config.bodies)

    @property
    def n_joints(self) -> int:
        """Get number of joints in the configuration.

        Returns:
            Number of joints.
        """
        return len(self.config.joints)

    @property
    def n_end_effectors(self) -> int:
        """Get number of end effectors.

        Returns:
            Number of end effector bodies.
        """
        return len(self.config.end_effectors)
