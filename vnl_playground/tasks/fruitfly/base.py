"""Base classes for fruitfly"""

import collections
from typing import Any, Dict, Mapping, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
import logging
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_playground.tasks.fruitfly import consts
from vnl_playground.tasks.utils import _scale_body_tree, _recolour_tree, dm_scale_spec
from vnl_playground.tasks.reward_registry import RewardRegistry


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.FRUITFLY_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.FRUITFLY_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.FRUITFLY_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        sim_dt=0.001,
        ctrl_dt=0.002,
        solver="newton",
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        mujoco_impl="jax",
    )


class FruitflyEnv(mjx_env.MjxEnv):
    """Base class for fruitfly environments."""

    # Subclasses should set this to their RewardRegistry instance
    _registry: RewardRegistry = None
    _default_render_camera: str = "track1"

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
        pos=(0, 0, 0.0),
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

    def _scale_fly_spec(self, spec, scale: float):
        """Scale fly spec (uses thorax as root, not walker).

        The dm_scale_spec utility looks for body("walker") which doesn't exist
        in the fruitfly XML, so we need a fly-specific version.

        Args:
            spec: MuJoCo spec to scale.
            scale: Scale factor to apply.

        Returns:
            Scaled MuJoCo spec.
        """
        scaled_spec = spec.copy()

        def scale_bodies(parent, scale=1.0):
            body = parent.first_body()
            while body:
                if body.pos is not None:
                    body.pos = body.pos * scale
                for geom in body.geoms:
                    geom.fromto = geom.fromto * scale
                    geom.size = geom.size * scale
                    if geom.pos is not None:
                        geom.pos = geom.pos * scale
                scale_bodies(body, scale)
                body = parent.next_body(body)

        for actuator in scaled_spec.actuators:
            actuator.gear = actuator.gear * scale * scale

        for keypoint in scaled_spec.keys:
            qpos = keypoint.qpos
            qpos[2] = qpos[2] * scale
            keypoint.qpos = qpos

        # Use thorax as root (not walker)
        scale_bodies(scaled_spec.body("thorax"), scale)
        return scaled_spec

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
            self._mj_model.opt.solver = {
                "cg": mujoco.mjtSolver.mjSOL_CG,
                "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            }[self._config.solver.lower()]
            self._mjx_model = mjx.put_model(
                self._mj_model, impl=self._config.mujoco_impl
            )
            self._compiled = True
            cam_name = f"{self._default_render_camera}{self._suffix}"
            cam_names = [
                self._mj_model.camera(i).name for i in range(self._mj_model.ncam)
            ]
            self._default_render_camera = cam_name if cam_name in cam_names else -1

    def _get_appendages_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[dict[str, jp.ndarray], jp.ndarray]:
        """Get _egocentric_ position of the appendages (claws)."""
        thorax = data.bind(self.mjx_model, self._spec.body(f"thorax{self._suffix}"))
        appendages_pos = collections.OrderedDict()
        for appendage_name in consts.END_EFFECTORS:
            global_xpos = data.bind(
                self.mjx_model, self._spec.body(f"{appendage_name}{self._suffix}")
            ).xpos
            egocentric_xpos = jp.dot(global_xpos - thorax.xpos, thorax.xmat)
            appendages_pos[appendage_name] = egocentric_xpos
        if flatten:
            appendages_pos, _ = jax.flatten_util.ravel_pytree(appendages_pos)
        return appendages_pos

    def _get_bodies_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[dict[str, jp.ndarray], jp.ndarray]:
        """Get _global_ positions of the body parts."""
        bodies_pos = collections.OrderedDict()
        for body_name in consts.BODIES:
            global_xpos = data.bind(
                self.mjx_model, self._spec.body(f"{body_name}{self._suffix}")
            ).xpos
            bodies_pos[body_name] = global_xpos
        if flatten:
            bodies_pos, _ = jax.flatten_util.ravel_pytree(bodies_pos)
        return bodies_pos

    def _get_joint_angles(self, data: mjx.Data) -> jp.ndarray:
        """Extract 36 joint angles from qpos (after 7 DoF for free joint)."""
        return data.qpos[7:]

    def _get_joint_ang_vels(self, data: mjx.Data) -> jp.ndarray:
        """Extract 36 joint velocities from qvel (after 6 DoF for free joint)."""
        return data.qvel[6:]

    def _get_actuator_ctrl(self, data: mjx.Data) -> jp.ndarray:
        """Get actuator control forces."""
        return data.qfrc_actuator

    def _get_body_height(self, data: mjx.Data) -> jp.ndarray:
        """Get thorax Z position."""
        thorax_pos = data.bind(
            self.mjx_model, self._spec.body(f"thorax{self._suffix}")
        ).xpos
        thorax_z = thorax_pos[2]
        return thorax_z

    def _get_world_zaxis(self, data: mjx.Data) -> jp.ndarray:
        """Get gravity direction in body frame."""
        return self.root_body(data).xmat.flatten()[6:]

    def _get_origin(self, data: mjx.Data) -> jp.ndarray:
        """Get origin position in the thorax frame."""
        thorax = data.bind(self.mjx_model, self._spec.body(f"thorax{self._suffix}"))
        return jp.dot(-thorax.xpos, thorax.xmat)

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
        """Get kinematic sensors data from the environment."""
        accelerometer = data.bind(
            self.mjx_model, self._spec.sensor(f"accelerometer{self._suffix}")
        ).sensordata
        velocimeter = data.bind(
            self.mjx_model, self._spec.sensor(f"velocimeter{self._suffix}")
        ).sensordata
        gyro = data.bind(
            self.mjx_model, self._spec.sensor(f"gyro{self._suffix}")
        ).sensordata
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

        Note: Touch sensors are currently commented out in fruitfly XMLs,
        so this returns an empty array. Enable touch sensors in the XML
        to get actual touch data.
        """
        return jp.array([])

    def get_joint_names(self):
        """Get joint names from the model specification."""
        return map(lambda j: j.name, self._spec.joints[1:])

    def root_body(self, data: mjx.Data):
        """Return thorax body as root reference."""
        return data.bind(self.mjx_model, self._spec.body(f"thorax{self._suffix}"))

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: dict
    ) -> float:
        """Compute total reward from registered reward functions.

        Args:
            data: Simulation data.
            info: State info dictionary.
            metrics: Metrics dictionary for logging.

        Returns:
            Total reward value.
        """
        if self._registry is None:
            raise RuntimeError(
                f"{type(self).__name__} has no RewardRegistry assigned. "
                "Subclasses must set `_registry` as a class attribute."
            )
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            if name not in self._registry.rewards:
                raise KeyError(
                    f"Reward '{name}' not found in {type(self).__name__}'s registry. "
                    f"Available: {list(self._registry.rewards.keys())}"
                )
            net_reward += self._registry.rewards[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics: dict) -> bool:
        """Check if episode should terminate based on registered criteria.

        Args:
            data: Simulation data.
            info: State info dictionary.
            metrics: Metrics dictionary for logging.

        Returns:
            Boolean indicating if episode should terminate.
        """
        if self._registry is None:
            raise RuntimeError(
                f"{type(self).__name__} has no RewardRegistry assigned. "
                "Subclasses must set `_registry` as a class attribute."
            )
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            if name not in self._registry.terminations:
                raise KeyError(
                    f"Termination '{name}' not found in {type(self).__name__}'s registry. "
                    f"Available: {list(self._registry.terminations.keys())}"
                )
            terminated = self._registry.terminations[name](self, data, info, **kwargs)
            any_terminated = jp.logical_or(any_terminated, terminated)
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            jax.flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0]
        )

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self):
        obs = self.non_flattened_observation_size
        return jp.sum(jax.flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)

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
