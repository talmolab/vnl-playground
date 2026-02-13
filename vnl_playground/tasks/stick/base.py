"""Base classes for stick bug (Sungaya inexpectata)."""

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
from vnl_playground.tasks.stick import consts
from vnl_playground.tasks.reward_registry import RewardRegistry
from vnl_playground.tasks.utils import _scale_body_tree, _recolour_tree


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.STICK_PATH / "xmls", "*.xml")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.STICK_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="newton",
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        mujoco_impl="jax",
    )


class StickBugEnv(mjx_env.MjxEnv):
    """Base class for stick bug environments."""

    _registry: RewardRegistry = None
    _default_render_camera: str = "close_profile"

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._arena_xml_path = str(config.arena_xml_path)
        self._spec = mujoco.MjSpec.from_file(str(config.arena_xml_path))
        self._compiled = False

    def add_stick(
        self,
        torque_actuators: bool = False,
        rescale_factor: float = 1.0,
        pos: tuple[float, float, float] = (0, 0, 0),
        quat: tuple[float, float, float, float] = (1, 0, 0, 0),
        rgba: Optional[tuple[float, float, float, float]] = None,
        suffix: str = "-stick",
    ) -> None:
        """Adds the stick bug model to the environment.

        Args:
            torque_actuators: Whether to convert motors to torque-mode actuators.
                Note: Not supported for stick bug, will raise if True.
            rescale_factor: Factor to rescale the stick bug body. Defaults to 1.0.
            pos: Position (x, y, z) to spawn the stick bug.
            quat: Quaternion (w, x, y, z) for orientation.
            rgba: RGBA color values for recoloring. If None, no recoloring.
            suffix: Suffix to append to body names. Defaults to "-stick".
        """
        stick = mujoco.MjSpec.from_file(self._walker_xml_path)

        if torque_actuators:
            raise ValueError(
                "Torque actuator conversion is not supported for stick bug."
            )

        if rescale_factor != 1.0:
            logging.info(f"Rescaling stick bug with scale factor {rescale_factor}")
            stick = self._scale_stick_spec(stick, rescale_factor)

        if rgba is not None:
            for body in stick.worldbody.bodies:
                _recolour_tree(body, rgba=rgba)

        spawn_frame = self._spec.worldbody.add_frame(
            pos=pos,
            quat=quat,
        )

        # Attach the reference_base body (root of the stick bug).
        # The stick XML already contains a free joint named "root",
        # so we do NOT call add_freejoint() here.
        spawn_body = spawn_frame.attach_body(
            stick.body("reference_base"), "", suffix=suffix
        )
        self._suffix = suffix

        # Add explicit floor-foot contact pairs.
        # The stick_fast.xml disables automatic contact generation
        # (contype="0" conaffinity="0") and relies on explicit pairs.
        for geom_name in consts.FOOT_GEOMS:
            self._spec.add_pair(
                geomname1="floor",
                geomname2=f"{geom_name}{self._suffix}",
            )

    def add_ghost_stick(
        self,
        rescale_factor: float = 1.0,
        pos=(0, 0, 0),
        ghost_rgba=(0.8, 0.8, 0.8, 0.3),
        suffix="-ghost",
    ):
        """Adds a ghost stick bug model to the environment."""
        stick_spec = mujoco.MjSpec.from_file(self._walker_xml_path)
        if rescale_factor != 1.0:
            stick_spec = self._scale_stick_spec(stick_spec, rescale_factor)
        for body in stick_spec.worldbody.bodies:
            _recolour_tree(body, rgba=ghost_rgba)
        spawn_frame = self._spec.worldbody.add_frame(pos=pos, quat=[1, 0, 0, 0])
        spawn_frame.attach_body(stick_spec.body("reference_base"), "", suffix=suffix)

    def _scale_stick_spec(self, spec, scale: float):
        """Scale stick bug spec using reference_base as root.

        The dm_scale_spec utility uses body("walker") which doesn't exist
        in the stick XML, so we need a stick-specific version.
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

        scale_bodies(scaled_spec.body("reference_base"), scale)
        return scaled_spec

    def compile(self, forced=False) -> None:
        """Compiles the model from the mj_spec and puts models to mjx."""
        if not self._compiled or forced:
            self._spec.option.noslip_iterations = self._config.noslip_iterations
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
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
        """Get egocentric position of the end effectors (claws)."""
        root = data.bind(
            self.mjx_model,
            self._spec.body(f"reference_base{self._suffix}"),
        )
        appendages_pos = collections.OrderedDict()
        for appendage_name in consts.END_EFFECTORS:
            global_xpos = data.bind(
                self.mjx_model,
                self._spec.body(f"{appendage_name}{self._suffix}"),
            ).xpos
            egocentric_xpos = jp.dot(global_xpos - root.xpos, root.xmat)
            appendages_pos[appendage_name] = egocentric_xpos
        if flatten:
            appendages_pos, _ = jax.flatten_util.ravel_pytree(appendages_pos)
        return appendages_pos

    def _get_bodies_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[dict[str, jp.ndarray], jp.ndarray]:
        """Get global positions of the body parts."""
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
        """Extract joint angles from qpos (after 7 DoF for free joint)."""
        return data.qpos[7:]

    def _get_joint_ang_vels(self, data: mjx.Data) -> jp.ndarray:
        """Extract joint velocities from qvel (after 6 DoF for free joint)."""
        return data.qvel[6:]

    def _get_actuator_ctrl(self, data: mjx.Data) -> jp.ndarray:
        return data.qfrc_actuator

    def _get_body_height(self, data: mjx.Data) -> jp.ndarray:
        """Get reference_base Z position."""
        root_pos = data.bind(
            self.mjx_model,
            self._spec.body(f"reference_base{self._suffix}"),
        ).xpos
        return root_pos[2]

    def _get_world_zaxis(self, data: mjx.Data) -> jp.ndarray:
        """Get gravity direction in body frame."""
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
        """Get kinematic sensors data from the environment."""
        accelerometer = data.bind(
            self.mjx_model,
            self._spec.sensor(f"accelerometer{self._suffix}"),
        ).sensordata
        velocimeter = data.bind(
            self.mjx_model,
            self._spec.sensor(f"velocimeter{self._suffix}"),
        ).sensordata
        gyro = data.bind(
            self.mjx_model,
            self._spec.sensor(f"gyro{self._suffix}"),
        ).sensordata
        sensors = collections.OrderedDict(
            accelerometer=accelerometer,
            velocimeter=velocimeter,
            gyro=gyro,
        )
        if flatten:
            sensors, _ = jax.flatten_util.ravel_pytree(sensors)
        return sensors

    def _get_origin(self, data: mjx.Data) -> jp.ndarray:
        """Get origin position in the body frame."""
        root = data.bind(
            self.mjx_model,
            self._spec.body(f"reference_base{self._suffix}"),
        )
        return jp.dot(-root.xpos, root.xmat)

    def get_joint_names(self):
        return map(lambda j: j.name, self._spec.joints[1:])

    def root_body(self, data: mjx.Data):
        """Return reference_base body as root reference."""
        return data.bind(
            self.mjx_model,
            self._spec.body(f"reference_base{self._suffix}"),
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

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: dict
    ) -> float:
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
