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
from vnl_playground.tasks.utils import _scale_body_tree, _recolour_tree, scale_spec


def get_assets() -> Dict[str, bytes]:
    """Bundle XML + mesh OBJ + texture so MjSpec can load from bytes."""
    assets = {}
    mjx_env.update_assets(assets, consts.STICK_PATH / "xmls", "*.xml")
    mesh_dir = consts.STICK_PATH / "xmls" / "stick_insect_urdf" / "meshes" / "obj"
    if mesh_dir.is_dir():
        mjx_env.update_assets(assets, mesh_dir, "*.obj")
        mjx_env.update_assets(assets, mesh_dir, "*.mtl")
        mjx_env.update_assets(assets, mesh_dir, "*.png")
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
            stick = scale_spec(stick, rescale_factor, root_body="reference_base")

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
            stick_spec = scale_spec(
                stick_spec, rescale_factor, root_body="reference_base"
            )
        for body in stick_spec.worldbody.bodies:
            _recolour_tree(body, rgba=ghost_rgba)
        spawn_frame = self._spec.worldbody.add_frame(pos=pos, quat=[1, 0, 0, 0])
        spawn_frame.attach_body(stick_spec.body("reference_base"), "", suffix=suffix)

    @staticmethod
    def _apply_cgs_rescaling(mj_model: mujoco.MjModel) -> None:
        """Convert the compiled model from SI (m / kg / s) to CGS (cm / g / s)
        in-place. The mesh XML still uses SI numbers; this rescales every
        length / mass / inertia / actuator / damping field so the solver
        operates at the float32 sweet spot the fly model uses.

        Why: at SI scale (50 mg, 22 mm bug, 489 µN body weight) the default
        MuJoCo contact stabilization is too soft — the bug penetrates the
        floor by 1-7 mm at equilibrium under gravity alone. CGS makes the
        numerical values 6-7 orders of magnitude larger, so default contact
        params behave like a rigid surface (see fly model precedent).

        Conversion factors:
          length    × 100    (m → cm)
          mass      × 1000   (kg → g)
          inertia   × 1e7    (kg·m² → g·cm²)
          force     × 1e5    (N → dyne)
          torque    × 1e7    (N·m → dyne·cm)
          damping   × 1e7    (N·m·s → dyne·cm·s)
          armature  × 1e7    (kg·m² → g·cm²)
          gravity   × 100    (m/s² → cm/s²; magnitude 9.81 → 981)

        Final actuator gear ≈ 100 dyne·cm peak torque per ctrl-unit
        (= 1e-5 N·m, matches a 50 mg insect leg muscle).
        """
        L = 100.0       # length: m → cm
        M = 1000.0      # mass:   kg → g
        I = 1.0e7       # inertia, torque, damping
        F = 1.0e5       # force:  N → dyne

        # --- Lengths ---
        mj_model.body_pos[:] *= L
        mj_model.body_ipos[:] *= L
        mj_model.geom_pos[:] *= L
        mj_model.geom_size[:] *= L
        mj_model.geom_rbound[:] *= L
        mj_model.jnt_pos[:] *= L
        mj_model.jnt_margin[:] *= L
        mj_model.site_pos[:] *= L
        mj_model.site_size[:] *= L
        if mj_model.nmesh > 0 and mj_model.nmeshvert > 0:
            mj_model.mesh_vert[:] *= L
            mj_model.mesh_pos[:] *= L
            mj_model.mesh_normal[:] *= 1.0  # unit vectors, unchanged
        if mj_model.ncam > 0:
            mj_model.cam_pos[:] *= L
            mj_model.cam_pos0[:] *= L
            mj_model.cam_poscom0[:] *= L
        if mj_model.nlight > 0:
            mj_model.light_pos[:] *= L
            mj_model.light_pos0[:] *= L
        # Initial qpos for free joints: first 3 components are world xyz (m → cm).
        # Joint angles (radians) and quaternions unchanged.
        for j in range(mj_model.njnt):
            if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                addr = mj_model.jnt_qposadr[j]
                mj_model.qpos0[addr : addr + 3] *= L
                mj_model.qpos_spring[addr : addr + 3] *= L

        # --- Mass and inertia ---
        mj_model.body_mass[:] *= M
        mj_model.body_inertia[:] *= I
        # body_subtreemass: recompute by walking leaves → root in topological order.
        subtreemass = mj_model.body_mass.copy()
        for i in range(mj_model.nbody - 1, 0, -1):
            parent = mj_model.body_parentid[i]
            subtreemass[parent] += subtreemass[i]
        mj_model.body_subtreemass[:] = subtreemass

        # --- Actuators / forces ---
        # gear is N·m → dyne·cm. XML has gear=1, so this brings it to 1e7.
        # Then apply biological torque scale (peak ~100 dyne·cm = 1e-5 N·m).
        mj_model.actuator_gear[:, 0] *= I  # SI → CGS unit conversion
        mj_model.actuator_gear[:, 0] *= 1.0e-5  # biological scale for 50 mg insect
        # final gear ≈ 100 dyne·cm
        mj_model.actuator_forcerange[:] *= I  # N·m bounds → dyne·cm bounds

        # --- Damping / armature floors in CGS units ---
        # Old SI floors: armature 1e-9 kg·m², damping 5e-7 N·m·s.
        # CGS equivalents: 1e-2 g·cm² and 5 dyne·cm·s.
        mj_model.dof_armature[:] *= I
        mj_model.dof_damping[:] *= I
        mj_model.dof_armature[:] = np.maximum(mj_model.dof_armature, 1.0e-2)
        # damping floor 5 dyne·cm·s → joint terminal velocity ≈
        # peak_torque/damping = 100 / 5 = 20 rad/s (same as SI value).
        mj_model.dof_damping[6:] = np.maximum(mj_model.dof_damping[6:], 5.0)

        # --- Gravity ---
        mj_model.opt.gravity[:] = np.array([0.0, 0.0, -981.0])

        # --- Per-body invweight (must be recomputed for compile-time consts).
        # MuJoCo derives body_invweight0 etc. at compile from a "neutral pose"
        # forward pass — these are no longer correct after our scaling. Force
        # a recompute via mj_setConst which re-runs the compile-time pass on
        # the current model.
        mujoco.mj_setConst(mj_model, mujoco.MjData(mj_model))

    # Backwards-compatible alias so existing callers keep working.
    _apply_si_rescaling = _apply_cgs_rescaling

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
            self._apply_si_rescaling(self._mj_model)
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
