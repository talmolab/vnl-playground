"""Base classes for mouse (arena-first, add walker later)."""

import os
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from tqdm import tqdm

from mujoco_playground._src import mjx_env
from vnl_playground.tasks.mouse import consts, contact_presets
from vnl_playground.tasks.reward_registry import RewardRegistry


def get_assets() -> Dict[str, bytes]:
    """Collect XML + asset files into a dict for bundling/remote loading.

    Returns:
        Dict[str, bytes]: Mapping of asset filenames to their byte contents.
    """
    assets = {}
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    """Default sim + XML config for mouse tasks.

    Returns:
        config_dict.ConfigDict: Configuration with walker/arena paths, solver
            settings, PD gains, timesteps, and episode length.
    """
    return config_dict.create(
        walker_xml_path=consts.MOUSE_XML_PATH,
        arena_xml_path=consts.MOUSE_ARENA_XML_PATH,  # separate empty arena
        # Top-level body name(s) of the walker XML to attach to the arena
        # (see MouseBaseEnv.add_mouse for why multi-root models need more
        # than one name here).
        root_bodies=("clavicle",),
        ctrl_dt=0.0025,  # physics_steps_per_control_step=2 -> 0.00125*2
        sim_dt=0.00125,  # mj_model_timestep from imitation settings
        solver="cg",  # CG solver as in imitation settings
        iterations=6,
        ls_iterations=6,
        noslip_iterations=0,
        Kp=35.0,
        Kd=0.5,
        episode_length=100,
        mujoco_impl="jax",
        # None = leave whatever the arena+walker attach conflict resolution
        # produces (kept for exact backward compat with existing tasks); set
        # explicitly when the walker XML's own <option integrator="..."/>
        # must win over the arena's (e.g. v22 wants Euler, arena.xml says
        # RK4, and the attach-conflict resolution keeps the arena's value).
        integrator=None,
        # Physics overrides (None = use XML defaults)
        joint_damping=None,   # float -> sets mj_model.dof_damping[:]
        joint_armature=None,  # float -> sets mj_model.dof_armature[:]
        joint_stiffness=None, # float -> sets mj_model.jnt_stiffness[:]
        force_scale=None,     # float -> multiplies mj_model.actuator_gainprm[:, 0]
        # Warp-backend contact/constraint buffer sizes, passed explicitly to
        # mjx.make_data(). The XML's own <size njmax=.../nconmax=.../> tag is
        # NOT read by mjx.make_data for impl="warp" -- left None, MJX's Warp
        # backend derives naconmax/njmax from a single-world (nworld=1)
        # heuristic based on mjm.nv (0 collision geoms -> defaults as low as
        # naconmax=16 total, shared across the WHOLE vmapped batch, not
        # per-world), which is far too small once vmapped across thousands of
        # parallel envs at training time -- this was the true root cause of
        # the "broadphase overflow"/"nefc overflow" warnings, not collision
        # geometry complexity. naconmax is a TOTAL across all vmapped worlds;
        # njmax is PER WORLD. Set explicitly (see imitation_arm_hand.py) for
        # any task with real contacts.
        naconmax=None,
        njmax=None,
        # Hand<->joystick contact hardening, applied in compile() (see
        # contact_presets.py for what each preset does and what the kinematic
        # probe measured it to deliver). None/"shipped" is a no-op, so every
        # config predating this key keeps its original contact behaviour.
        contact_preset=None,
        contact_stiffness_mult=30.0,
    )


class MouseBaseEnv(mjx_env.MjxEnv):
    """Arena-first base for mouse environments with add_mouse() then compile."""

    _registry: RewardRegistry = None
    _default_render_camera: str = "my_camera"

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

    def _load_walker_spec(self) -> mujoco.MjSpec:
        """Load the walker XML, repointing meshdir if the XML's own is dead.

        The Janelia v22-v26 XMLs hardcode an absolute
        meshdir="/root/vast/eric/janelia_model/vNN" that only exists inside
        Eric's container, because the bone meshes are an unreleased asset and
        are intentionally not committed (see consts.janelia_mesh_dir()).
        Without this, a fresh clone dies inside mj_compile on a missing .obj.

        Only overrides when the XML's own meshdir does not resolve on this
        machine, so Eric's container and the mouse XMLs that use working
        relative meshdirs (akira_muscle, moving_shoulder_ik) keep their
        current behaviour untouched.
        """
        spec = mujoco.MjSpec.from_file(self._walker_xml_path)
        mesh_dir = consts.janelia_mesh_dir()
        if mesh_dir is None:
            return spec
        # MuJoCo resolves a relative meshdir against the XML's own directory.
        declared = spec.meshdir
        resolved = (
            declared
            if os.path.isabs(declared)
            else os.path.join(os.path.dirname(self._walker_xml_path), declared)
        )
        if not os.path.isdir(resolved):
            spec.meshdir = mesh_dir
        return spec

    def add_mouse(
        self,
        freejoint: bool = False,
        pos: Union[tuple[float, float, float], list[float]] = (0.0, 0.0, 0.02),
        suffix: str = "-mouse",
        rgba: Optional[tuple[float, float, float, float]] = None,
        root_bodies: Sequence[str] = ("clavicle",),
    ) -> None:
        """
        Attach a mouse model to the arena at the given position.

        Args:
            freejoint: If True, add a freejoint on each attached root body.
            pos: Spawn position (x, y, z) in arena frame.
            suffix: Name suffix to avoid collisions for multiple mice.
            rgba: Optional per-geom RGBA override for the attached mouse.
            root_bodies: Names of the walker model's top-level bodies to
                attach. Most models have a single kinematic root
                ("clavicle"). Models with disconnected subtrees under
                worldbody (e.g. the v22 arm+hand+joystick model, where
                "shoulder_base" carries the arm and a separate
                "joystick_base" carries the manipulandum) need every root
                listed here, or the un-listed subtree's joints/bodies are
                silently dropped. When more than one name is given, the
                *entire* walker spec is attached in one shot (see the
                multi-root branch below) rather than attached body-by-body,
                since `attach_body()` re-imports the source spec's whole
                asset table on every call -- attaching two subtrees from the
                same file with the same suffix collides on mesh names the
                second time. This means any other top-level bodies the
                walker XML happens to declare (e.g. a camera-mount "ground"
                body) come along too; harmless in practice, but a real
                behavior difference from the single-root path below.

        Returns:
            None
        """
        # Attach using a frame (like rodent) instead of site for better positioning
        spawn_frame = self._spec.worldbody.add_frame(
            pos=list(pos),
            quat=[1, 0, 0, 0],
        )
        self._suffix = suffix

        if len(root_bodies) == 1:
            mouse_spec = self._load_walker_spec()
            body = spawn_frame.attach_body(mouse_spec.body(root_bodies[0]), "", suffix)
            if freejoint:
                body.add_freejoint()
            if rgba is not None:
                for g in getattr(body, "geom", []):
                    g.rgba = list(rgba)
        else:
            if freejoint or rgba is not None:
                raise NotImplementedError(
                    "freejoint/rgba are not yet supported when attaching a "
                    "multi-root walker (root_bodies has more than one "
                    "entry); neither is needed by any current caller."
                )
            mouse_spec = self._load_walker_spec()
            self._spec.attach(mouse_spec, prefix="", suffix=suffix, frame=spawn_frame)

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
        root_bodies: Sequence[str] = ("clavicle",),
    ) -> None:
        """
        Attach a ghost/reference mouse (no freejoint, translucent, non-colliding).

        Args:
            pos: Spawn position (x, y, z) in arena frame.
            suffix: Name suffix to avoid collisions for multiple ghosts.
            ghost_rgba: RGBA to tint all geoms of the ghost mouse.
            no_collision: If True, set contype=conaffinity=0 on all geoms.
            root_bodies: Names of the walker model's top-level bodies to
                attach (see `add_mouse` for why multi-root models need this).

        Returns:
            None
        """
        # Attach using a frame for consistent positioning
        spawn_frame = self._spec.worldbody.add_frame(
            pos=list(pos),
            quat=[1, 0, 0, 0],
        )
        # Intentionally NO freejoint: kinematically tied through the attached tree.
        def _recolor(body):
            for g in body.geoms:
                g.rgba = list(ghost_rgba)
                if no_collision:
                    g.contype = 0
                    g.conaffinity = 0
            for child in body.bodies:
                _recolor(child)

        if len(root_bodies) == 1:
            mouse_spec = self._load_walker_spec()
            body = spawn_frame.attach_body(mouse_spec.body(root_bodies[0]), "", suffix)
            _recolor(body)
        else:
            # Whole-model attach (see add_mouse) -- recolor every newly
            # attached top-level body (name ends with `suffix`) recursively.
            mouse_spec = self._load_walker_spec()
            self._spec.attach(mouse_spec, prefix="", suffix=suffix, frame=spawn_frame)
            for body in self._spec.worldbody.bodies:
                if body.name.endswith(suffix):
                    _recolor(body)

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
            if self._config.integrator is not None:
                self._mj_model.opt.integrator = {
                    "euler": mujoco.mjtIntegrator.mjINT_EULER,
                    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                }[self._config.integrator.lower()]

            # Apply physics overrides before mjx.put_model()
            if self._config.joint_damping is not None:
                self._mj_model.dof_damping[:] = self._config.joint_damping
            if self._config.joint_armature is not None:
                self._mj_model.dof_armature[:] = self._config.joint_armature
            if self._config.joint_stiffness is not None:
                self._mj_model.jnt_stiffness[:] = self._config.joint_stiffness
            if self._config.force_scale is not None:
                self._mj_model.actuator_gainprm[:, 0] *= self._config.force_scale

            # Hand<->joystick contact hardening. Must run after opt.timestep is
            # final above: the shipped stiffness is read back through MuJoCo's
            # REFSAFE clamp (solref timeconst floored at 2*dt), so measuring it
            # earlier would read a different k than the sim will actually use.
            if getattr(self._config, "contact_preset", None):
                contact_presets.apply_contact_preset(
                    self._mj_model,
                    self._config.contact_preset,
                    stiffness_mult=self._config.contact_stiffness_mult,
                )

            # High-res offscreen buffer for nice renders
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160

            # Use configured implementation (warp/jax)
            self._mjx_model = mjx.put_model(
                self._mj_model, impl=self._config.mujoco_impl
            )
            self._compiled = True
            cam_name = f"{self._default_render_camera}{self._suffix}"
            cam_names = [
                self._mj_model.camera(i).name for i in range(self._mj_model.ncam)
            ]
            self._default_render_camera = cam_name if cam_name in cam_names else -1

    @property
    def action_size(self) -> int:
        """Number of actuators (action dimensions) in the compiled model.

        Returns:
            int: Number of actuators.
        """
        return self._mjx_model.nu

    @property
    def xml_path(self) -> str:
        """Path to the walker XML file (alias for walker_xml_path).

        Returns:
            str: Filesystem path to the walker MJCF XML.
        """
        return self._walker_xml_path

    @property
    def walker_xml_path(self) -> str:
        """Path to the walker (mouse arm) XML file.

        Returns:
            str: Filesystem path to the walker MJCF XML.
        """
        return self._walker_xml_path

    @property
    def arena_xml_path(self) -> str:
        """Path to the arena XML file.

        Returns:
            str: Filesystem path to the arena MJCF XML.
        """
        return self._arena_xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        """The compiled MuJoCo model (CPU).

        Returns:
            mujoco.MjModel: The compiled MuJoCo model.
        """
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """The compiled MJX model (JAX/accelerator).

        Returns:
            mjx.Model: The compiled MJX model for use with JAX.
        """
        return self._mjx_model

    @property
    def dt(self) -> float:
        """Control timestep (ctrl_dt)."""
        return self._config.ctrl_dt

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
