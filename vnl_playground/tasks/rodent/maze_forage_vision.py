"""Vision-guided sparse maze-foraging task for the virtual rodent.

The rodent is dropped into a **fixed** box-walled maze and must find and reach
``n_treats`` treats scattered through it.  Each treat pays ``+weight`` the first
time the rodent gets within ``treat_reach_threshold`` of it (in xy), then slides
underground so it can neither be seen nor re-collected.  The episode ends when
every treat has been collected, or on a fall / NaN (dm_control ``ManyGoalsMaze``
semantics).

What makes this task *pure vision*:
``task_obs`` is ``[prev_action, kinematic_sensors, touch_sensors, origin]`` and
carries **no** treat vector.  Every other vision task in this repo leaks an
egocentric target vector into ``task_obs``; this one deliberately does not, so
the only channel that can tell the policy where a treat is, is the egocentric
camera image (a zeros placeholder here, filled in by ``VisionRenderWrapper``).
``privileged_state`` does carry egocentric treat vectors and the collected mask
for the critic / for offline analysis.

What is randomised, and what is not:

* **Fixed for the whole run:** the maze layout.  It is generated once, host-side,
  in ``__init__`` from ``maze_seed``.  A fixed maze is identical across all
  worlds, so no per-world model batching is needed -- which is what makes this
  task tractable on the warp backend at all (``Model``-side per-world edits such
  as ``hfield_data`` / ``geom_pos`` are *silently ignored* there).
* **Randomised every episode:** the rodent spawn cell + heading and every treat
  cell, sampled without replacement from the maze's free cells.  All of that
  lives in ``Data`` (root free-joint ``qpos`` and treat slide-joint ``qpos``),
  which is the only per-world route that works.

Walls are box geoms, not a heightfield: a heightfield can only make ramps and
the rodent *will* climb them, whereas boxes give true vertical occluders and are
much cheaper to collide against.

Dynamic geometry follows ``run_gap.py`` exactly: each treat is a body carrying
three slide joints (x, y, z) with ``damping=1e8, stiffness=0``, whose ``qpos`` is
written at ``reset()`` and followed by a mandatory ``mjx.forward``.

.. warning::

   **Slide joints shift the qpos/qvel layout.**  ``RodentEnv``'s proprioception
   getters hard-code ``qpos[7:]`` / ``qvel[6:]`` / the full ``qfrc_actuator``.
   The treat bodies are therefore added *before* ``add_rodent()`` and this class
   overrides ``_get_joint_angles`` / ``_get_joint_ang_vels`` /
   ``_get_actuator_ctrl`` against cached offsets, exactly as ``run_gap`` does.
   Getting this wrong silently corrupts proprioception instead of raising.

.. warning::

   **``info`` does not reset across auto-resets.**  ``BraxAutoResetWrapper``
   with ``full_reset=False`` swaps ``data``/``obs`` back but leaves
   ``state.info`` alone, so the ``collected`` bitmask would ratchet across
   episodes and every treat would be collectable exactly once per *environment*,
   forever.  Wrap with ``wrappers_info_reset.InfoResetOnDoneWrapper`` and pass
   :data:`INFO_RESET_KEYS`.

Usage::

    env = MazeForageVision(config=default_config())
    state = env.reset(rng)
    state = env.step(state, action)
"""

import collections
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from vnl_playground.tasks.reward_registry import RewardRegistry
from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.rodent import maze_utils

_registry = RewardRegistry()

_WALL_MATERIAL = "maze_wall_mat"
_WALL_TEXTURE = "maze_wall_tex"
_TREAT_MATERIAL = "maze_treat_mat"

# Walls are sunk this far into the floor so there is no light-leaking seam
# between the bottom of a wall box and the arena floor plane.
_WALL_FLOOR_EMBED = 0.01

# Per-episode ``state.info`` keys written by :meth:`MazeForageVision.reset`.
# Pass these to ``InfoResetOnDoneWrapper`` -- see the module docstring.
INFO_RESET_KEYS = (
    "prev_action",
    "action",
    "step_count",
    "collected",
    "n_collected",
)


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the MazeForageVision environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.  Every key
        a config file may override has to exist here: ``MjxEnv.__init__`` locks
        the dict, so unknown keys raise at construction time.
    """
    return config_dict.create(
        walker_xml_path=consts.RODENT_NO_TAIL_COLLISION_XML,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.02,
        sim_dt=0.002,
        solver="newton",
        mujoco_impl="warp",
        naconmax=20 * 1024,
        njmax=400,
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        # Episode limits (the real truncation is EpisodeWrapper's; see the
        # `timeout` termination for why it is not on by default).
        episode_length=1000,
        action_repeat=1,
        # --- Maze ---
        maze_cells=3,  # logical cells per side -> (2n+1) x (2n+1) grid
        cell_size=0.35,  # m, corridor width
        wall_height=0.3,  # m, tall enough to occlude and not be climbed
        wall_thickness=0.05,  # m, in-plane wall thickness (< cell_size)
        maze_seed=0,  # fixed for the whole run
        maze_loop_fraction=0.0,  # >0 knocks out walls to create loops
        # --- Treats ---
        n_treats=4,
        treat_radius=0.03,  # m, sphere radius
        treat_height=0.05,  # m, world z of a live treat's centre
        treat_reach_threshold=0.1,  # m, xy distance that counts as "reached"
        park_depth=1.0,  # m below the floor a collected treat slides to
        # --- Spawn ---
        spawn_height=0.005,  # m of clearance above the floor at spawn
        # --- Vision (rendered by VisionRenderWrapper, not by this env) ---
        vision=True,
        vision_width=64,
        vision_height=64,
        grayscale=True,
        binocular=False,
        vision_camera_name="egocentric-rodent",
        render_depth=False,
        use_textures=True,
        use_shadows=False,
        # --- Reward terms ---
        reward_terms={
            "treat_collected": {"weight": 1.0},
        },
        # --- Termination criteria ---
        termination_criteria={
            "fallen": {"min_torso_z": 0.01, "max_torso_angle": 70},
            "all_treats_collected": {},
            "nan_termination": {},
        },
    )


class MazeForageVision(rodent_base.RodentEnv):
    """Sparse, vision-only maze foraging.

    The maze is built once at construction; only ``Data`` changes per episode.
    See the module docstring for the observation contract and the two gotchas
    (qpos shift, ``info`` ratchet).
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initializes the MazeForageVision environment.

        Args:
            rng: Random number generator key (kept for API parity with the other
                rodent tasks; the maze itself is seeded by ``config.maze_seed``).
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.

        Raises:
            ValueError: If ``mujoco_impl`` is not ``"warp"`` (the vision renderer
                requires it), or if the maze has too few free cells to place the
                spawn plus every treat.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        if self._config.mujoco_impl != "warp":
            raise ValueError(
                "MazeForageVision requires mujoco_impl='warp' for rendering"
            )

        self._vision_width = int(self._config.vision_width)
        self._vision_height = int(self._config.vision_height)
        self._grayscale = bool(self._config.get("grayscale", False))

        self._n_treats = int(self._config.n_treats)
        self._cell_size = float(self._config.cell_size)
        self._treat_height = float(self._config.treat_height)
        self._park_depth = float(self._config.park_depth)
        self._treat_reach_threshold = float(self._config.treat_reach_threshold)
        self._spawn_height = float(self._config.spawn_height)

        # --- Host-side maze construction (runs once, never under trace) ---
        self._maze_grid = maze_utils.generate_maze(
            maze_cells=int(self._config.maze_cells),
            seed=int(self._config.maze_seed),
            loop_fraction=float(self._config.maze_loop_fraction),
        )
        self._maze_walls = maze_utils.make_walls(self._maze_grid)
        free_xy = maze_utils.free_cells(self._maze_grid, self._cell_size)
        if free_xy.shape[0] < self._n_treats + 1:
            raise ValueError(
                f"Maze has {free_xy.shape[0]} free cells but the task needs "
                f"{self._n_treats + 1} distinct ones (1 spawn + "
                f"{self._n_treats} treats). Increase maze_cells or lower "
                "n_treats."
            )
        self._free_cell_xy_np = free_xy
        self._free_cell_xy = jp.array(free_xy, dtype=jp.float32)
        self._n_free_cells = int(free_xy.shape[0])

        half_x, half_y = maze_utils.maze_extent(self._cell_size, self._maze_grid.shape)
        self._maze_half_extent = (float(half_x), float(half_y))
        # One generous symmetric range shared by all three treat slide axes.
        # MuJoCo does NOT clamp a directly written qpos to a joint range -- it
        # fights it with a limit constraint instead -- so the range has to cover
        # every position we ever write, including the parked depth.
        self._slide_range = float(
            max(half_x, half_y, self._park_depth + self._treat_height)
            + self._cell_size
        )

        self._add_materials()
        self._build_maze()
        # Treat bodies MUST be added before the rodent: their slide joints then
        # occupy the LOW qpos/qvel addresses and the open-ended proprioception
        # slices below stay correct (see the class overrides).
        self._build_treats()

        # Spawn pose is overwritten every reset(); the attach pose only sets the
        # free joint's qpos0.
        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[0.0, 0.0, self._spawn_height],
            quat=(1, 0, 0, 0),
        )

        # Directional key light: the warp renderer ignores light attenuation and
        # needs a directional source for consistent illumination (cf. run_gap).
        self._spec.worldbody.add_light(
            name="key_light",
            pos=[0, 0, 8],
            dir=[-0.1, -0.1, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
            castshadow=1,
        )
        self._spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self._spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        self._spec.visual.headlight.specular = [0.1, 0.1, 0.1]

        self.compile()

        # --- Post-compile index caching (host-side, never under trace) ---
        self._cache_treat_indices()
        self._cache_rodent_qpos_layout()

    # ------------------------------------------------------------------
    # Arena construction (host-side, all of it before compile())
    # ------------------------------------------------------------------

    def _add_materials(self) -> None:
        """Registers the wall and treat materials on the arena spec."""
        self._spec.add_texture(
            name=_WALL_TEXTURE,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=256,
            height=256,
            rgb1=[0.30, 0.30, 0.34],
            rgb2=[0.55, 0.55, 0.60],
        )
        wall_mat = self._spec.add_material(name=_WALL_MATERIAL)
        wall_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _WALL_TEXTURE
        wall_mat.texrepeat = [2, 2]
        wall_mat.texuniform = True
        wall_mat.reflectance = 0.0

        # Treats are deliberately much brighter than anything else in the scene
        # so they survive the grayscale conversion the renderer applies.
        treat_mat = self._spec.add_material(name=_TREAT_MATERIAL)
        treat_mat.rgba = [1.0, 0.85, 0.1, 1.0]
        treat_mat.emission = 0.4
        treat_mat.reflectance = 0.0

    def _wall_box_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes ``(pos, size)`` for one box geom per covering rectangle.

        ``maze_utils.wall_boxes`` gives dm_control's geometry, where every wall
        cell is filled edge to edge.  With ``cell_size=0.35`` that would make the
        walls as thick as the corridors are wide, so each rectangle is thinned to
        ``wall_thickness`` along any axis it spans a *single* cell on.

        Thinning alone would leave diagonal holes where a thin wall meets a
        perpendicular one, so each side of a rectangle is then extended by
        ``cell_size - wall_thickness`` **iff every grid cell just beyond that
        side is also a wall**.  That extension always reaches the neighbouring
        rectangle's near face and never reaches past the neighbouring wall
        cell, so it can seal a junction but can never intrude into a corridor.
        Overlapping wall boxes are free: they all live on ``worldbody``, and
        MuJoCo never generates contact pairs within one body.

        Returns:
            ``(pos, size)``, each ``(n_walls, 3)``; ``size`` is MuJoCo's
            half-extent convention.
        """
        grid = self._maze_grid
        pos, size = maze_utils.wall_boxes(
            self._maze_walls,
            self._cell_size,
            grid.shape,
            float(self._config.wall_height),
            z_offset=-_WALL_FLOOR_EMBED,
        )

        thickness = float(self._config.wall_thickness)
        if not 0.0 < thickness < self._cell_size:
            # Degenerate/oversized thickness: fall back to dm_control's
            # full-cell walls rather than producing a leaky maze.
            return pos, size

        seal = self._cell_size - thickness
        is_wall = grid == maze_utils.WALL_CHAR
        height, width = grid.shape

        for i, wall in enumerate(self._maze_walls):
            y0, y1 = int(wall.start.y), int(wall.end.y)
            x0, x1 = int(wall.start.x), int(wall.end.x)

            # Thin any axis the rectangle is one cell wide on.
            if x1 - x0 == 1:
                size[i, 0] = thickness / 2.0
            if y1 - y0 == 1:
                size[i, 1] = thickness / 2.0

            # World +x follows increasing column, world +y follows DECREASING
            # row (dm_control convention, see maze_utils.grid_to_world).
            sides = (
                (0, +1.0, is_wall[y0:y1, x1] if x1 < width else None),
                (0, -1.0, is_wall[y0:y1, x0 - 1] if x0 > 0 else None),
                (1, +1.0, is_wall[y0 - 1, x0:x1] if y0 > 0 else None),
                (1, -1.0, is_wall[y1, x0:x1] if y1 < height else None),
            )
            for axis, sign, neighbours in sides:
                if neighbours is not None and np.all(neighbours):
                    size[i, axis] += seal / 2.0
                    pos[i, axis] += sign * seal / 2.0

        return pos, size

    def _build_maze(self) -> None:
        """Emits the static maze walls as box geoms on ``worldbody``."""
        pos, size = self._wall_box_geometry()
        for i in range(pos.shape[0]):
            self._spec.worldbody.add_geom(
                name=f"maze_wall_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(pos[i]),
                size=list(size[i]),
                material=_WALL_MATERIAL,
                contype=1,
                conaffinity=1,
            )

    def _build_treats(self) -> None:
        """Adds the slide-jointed treat bodies (must precede ``add_rodent``).

        Each treat is a non-colliding sphere (``contype=0, conaffinity=0``) --
        a pure trigger volume, read with a distance check rather than a contact
        query -- carried by three slide joints.  ``damping=1e8, stiffness=0`` is
        what pins it: terminal velocity under gravity is ``mg/damping`` and zero
        stiffness means nothing drags it back toward ``qpos0``.

        The bodies are parked at the maze centre at compile time; ``reset()``
        writes the real position into the slide joints, so the compile-time pose
        is never observed.
        """
        radius = float(self._config.treat_radius)
        for i in range(self._n_treats):
            body = self._spec.worldbody.add_body(
                name=f"treat_{i}",
                pos=[0.0, 0.0, self._treat_height],
            )
            for axis, axis_name in (
                ((1, 0, 0), "x"),
                ((0, 1, 0), "y"),
                ((0, 0, 1), "z"),
            ):
                body.add_joint(
                    name=f"treat_{i}_slide_{axis_name}",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=list(axis),
                    range=[-self._slide_range, self._slide_range],
                    damping=1e8,
                    stiffness=0,
                )
            body.add_geom(
                name=f"treat_{i}_geom",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0.0, 0.0],
                material=_TREAT_MATERIAL,
                contype=0,
                conaffinity=0,
            )

    # ------------------------------------------------------------------
    # Post-compile index caching
    # ------------------------------------------------------------------

    def _cache_treat_indices(self) -> None:
        """Caches treat slide-joint qpos addresses and treat body ids.

        Raises:
            ValueError: If a treat joint or body name is missing.  ``mj_name2id``
                returns ``-1`` rather than raising, and ``-1`` would silently
                index the *last* joint (i.e. one of the rodent's).
        """
        qpos_idxs = np.zeros((self._n_treats, 3), dtype=np.int32)
        body_ids = np.zeros((self._n_treats,), dtype=np.int32)
        for i in range(self._n_treats):
            for a, axis_name in enumerate(("x", "y", "z")):
                name = f"treat_{i}_slide_{axis_name}"
                jnt_id = mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                if jnt_id < 0:
                    raise ValueError(f"Joint '{name}' not found in compiled model.")
                qpos_idxs[i, a] = self._mj_model.jnt_qposadr[jnt_id]
            body_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_BODY, f"treat_{i}"
            )
            if body_id < 0:
                raise ValueError(f"Body 'treat_{i}' not found in compiled model.")
            body_ids[i] = body_id

        self._treat_slide_qpos_idxs_np = qpos_idxs
        self._treat_slide_qpos_idxs = jp.array(qpos_idxs)
        self._treat_z_qpos_idxs = jp.array(qpos_idxs[:, 2])
        self._treat_body_ids = jp.array(body_ids)

    def _cache_rodent_qpos_layout(self) -> None:
        """Caches where the rodent's own dofs start in ``qpos`` / ``qvel``.

        The ``n_treats * 3`` slide joints prepend elements to ``qpos``/``qvel``,
        and the base-class proprioception getters assume the rodent's motor
        joints start at ``qpos[7:]`` / ``qvel[6:]``.  Same fix as
        ``run_gap.py:183-195``.

        Raises:
            ValueError: If the rodent root free joint is missing.
        """
        root_jnt_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        if root_jnt_id < 0:
            raise ValueError("Rodent root free joint 'root' not found.")
        self._rodent_root_qpos = int(self._mj_model.jnt_qposadr[root_jnt_id])
        # Motor joints start right after the 7-element free joint.
        self._rodent_qpos_start = self._rodent_root_qpos + 7
        self._rodent_qvel_start = int(self._mj_model.jnt_dofadr[root_jnt_id]) + 6
        # Root joint DOF address (for qfrc_actuator slicing -- note this KEEPS
        # the root's 6 dofs, matching the base class's full qfrc_actuator).
        self._rodent_root_dof = int(self._mj_model.jnt_dofadr[root_jnt_id])

    # ------------------------------------------------------------------
    # Core environment interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Samples a fresh layout: spawn cell + heading and all treat cells.

        Everything here is traceable: the free-cell table is a host-side array
        built in ``__init__`` and only *indexed* with traced indices, so this
        survives ``jax.jit``, ``jax.vmap`` and the ``jax.eval_shape`` call that
        ``observation_size`` makes.

        Args:
            rng: Random number generator key.

        Returns:
            The initial environment state.
        """
        rng, cell_rng, yaw_rng = jax.random.split(rng, 3)

        # One draw without replacement gives distinct cells for the spawn and
        # every treat, and automatically excludes the spawn cell from the treats.
        cell_idx = jax.random.choice(
            cell_rng,
            self._n_free_cells,
            shape=(self._n_treats + 1,),
            replace=False,
        )
        cells = self._free_cell_xy[cell_idx]
        spawn_xy, treat_xy = cells[0], cells[1:]

        yaw = jax.random.uniform(yaw_rng, minval=-jp.pi, maxval=jp.pi)
        half_yaw = 0.5 * yaw
        spawn_quat = jp.array(
            [jp.cos(half_yaw), 0.0, 0.0, jp.sin(half_yaw)], dtype=jp.float32
        )

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "step_count": jp.array(0, dtype=jp.int32),
            "collected": jp.zeros((self._n_treats,), dtype=bool),
            "n_collected": jp.array(0, dtype=jp.int32),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        qpos = data.qpos
        # Rodent root free joint: position then yaw quaternion.  qpos0 already
        # holds the attach-frame pose, so both halves must be overwritten.
        root = self._rodent_root_qpos
        spawn_pos = jp.array(
            [spawn_xy[0], spawn_xy[1], jp.float32(self._spawn_height)]
        )
        qpos = qpos.at[root : root + 3].set(spawn_pos)
        qpos = qpos.at[root + 3 : root + 7].set(spawn_quat)
        # Treat slide joints: offsets from each body's reference pose, which is
        # (0, 0, treat_height), so the xy offset IS the target xy.
        treat_offsets = jp.concatenate(
            [treat_xy, jp.zeros((self._n_treats, 1))], axis=-1
        )
        qpos = qpos.at[self._treat_slide_qpos_idxs].set(treat_offsets)
        data = data.replace(qpos=qpos)

        # MANDATORY: make_data leaves xpos at the compile-time pose, so every
        # xpos-based reward / obs / termination would read the parked layout.
        data = mjx.forward(self.mjx_model, data)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Steps physics, then scores and updates the collected mask.

        Ordering mirrors ``go_to_target``: observation, termination and reward
        are all computed against the mask as it stood *entering* this step, and
        the mask is advanced afterwards -- so the step on which the last treat is
        reached pays out, and ``all_treats_collected`` fires one step later.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            The next environment state.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action
        info["step_count"] = info["step_count"] + 1

        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        # --- Collected-mask update (AFTER reward) ---
        collected = jp.logical_or(info["collected"], self._treats_in_reach(data))
        info["collected"] = collected
        info["n_collected"] = jp.sum(collected).astype(jp.int32)

        # Collected treats slide underground so they vanish from the camera and
        # cannot re-trigger.  Only qpos is written here; xpos catches up on the
        # next physics step, i.e. a treat disappears one control step after it
        # is collected.
        data = self._park_collected_treats(data, collected)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> collections.OrderedDict:
        """Builds the observation tree.

        ``task_obs`` is ``[prev_action, kinematic_sensors, touch_sensors,
        origin]`` and carries **no** treat information -- that is the entire
        point of the task, so do not add an ``ego_target`` here.  ``vision`` is a
        zeros placeholder that ``VisionRenderWrapper`` overwrites with real
        pixels; drop it and the wrapper silently no-ops.

        Args:
            data: Simulation data.
            info: State info (must contain ``prev_action`` and ``collected``).

        Returns:
            ``OrderedDict(state=..., privileged_state=...)``.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
            ]
        )

        proprioception = self._get_proprioception(data, info, flatten=False)
        vision = jp.zeros(self.vision_shape)

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=proprioception,
            vision=vision,
        )

        # The critic (and offline analysis) may see where the treats are; the
        # policy may not.  'task_obs' must stay present here -- HighLevelWrapper
        # indexes privileged_state[highlvl_obs_key].
        privileged_obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=proprioception,
            vision=vision,
            treat_vectors=self._egocentric_treat_vectors(data).ravel(),
            collected=info["collected"].astype(jp.float32),
        )

        return collections.OrderedDict(
            state=obs,
            privileged_state=privileged_obs,
        )

    # ------------------------------------------------------------------
    # Treat helpers
    # ------------------------------------------------------------------

    def _torso(self, data: mjx.Data):
        """Binds the rodent torso body."""
        return data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))

    def _treat_distances(self, data: mjx.Data) -> jp.ndarray:
        """Horizontal (xy) distance from the torso to each treat, shape ``(n,)``."""
        torso_xy = self._torso(data).xpos[:2]
        treat_xy = data.xpos[self._treat_body_ids, :2]
        return jp.linalg.norm(treat_xy - torso_xy, axis=-1)

    def _treats_in_reach(self, data: mjx.Data) -> jp.ndarray:
        """Boolean ``(n,)`` mask of treats within ``treat_reach_threshold``.

        A parked (collected) treat keeps its xy, so this stays True for it --
        harmless, because the mask update is monotone and the reward is gated on
        the *previous* mask.
        """
        return self._treat_distances(data) < self._treat_reach_threshold

    def _egocentric_treat_vectors(self, data: mjx.Data) -> jp.ndarray:
        """Treat positions in the torso frame, shape ``(n_treats, 3)``."""
        torso = self._torso(data)
        return jp.dot(data.xpos[self._treat_body_ids] - torso.xpos, torso.xmat)

    def _park_collected_treats(
        self, data: mjx.Data, collected: jp.ndarray
    ) -> mjx.Data:
        """Writes the z slide offset that hides every collected treat.

        Args:
            data: Simulation data.
            collected: Boolean ``(n_treats,)`` mask.

        Returns:
            ``data`` with the treat z slide qpos rewritten (idempotent).
        """
        parked_offset = -(self._park_depth + self._treat_height)
        z_offsets = jp.where(collected, parked_offset, 0.0)
        qpos = data.qpos.at[self._treat_z_qpos_idxs].set(z_offsets)
        return data.replace(qpos=qpos)

    # ------------------------------------------------------------------
    # Proprioception overrides (qpos/qvel shift from the treat slide joints)
    # ------------------------------------------------------------------
    # The treat bodies are added before the rodent, so their 3 * n_treats slide
    # joints occupy the low qpos/qvel addresses and these open-ended slices land
    # exactly on the rodent's own dofs.  Sizes are unchanged from a bare rodent:
    # 67 / 67 / 73.

    def _get_joint_angles(self, data: mjx.Data) -> jp.ndarray:
        return data.qpos[self._rodent_qpos_start :]

    def _get_joint_ang_vels(self, data: mjx.Data) -> jp.ndarray:
        return data.qvel[self._rodent_qvel_start :]

    def _get_actuator_ctrl(self, data: mjx.Data) -> jp.ndarray:
        # Slices from the root DOF, NOT root + 6: the base class returns the
        # FULL qfrc_actuator (length nv), root dofs included.
        return data.qfrc_actuator[self._rodent_root_dof :]

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    @_registry.reward("treat_collected")
    def _treat_collected_reward(self, data, info, metrics, weight) -> float:
        """Sparse ``+weight`` per treat newly reached on this step.

        Reach is an xy distance test against ``config.treat_reach_threshold``,
        not a contact query: the treat geoms are non-colliding trigger volumes.

        Args:
            data: Simulation data.
            info: State info; ``collected`` gates already-taken treats.
            metrics: Metrics dict for logging.
            weight: Reward per treat.

        Returns:
            ``weight * (number of treats newly collected this step)``.
        """
        collected = info["collected"]
        dists = self._treat_distances(data)
        newly = jp.logical_and(
            dists < self._treat_reach_threshold, jp.logical_not(collected)
        )
        n_new = jp.sum(newly.astype(jp.float32))
        reward_val = weight * n_new

        # Distance to the nearest treat that is still up for grabs.
        remaining = jp.logical_not(collected)
        nearest = jp.min(jp.where(remaining, dists, jp.inf))
        nearest = jp.where(jp.any(remaining), nearest, 0.0)

        metrics["rewards/treat_collected"] = reward_val
        metrics["rewards/n_treats_collected"] = jp.sum(collected.astype(jp.float32))
        metrics["rewards/nearest_treat_distance"] = nearest
        return reward_val

    @_registry.reward("termination_penalty")
    def _termination_penalty(self, data, info, metrics, weight) -> float:
        """Negative reward on the timestep the episode terminates.

        Reads ``terminations/any``, which ``_is_done`` writes -- so it is
        structurally zero during ``reset()`` (reward runs first there).

        Args:
            data: Simulation data (unused).
            info: State info (unused).
            metrics: Metrics dict; reads ``terminations/any``.
            weight: Penalty magnitude (positive; applied as negative reward).

        Returns:
            Weighted termination penalty.
        """
        del data, info
        terminated = metrics.get("terminations/any", 0.0)
        penalty = -weight * terminated
        metrics["rewards/termination_penalty"] = penalty
        return penalty

    # ------------------------------------------------------------------
    # Termination criteria
    # ------------------------------------------------------------------

    @_registry.termination("all_treats_collected")
    def _all_treats_collected_termination(self, data, info) -> bool:
        """Terminate once every treat has been collected."""
        del data
        return jp.all(info["collected"])

    @_registry.termination("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = 0.01,
        max_torso_angle: float = 70,
    ) -> bool:
        """Terminate if the torso drops too low or tips too far.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum torso world z.
            max_torso_angle: Maximum tilt from vertical, in degrees.

        Returns:
            Boolean indicating whether the rodent has fallen.
        """
        del info
        torso = self._torso(data)
        below_ground = torso.xpos[2] < min_torso_z
        # xmat[-1, -1] is cos(angle between the torso z axis and world z).
        too_tilted = torso.xmat[-1, -1] < np.cos(np.deg2rad(max_torso_angle))
        return jp.logical_or(below_ground, too_tilted)

    @_registry.termination("timeout")
    def _timeout_termination(self, data, info, max_steps=1000) -> bool:
        """Terminate if the step count exceeds ``max_steps``.

        Registered but deliberately NOT in ``default_config``'s
        ``termination_criteria``: ``info["step_count"]`` is not cleared by
        ``BraxAutoResetWrapper(full_reset=False)``, so once it latches past
        ``max_steps`` the env collapses into one-step episodes.  Truncation is
        ``EpisodeWrapper``'s job (``train_config.episode_length``).  Only enable
        this alongside ``InfoResetOnDoneWrapper`` with ``step_count`` listed in
        its keys (:data:`INFO_RESET_KEYS` includes it).

        Args:
            data: Simulation data (unused).
            info: State info containing ``step_count``.
            max_steps: Step budget.

        Returns:
            Boolean indicating whether the budget is exhausted.
        """
        del data
        return info["step_count"] >= max_steps

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Terminate on NaN values in simulation data."""
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        return jp.sum(jp.isnan(flattened_vals)) > 0

    # ------------------------------------------------------------------
    # Utility methods and observation-size contract
    # ------------------------------------------------------------------

    def null_action(self) -> jp.ndarray:
        """Zero action of the correct size."""
        return jp.zeros(self.action_size)

    @property
    def maze_grid(self) -> np.ndarray:
        """Copy of the fixed maze character grid (for tests / rendering)."""
        return self._maze_grid.copy()

    @property
    def maze_walls(self) -> tuple:
        """The covering rectangles the wall geoms were built from."""
        return self._maze_walls

    @property
    def free_cell_positions(self) -> np.ndarray:
        """``(M, 2)`` world xy of the maze's open cells (host-side copy)."""
        return self._free_cell_xy_np.copy()

    @property
    def n_treats(self) -> int:
        """Number of treats in the maze."""
        return self._n_treats

    @property
    def vision_shape(self) -> Tuple[int, int, int]:
        """Shape of the vision observation: ``(H, W, C)``, doubled if binocular."""
        mono_channels = 1 if self._grayscale else 3
        channels = (
            2 * mono_channels
            if self._config.get("binocular", False)
            else mono_channels
        )
        return (self._vision_height, self._vision_width, channels)

    @property
    def vision_enabled(self) -> bool:
        """Whether vision observations are enabled."""
        return True

    @property
    def vision_obs_size(self) -> int:
        """Number of pixel elements in one vision observation."""
        h, w, c = self.vision_shape
        return h * w * c

    @property
    def observation_size(self) -> int:
        """Flat size of ``task_obs`` + ``proprioception``.

        Deliberately excludes the vision pixels and ``privileged_state`` -- the
        base implementation sums the whole obs tree, which double-counts here.
        """
        obs_size = self.non_flattened_observation_size
        total = 0
        for key in ("task_obs", "proprioception"):
            total += jp.sum(flatten_util.ravel_pytree(obs_size["state"][key])[0])
        return total

    @property
    def proprioceptive_obs_size(self) -> int:
        """Flat size of the proprioception subtree (277 for a bare rodent)."""
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0]
        )
