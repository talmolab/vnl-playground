"""Vision-guided sparse maze-foraging task for the virtual rodent.

The rodent is dropped into a **fixed** box-walled maze and must find and reach
``n_treats`` treats scattered through it.  Each treat pays ``+weight`` the first
time the rodent gets within ``treat_reach_threshold`` of it (in xy), then slides
underground so it can neither be seen nor re-collected.  The episode ends when
every treat has been collected, or on a fall / NaN (dm_control ``ManyGoalsMaze``
semantics).

What makes this task *pure vision*:
``task_obs`` is ``[prev_action, kinematic_sensors, touch_sensors]`` (plus an
optional ``origin``, **off** by default -- see ``config.include_origin``) and
carries **no** treat vector.  Every other vision task in this repo leaks an
egocentric target vector into ``task_obs``; this one deliberately does not, so
the only channel that can tell the policy where a treat is, is the egocentric
camera image (a zeros placeholder here, filled in by ``VisionRenderWrapper``).
``privileged_state`` also carries egocentric treat vectors and the collected
mask.  Read the next paragraph before relying on that: with the shipped
``arch_name: shared_vision_task_obs`` those two entries reach **nothing** --
``tasks.wrappers.HighLevelWrapper._process_state`` rebuilds the network
observation from ``obs['state']`` alone in its vision+task_obs branch and drops
the whole ``privileged_state`` subtree, so the value function is exactly as
blind as the policy.  They are kept because they are the natural offline-
analysis channel (rollout scripts read them straight off the state) and because
the MLP arch's ``privileged_state['task_obs']`` path still needs the key to
exist.  If you want a genuinely asymmetric critic you have to teach the network
factory to consume ``privileged_state`` first -- do not assume it already does.

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

  "Every episode" is a property of ``reset()``, and it only survives into
  training if the auto-reset actually *calls* ``reset()`` -- see the
  ``full_reset`` warning below.  This env therefore sets
  :attr:`MazeForageVision.requires_per_episode_reset`, and
  ``train_highlvl.py`` refuses to launch it without ``wrappers.full_reset``.

Walls are box geoms, not a heightfield: a heightfield can only make ramps and
the rodent *will* climb them, whereas boxes give true vertical occluders and are
much cheaper to collide against.

Appearance is a training concern, not a cosmetic one.  The warp ray-tracer
shades a surface by ``N.L`` over the *directional* lights only and ignores
``light.ambient`` outright, so an overhead key light alone leaves every vertical
wall face black in the policy's own camera.  Measured over 16 reset frames of
the shipped 64x64 grayscale egocentric view, fraction of pixels below 0.15::

  builtin checker texture, key light only          73.6%   (mean 0.183)
  labmaze style_03/gray_bright, key light only     43.4%   (mean 0.229)
  labmaze style_03/gray_bright + 4 wall lights     11.8%   (mean 0.653)

Each row changes exactly one factor from the row above it, so both the texture
and the light ring are separately load-bearing.  See ``config.wall_texture``,
``config.wall_lights`` and :meth:`MazeForageVision._add_wall_lights`.

Sizing (all of it derived from one number):

``config.maze_extent`` is the **outer extent of the maze footprint in metres**
and is what stays fixed -- the arena is ``maze_extent`` x ``maze_extent``,
centred on the world origin, for *every* value of ``maze_cells``.  The grid is
``(2 * maze_cells + 1)`` square, so the grid pitch is

    ``cell_size = maze_extent / (2 * maze_cells + 1)``

and ``config.cell_size`` is ``None`` (= derive) by default.  Turning
``maze_cells`` up therefore buys more maze structure at the cost of narrower
corridors, and never changes the arena.  ``__init__`` cross-checks
``grid_size * cell_size`` against ``maze_extent`` and raises if they disagree;
:attr:`MazeForageVision.maze_extent` reports the realised extent.

.. note::

   ``cell_size`` is the **grid pitch**, not the corridor width.  Wall
   rectangles are thinned to ``wall_thickness`` (see
   :meth:`MazeForageVision._wall_box_geometry`), which hands back
   ``cell_size - wall_thickness`` of floor on each side of a corridor, so the
   clear width between two parallel walls is
   ``2 * cell_size - wall_thickness``.
   :attr:`MazeForageVision.corridor_width` reports it.

   At the defaults (``maze_extent=6.46``, ``maze_cells=10`` -> 21x21 grid,
   ``cell_size = 6.46/21 = 0.307619``, ``wall_thickness=0.15``) that is
   **0.4652 m**, against a rat measured at ``rescale_factor=0.9`` as 0.308 m
   long x 0.080 m wide x 0.092 m tall (AABB over the rodent geoms at the
   compile pose) -- so a corridor clears the body *length* and the rat can turn
   around without relying on spine flexion.  Same 6.46 m arena, same
   ``wall_thickness``, measured off the compiled box geoms (``maze_seed=0``)::

     maze_cells   grid     cell_size   corridor   measured min   free cells
     ----------   -----    ---------   --------   ------------   ----------
     6            13x13    0.496923    0.8438     0.6703          71
     8            17x17    0.380000    0.6100     0.4948         127
     10 (default) 21x21    0.307619    0.4652     0.3862         199
     12           25x25    0.258400    0.3668     0.2585         287
     14           29x29    0.222759    0.2955     0.2228         391

   ``corridor`` is the formula above (and the widest measured corridor);
   ``measured min`` is the narrowest one-cell corridor in that maze.  Two
   bounded effects make the minimum smaller and neither is a bug: a flanking
   wall rectangle spanning several cells along the scan axis is not thinned on
   that axis, and a perpendicular rectangle seal-extended into a T-junction
   pokes a corner nub into the corridor (see
   :meth:`MazeForageVision._wall_box_geometry`).  The nub only reaches past the
   flanking wall's inner face while ``0.5 * cell_size > 1.5 * wall_thickness``,
   which the shipped 0.15 m walls are *not*: at the defaults the narrowest
   corridor is set by the un-thinned multi-cell flanks alone.  The floor is one
   ``cell_size``.

   From ``maze_cells=12`` on the corridor is narrower than the rat is long, so
   a turn-in-place needs body flexion.  ``maze_cells`` 8 / 10 / 12 are all built
   and checked by
   ``tests/test_maze_forage_vision.py::test_parameterised_maze_sizes_build_and_are_navigable``.

.. note::

   The **sealed enclosure is smaller than the footprint**: the border walls are
   thinned and sit on their border-cell centres, so the free space the rat can
   reach is ``maze_extent - cell_size - wall_thickness`` = 6.002 x 6.002 m at
   the defaults (symmetric about the origin, verified sealed by flood filling a
   1 mm occupancy raster).  ``maze_extent`` is the grid footprint, which is what
   the wall geometry, ``free_cells`` and the render harness are all built on.

.. note::

   ``n_treats`` does **not** scale with the maze.  At the defaults 20 treats are
   hidden among 199 reachable cells (:attr:`treat_cell_fraction` = 0.101, vs
   0.157 at ``maze_cells=8`` and 0.070 at ``maze_cells=12``), so raising
   ``maze_cells`` at fixed ``n_treats`` makes an already-sparse exploration
   problem strictly harder.  Deliberately left as a human decision.

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

   **``BraxAutoResetWrapper(full_reset=False)`` breaks this task in two ways.**
   In that mode the auto-reset restores ``data``/``obs`` from the *first*
   reset and never touches ``state.info``, so ``env.reset`` is called exactly
   once per env index for the whole run.  Measured on the real stack (batch 3,
   ``episode_length=4``): after ``done`` the treat ``xpos`` and the 7-element
   root ``qpos`` are bit-identical to episode 1, and ``collected`` stays
   all-True -- the env then reports ``done=1, reward=0`` on step 1 of every
   subsequent episode, forever (the ratchet shape that invalidated eight DMPO
   gap-jump runs).

   The fix used by the entry point is ``full_reset=True``, which calls
   ``env.reset`` on ``done``: measured on the same stack it re-randomises the
   layout *and* returns ``collected`` to all-False, for ~7% throughput at
   batch 64 (1770 -> 1649 env-steps/s on a contended 5090).  Enable it with a
   top-level ``wrappers: {full_reset: true}`` block; ``train_highlvl.py``
   raises if it is missing.  ``wrappers_info_reset.InfoResetOnDoneWrapper``
   with :data:`INFO_RESET_KEYS` fixes *only* the ``info`` half and leaves the
   layout frozen, so it is not sufficient on its own.

Usage::

    env = MazeForageVision(config=default_config())
    state = env.reset(rng)
    state = env.step(state, action)
"""

import collections
import os
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
_SKY_TEXTURE = "maze_sky_tex"
_FLOOR_TEXTURE = "maze_floor_tex"
_FLOOR_MATERIAL = "maze_floor_mat"

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
    "belly_up_steps",
    "low_torso_steps",
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
        # Outer extent of the maze footprint, in metres.  The arena is
        # maze_extent x maze_extent, centred on the world origin, for EVERY
        # value of maze_cells: cell_size is derived from it, so turning
        # maze_cells up narrows the corridors instead of growing the arena.
        maze_extent=6.46,
        maze_cells=10,  # logical cells per side -> (2n+1) x (2n+1) grid
        # GRID PITCH, not corridor width, and normally DERIVED: None means
        # maze_extent / (2 * maze_cells + 1) = 0.307619 m at the defaults.
        # Set it explicitly only alongside a consistent maze_extent -- the two
        # are cross-checked in __init__ and any disagreement raises.  The clear
        # width between two parallel walls is 2 * cell_size - wall_thickness
        # (0.4652 m at the defaults); see the corridor_width property.
        cell_size=None,
        wall_height=0.3,  # m, tall enough to occlude and not be climbed
        wall_thickness=0.15,  # m, in-plane wall thickness (< cell_size)
        maze_seed=0,  # fixed for the whole run
        maze_loop_fraction=0.0,  # >0 knocks out walls to create loops
        # --- Treats ---
        n_treats=20,
        treat_radius=0.05,  # m, sphere radius
        treat_height=0.125,  # m, world z of a live treat's centre
        treat_reach_threshold=0.1,  # m, xy distance that counts as "reached"
        park_depth=1.0,  # m below the floor a collected treat slides to
        # --- Spawn ---
        spawn_height=0.005,  # m of clearance above the floor at spawn
        # --- Observation ---
        # `origin` is the world origin expressed in the torso frame, i.e. an
        # exact allocentric position + heading fix relative to a FIXED maze --
        # global self-localisation handed to the policy for free.  That both
        # defeats the vision-only premise of this task and confounds any
        # "the CNN had to learn place coding" claim, so it is OFF by default.
        # go_to_target_vision's task_obs contract (DESIGN.md 3e) includes it;
        # set this true to recover that contract exactly.
        include_origin=False,
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
        # --- Appearance ---------------------------------------------------
        # The warp ray-tracer shades a surface by N.L from the directional
        # lights alone; it ignores `light.ambient`, so an overhead-only rig
        # leaves every vertical wall face near-black in the policy's own view
        # (measured: 75% of egocentric pixels below 0.15). `wall_lights` adds
        # this many directional lights spaced evenly in azimuth and tilted
        # `wall_light_elevation` below horizontal, which is what puts a
        # gradient on all four wall orientations. 0 disables them.
        wall_lights=4,
        wall_light_elevation=0.35,  # tan of the downward tilt, not an angle
        # labmaze wall texture as "<style>/<tint>" (dm_control's mazes use
        # style_01..style_05 from the same asset pack). Empty string falls back
        # to the builtin checker, which is much darker.
        wall_texture="style_03/gray_bright",
        # Blue gradient skybox. Ignored when `aesthetic` supplies a real one.
        sky=True,
        # "default" keeps the builtin checker floor and gradient sky.
        # "outdoor_natural" reproduces dm_control's rodent_maze_forage look by
        # loading ITS assets out of the installed dm_control package: a
        # photographic grass floor and a photographic skybox. The point is not
        # cosmetic -- a flat checker floor gives the policy almost no optic
        # flow, while grass is dense high-frequency texture. See
        # _apply_aesthetic for the measured effect on the policy's own view.
        aesthetic="default",
        # --- Reward terms ---
        reward_terms={
            "treat_collected": {"weight": 1.0},
        },
        # --- Termination criteria ---
        # `belly_up` + `torso_too_low` rather than `fallen`: the latter's tilt
        # gate cannot distinguish a reared rat from a fallen one, and real rat
        # motion spends 8% of its frames past 85 deg (see _belly_up_termination
        # for the measured distribution). Both new criteria are rate-limited by
        # `patience` consecutive steps, which is what encodes "cannot recover".
        termination_criteria={
            "belly_up": {
                "max_tilt_angle": 140.0,
                "fallen_tilt_angle": 90.0,
                "fallen_max_torso_z": 0.10,
                "patience": 50,
            },
            "torso_too_low": {"min_torso_z": 0.0325, "patience": 50},
            "all_treats_collected": {},
            "nan_termination": {},
        },
    )


class MazeForageVision(rodent_base.RodentEnv):
    """Sparse, vision-only maze foraging.

    The maze is built once at construction; only ``Data`` changes per episode.
    See the module docstring for the observation contract and the two gotchas
    (qpos shift, ``full_reset``).
    """

    _registry = _registry

    #: Declares that ``env.reset`` must run at every episode boundary.  The
    #: spawn cell, heading and treat cells live in ``Data``, so
    #: ``BraxAutoResetWrapper(full_reset=False)`` -- which restores the FIRST
    #: reset's ``data`` -- freezes one layout per env index for the whole run
    #: and never clears ``collected``.  ``train_highlvl.py`` reads this flag
    #: and refuses to launch without ``wrappers.full_reset: true``.
    requires_per_episode_reset = True

    #: ``state.info`` keys a per-episode info reset has to restore if you run
    #: with ``full_reset=False`` on purpose (ablation only -- it leaves the
    #: layout frozen).
    info_reset_keys = INFO_RESET_KEYS

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initializes the MazeForageVision environment.

        Args:
            rng: Random number generator key (kept for API parity with the other
                rodent tasks; the maze itself is seeded by ``config.maze_seed``).
            config: Configuration dictionary.  Defaults to a **freshly built**
                ``default_config()``: ``MjxEnv.__init__`` calls
                ``update_from_flattened_dict(config_overrides)``, which mutates
                the dict in place, so a ``config=default_config()`` default
                argument (evaluated once at class-definition time) would leak
                one instance's overrides into every later instance.
            config_overrides: Optional configuration overrides.

        Raises:
            ValueError: If ``mujoco_impl`` is not ``"warp"`` (the vision renderer
                requires it), if an explicit ``cell_size`` disagrees with
                ``maze_extent / grid_size``, or if the maze has too few free
                cells to place the spawn plus every treat.
        """
        # NOT a default argument: see the `config` docstring above.
        if config is None:
            config = default_config()
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
        self._treat_height = float(self._config.treat_height)
        self._park_depth = float(self._config.park_depth)
        self._treat_reach_threshold = float(self._config.treat_reach_threshold)
        self._spawn_height = float(self._config.spawn_height)
        self._include_origin = bool(self._config.get("include_origin", False))

        # --- Host-side maze construction (runs once, never under trace) ---
        self._maze_grid = maze_utils.generate_maze(
            maze_cells=int(self._config.maze_cells),
            seed=int(self._config.maze_seed),
            loop_fraction=float(self._config.maze_loop_fraction),
        )
        # cell_size is DERIVED from the arena extent; see _resolve_cell_size.
        self._cell_size = self._resolve_cell_size(self._maze_grid.shape)
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
        # Independent re-derivation of the footprint (maze_utils' own formula),
        # so a drift in either half of the geometry is caught, not averaged.
        self._check_realised_extent()
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
        self._add_wall_lights()
        self._spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self._spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        self._spec.visual.headlight.specular = [0.1, 0.1, 0.1]

        self.compile()

        # --- Post-compile index caching (host-side, never under trace) ---
        self._cache_treat_indices()
        self._cache_rodent_qpos_layout()

    # ------------------------------------------------------------------
    # Sizing (host-side, before anything geometric is built)
    # ------------------------------------------------------------------

    def _resolve_cell_size(self, grid_shape: Tuple[int, int]) -> float:
        """Derives the grid pitch from ``config.maze_extent``, or checks it.

        The invariant this enforces is the whole point of ``maze_extent``::

            grid_size * cell_size == maze_extent

        with ``grid_size = 2 * maze_cells + 1``.  ``config.cell_size`` is
        ``None`` by default, in which case the pitch is derived and the
        invariant holds by construction; if it is set explicitly the two are
        cross-checked so a stale ``cell_size:`` in a yaml cannot silently
        resize the arena.

        Args:
            grid_shape: ``(height, width)`` of the generated maze grid.

        Returns:
            The grid pitch in metres.

        Raises:
            ValueError: If the grid is not square, if ``maze_extent`` is not
                positive, or if an explicit ``cell_size`` deviates from
                ``maze_extent / grid_size`` by more than 1e-9 m.
        """
        height, width = int(grid_shape[0]), int(grid_shape[1])
        if height != width:
            raise ValueError(
                f"Maze grid must be square, got {grid_shape}. maze_cells has "
                "to be a single int for a fixed-extent arena."
            )
        extent = float(self._config.maze_extent)
        if not extent > 0.0:
            raise ValueError(f"maze_extent must be > 0, got {extent}.")

        derived = extent / float(width)
        configured = self._config.get("cell_size", None)
        if configured is None:
            return derived

        cell_size = float(configured)
        realised = cell_size * width
        if abs(realised - extent) > 1e-9:
            raise ValueError(
                f"cell_size={cell_size!r} gives a {realised:.6f} m maze on a "
                f"{width}x{width} grid, but maze_extent={extent} m was asked "
                f"for (off by {realised - extent:+.6f} m). Either drop "
                f"cell_size (it is derived: {derived:.6f} m for "
                f"maze_cells={int(self._config.maze_cells)}) or set a "
                "maze_extent that matches."
            )
        return cell_size

    def _check_realised_extent(self) -> None:
        """Re-checks the compiled footprint against ``config.maze_extent``.

        ``_resolve_cell_size`` guarantees the invariant arithmetically; this
        re-derives the same number through ``maze_utils.maze_extent`` (the
        function the wall geometry and the render harness actually use), so a
        divergence between the two code paths raises here rather than shipping
        a maze that is not the size it claims to be.

        Raises:
            ValueError: If either realised extent differs from
                ``config.maze_extent`` by more than 1e-9 m.
        """
        extent = float(self._config.maze_extent)
        x_extent, y_extent = self.maze_extent
        for axis, value in (("x", x_extent), ("y", y_extent)):
            if abs(value - extent) > 1e-9:
                raise ValueError(
                    f"Realised maze {axis} extent is {value:.9f} m but "
                    f"maze_extent={extent} m was configured (cell_size="
                    f"{self._cell_size!r}, grid {self._maze_grid.shape})."
                )

    # ------------------------------------------------------------------
    # Arena construction (host-side, all of it before compile())
    # ------------------------------------------------------------------

    def _add_wall_lights(self) -> None:
        """Adds a ring of near-horizontal directional lights around the maze.

        The key light points almost straight down, which is fine for a top-down
        render and useless for the policy: the warp ray-tracer shades a face by
        ``N.L`` over the directional lights only -- it ignores ``light.ambient``
        entirely (verified: egocentric frames are bit-identical at ambient 0.0,
        0.3 and 0.5) -- so a vertical wall lit only from above gets ``N.L ~ 0``
        and renders black.  Spacing ``config.wall_lights`` lights evenly in
        azimuth and tilting them just below horizontal gives every wall
        orientation a lit side.  Measured on the egocentric camera, this takes
        the fraction of pixels below 0.15 from 75% to 11.6%.
        """
        n_lights = int(self._config.get("wall_lights", 0))
        if n_lights <= 0:
            return
        # Far enough out that the lights never sit inside the maze; they are
        # directional, so the distance only matters for shadow casting.
        radius = 2.0 * float(self._config.maze_extent)
        elevation = float(self._config.get("wall_light_elevation", 0.35))
        for i in range(n_lights):
            angle = 2.0 * np.pi * i / n_lights
            direction = [np.cos(angle), np.sin(angle), -elevation]
            light = self._spec.worldbody.add_light(
                name=f"wall_light_{i}",
                pos=[-direction[0] * radius, -direction[1] * radius, radius * 0.5],
                dir=direction,
                diffuse=[0.45, 0.45, 0.45],
                specular=[0.0, 0.0, 0.0],
                castshadow=0,
            )
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    @staticmethod
    def _dm_control_asset(filename: str) -> str:
        """Absolute path to one of dm_control's `outdoor_natural` assets.

        Resolved through the INSTALLED package rather than a checkout path, so
        this does not depend on a sibling clone existing.
        """
        import dm_control

        path = os.path.join(
            os.path.dirname(dm_control.__file__),
            "locomotion", "arenas", "assets", "outdoor_natural", filename,
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"dm_control aesthetic asset missing: {path}. Install dm_control "
                "or set config.aesthetic='default'."
            )
        return path

    def _apply_aesthetic(self) -> None:
        """Swaps in dm_control's `outdoor_natural` floor and sky.

        WHY THIS IS NOT COSMETIC. The arena's default floor is a two-tone
        checker with a 1 m period; across a 0.465 m corridor the policy sees at
        most one edge, so translation produces almost no change in the image
        and the 32x32 view carries little depth or speed information. The
        dm_control maze uses a photographic grass texture for exactly this
        reason. Measured over 32 reset frames of the policy's own 32x32
        grayscale camera:

                                 mean   std   mean|grad|   dark<0.15
            default            0.643  0.219      0.0318       12.4%
            outdoor_natural    0.372  0.308      0.0506       12.4%

        i.e. +59% spatial gradient energy and +41% intensity spread, at the
        same dark-pixel fraction. The scene is darker overall (grass is darker
        than the pale checker) but carries substantially more structure, which
        is the quantity a CNN can use.

        Wall texture follows dm_control's example too: `basic_rodent_2020`
        builds its maze with labmaze `style_01`, whose green tint is the pale
        circuit-board pattern in dm_control's published screenshots. Set
        `wall_texture` explicitly to override.
        """
        if str(self._config.get("aesthetic", "default")) != "outdoor_natural":
            return

        # Skybox: 3x4 cube-map atlas, layout straight from
        # dm_control.locomotion.arenas.assets.get_sky_texture_info.
        sky = self._spec.add_texture(
            name=_SKY_TEXTURE,
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            file=self._dm_control_asset("OutdoorSkybox2048.png"),
        )
        sky.gridsize = [3, 4]
        sky.gridlayout = ".U..LFRB.D.."

        self._spec.add_texture(
            name=_FLOOR_TEXTURE,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=self._dm_control_asset("OutdoorGrassFloorD.png"),
        )
        floor_mat = self._spec.add_material(name=_FLOOR_MATERIAL)
        floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _FLOOR_TEXTURE
        # One tile per grid cell: fine enough to give optic flow inside a
        # corridor, coarse enough not to alias at 32x32.
        repeat = float(self._config.maze_extent) / float(self._cell_size)
        floor_mat.texrepeat = [repeat, repeat]
        floor_mat.texuniform = False
        floor_mat.reflectance = 0.0
        for geom in self._spec.geoms:
            if geom.name == "floor":
                geom.material = _FLOOR_MATERIAL

    def _add_sky(self) -> None:
        """Adds a blue gradient skybox, matching dm_control's outdoor aesthetic."""
        if str(self._config.get("aesthetic", "default")) == "outdoor_natural":
            return  # _apply_aesthetic installs a photographic skybox instead
        if not bool(self._config.get("sky", False)):
            return
        self._spec.add_texture(
            name=_SKY_TEXTURE,
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            width=256,
            height=256,
            rgb1=[0.40, 0.60, 0.85],
            rgb2=[0.85, 0.92, 1.0],
        )

    def _wall_texture_asset(self) -> Optional[str]:
        """Absolute path to the configured labmaze wall texture, or ``None``.

        ``config.wall_texture`` is ``"<style>/<tint>"`` -- the same asset pack
        dm_control's mazes draw from.  Returns ``None`` when it is unset, in
        which case the caller falls back to the builtin checker.
        """
        spec_str = str(self._config.get("wall_texture", "") or "").strip()
        if not spec_str:
            return None
        if spec_str == "dm_control":
            # What basic_rodent_2020.rodent_maze_forage uses.
            spec_str = "style_01/green"
        if "/" not in spec_str:
            raise ValueError(
                f"config.wall_texture must be '<style>/<tint>', got {spec_str!r}."
            )
        style, tint = spec_str.split("/", 1)
        from labmaze import assets as labmaze_assets

        try:
            paths = labmaze_assets.get_wall_texture_paths(style)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Unknown labmaze wall texture style {style!r} "
                f"(config.wall_texture={spec_str!r})."
            ) from exc
        if tint not in paths:
            raise ValueError(
                f"Unknown labmaze tint {tint!r} for style {style!r}; "
                f"available: {sorted(paths)}."
            )
        # labmaze hands back a template ("wall_{}_d.png" in some versions).
        return str(paths[tint]).format(tint)

    def _add_materials(self) -> None:
        """Registers the wall and treat materials on the arena spec."""
        self._add_sky()
        self._apply_aesthetic()
        texture_path = self._wall_texture_asset()
        if texture_path is None:
            self._spec.add_texture(
                name=_WALL_TEXTURE,
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                width=256,
                height=256,
                rgb1=[0.30, 0.30, 0.34],
                rgb2=[0.55, 0.55, 0.60],
            )
        else:
            # Absolute path, NOT compiler.texturedir + basename: texturedir is a
            # single directory for the whole model, and the aesthetic assets
            # live in dm_control's package while the wall textures live in
            # labmaze's. Absolute paths compile fine and keep both reachable.
            self._spec.add_texture(
                name=_WALL_TEXTURE,
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=texture_path,
            )
        wall_mat = self._spec.add_material(name=_WALL_MATERIAL)
        wall_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _WALL_TEXTURE
        wall_mat.texrepeat = [2, 2]
        wall_mat.texuniform = True
        wall_mat.reflectance = 0.0

        # Treats are DARK, not bright, and that is dm_control's own choice:
        # `basic_rodent_2020.rodent_maze_forage` builds its TargetSphere with
        # `rgb1=(0, 0, 0.4)`.  This env shipped a bright yellow instead, on the
        # theory that a treat should be the brightest thing in the frame.  It
        # cannot be: once the wall lights are on, the walls sit at ~0.72 and
        # the floor at ~0.90 grayscale, so a bright treat competes with a
        # bright floor while a dark one has the whole range to itself.
        # Measured over 48 reset frames of the 32x32 policy view, by rendering
        # each frame with treats live and again with them parked underground
        # and looking only at the pixels that changed:
        #
        #   treat rgba              luminance  |contrast|  outside frame p1/p99
        #   yellow 1, 0.85, 0.1       0.679       0.295          39.5%
        #   white  1, 1, 1            0.757       0.324          50.1%
        #   blue   0, 0, 0.4          0.212       0.496          60.7%
        #   black  0, 0, 0            0.161       0.502          68.2%
        #
        # 1.7x the contrast for one line. Black is marginally more extreme but
        # 0,0,0.4 is what the reference uses and is within noise of it.
        #
        # `emission` is NOT set: the warp ray-tracer ignores it outright, the
        # same way it ignores `light.ambient`. Verified bit-identical treat
        # pixels at emission 0.0, 0.4 and 1.0 (0.679 / 0.659 / 0.295 in all
        # three). Setting it would look like a knob and do nothing.
        treat_mat = self._spec.add_material(name=_TREAT_MATERIAL)
        treat_mat.rgba = [0.0, 0.0, 0.4, 1.0]
        treat_mat.reflectance = 0.0

    def _wall_box_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes ``(pos, size)`` for one box geom per covering rectangle.

        ``maze_utils.wall_boxes`` gives dm_control's geometry, where every wall
        cell is filled edge to edge.  That would make the walls as thick as the
        grid pitch (0.1818 m at the defaults, i.e. as wide as the corridors), so
        each rectangle is thinned to ``wall_thickness`` along any axis it spans
        a *single* cell on.

        Thinning alone would leave diagonal holes where a thin wall meets a
        perpendicular one, so each side of a rectangle is then extended by
        ``cell_size - wall_thickness`` **iff every grid cell just beyond that
        side is also a wall**.  The extension never reaches past the
        neighbouring wall *cell*, so the maze stays sealed and no corridor cell
        is ever entered.  Overlapping wall boxes are free: they all live on
        ``worldbody``, and MuJoCo never generates contact pairs within one body.

        .. note::

           The neighbouring wall cell is itself only ``wall_thickness`` full
           once *it* has been thinned, so at a T-junction the extension does
           poke a nub ``cell_size / 2 - 1.5 * wall_thickness`` into the free
           space that thinning had handed back (0.046 m at the defaults).  It
           is a corner nub, not a narrowing of the corridor along its length:
           the corridor still measures ``1.5 * cell_size + wall_thickness / 2``
           there (0.288 m) instead of 0.334 m.  Measured and pinned in
           ``test_corridor_width_is_two_cells_minus_thickness``.

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
        """Caches treat slide-joint qpos addresses, body ids and geom ids.

        The geom ids and the bodies' inertial z offsets are what let
        :meth:`_park_collected_treats` keep ``xpos`` / ``xipos`` / ``geom_xpos``
        in step with the ``qpos`` it writes without paying for an
        ``mjx.forward``.

        Raises:
            ValueError: If a treat joint, body or geom name is missing.
                ``mj_name2id`` returns ``-1`` rather than raising, and ``-1``
                would silently index the *last* joint (i.e. one of the
                rodent's).
        """
        qpos_idxs = np.zeros((self._n_treats, 3), dtype=np.int32)
        body_ids = np.zeros((self._n_treats,), dtype=np.int32)
        geom_ids = np.zeros((self._n_treats,), dtype=np.int32)
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
            geom_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"treat_{i}_geom"
            )
            if geom_id < 0:
                raise ValueError(
                    f"Geom 'treat_{i}_geom' not found in compiled model."
                )
            geom_ids[i] = geom_id

        self._treat_slide_qpos_idxs_np = qpos_idxs
        self._treat_slide_qpos_idxs = jp.array(qpos_idxs)
        self._treat_z_qpos_idxs = jp.array(qpos_idxs[:, 2])
        self._treat_body_ids_np = body_ids
        self._treat_body_ids = jp.array(body_ids)
        self._treat_geom_ids_np = geom_ids
        self._treat_geom_ids = jp.array(geom_ids)
        # Treat bodies hang off worldbody with identity orientation and their
        # geoms sit at the body origin, so world z is just the body reference z
        # plus the z slide offset (plus the inertial/geom local offsets, which
        # are 0 here but are read rather than assumed).
        self._treat_body_ref_z = jp.array(
            self._mj_model.body_pos[body_ids, 2], dtype=jp.float32
        )
        self._treat_body_ipos_z = jp.array(
            self._mj_model.body_ipos[body_ids, 2], dtype=jp.float32
        )
        self._treat_geom_local_z = jp.array(
            self._mj_model.geom_pos[geom_ids, 2], dtype=jp.float32
        )

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
            # Consecutive-step counters behind the `belly_up` /
            # `torso_too_low` terminations; see _update_posture_counters.
            "belly_up_steps": jp.array(0, dtype=jp.int32),
            "low_torso_steps": jp.array(0, dtype=jp.int32),
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

        Ordering mirrors ``go_to_target``: termination and reward are computed
        against the mask as it stood *entering* this step, and the mask is
        advanced afterwards -- so the step on which the last treat is reached
        pays out, and ``all_treats_collected`` fires one step later.

        The observation is built *last*, from the post-park data and the
        advanced mask, so that a treat collected on this step is already
        underground in the frame ``VisionRenderWrapper`` renders from
        ``state.data`` and in ``privileged_state``.  Building it earlier left
        the treat visibly floating for exactly one control step after it had
        stopped paying.

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
        # MUST run before _is_done: the posture terminations only read counters.
        self._update_posture_counters(data, info)

        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        # --- Collected-mask update (AFTER reward) ---
        collected = jp.logical_or(info["collected"], self._treats_in_reach(data))
        info["collected"] = collected
        info["n_collected"] = jp.sum(collected).astype(jp.int32)

        # Collected treats slide underground so they vanish from the camera and
        # cannot re-trigger.  This also rewrites the derived kinematics, so the
        # treat is gone from `data` (and therefore from the render) on the same
        # step it stops paying.
        data = self._park_collected_treats(data, collected)

        obs = self._get_obs(data, info)

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

        ``task_obs`` is ``[prev_action, kinematic_sensors, touch_sensors]``
        (plus ``origin`` iff ``config.include_origin``) and carries **no** treat
        information -- that is the entire point of the task, so do not add an
        ``ego_target`` here.  ``origin`` (an exact allocentric position/heading
        fix in a fixed maze, i.e. free global self-localisation) is **off** by
        default; see that config key.
        ``vision`` is a zeros placeholder that ``VisionRenderWrapper``
        overwrites with real pixels; drop it and the wrapper silently no-ops.

        ``privileged_state`` mirrors ``task_obs`` (``HighLevelWrapper`` sizes
        its privileged branch off it) and adds the treat vectors + collected
        mask for offline analysis.  It carries no ``vision`` entry: the shipped
        vision architectures read pixels from ``state`` only, so a second
        placeholder would just be filled in and thrown away.

        Args:
            data: Simulation data.
            info: State info (must contain ``prev_action`` and ``collected``).

        Returns:
            ``OrderedDict(state=..., privileged_state=...)``.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)

        components = [
            info["prev_action"],
            kinematic_sensors,
            touch_sensors,
        ]
        if self._include_origin:
            components.append(self._get_origin(data))
        task_obs = jp.concatenate(components)

        proprioception = self._get_proprioception(data, info, flatten=False)
        vision = jp.zeros(self.vision_shape)

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=proprioception,
            vision=vision,
        )

        # Offline analysis may see where the treats are; the policy may not.
        # NOTE the shipped `shared_vision_task_obs` arch drops this whole
        # subtree in HighLevelWrapper._process_state, so nothing here reaches
        # the value head today (module docstring).  'task_obs' must stay
        # present -- HighLevelWrapper sizes privileged_state[highlvl_obs_key].
        privileged_obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=proprioception,
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
        """Hides every collected treat, in ``qpos`` *and* in the derived pose.

        Writing ``qpos`` alone would leave ``xpos`` / ``geom_xpos`` stale until
        the next physics step, and ``mjx.render`` reads ``geom_xpos`` -- so the
        camera kept showing a treat that had already been collected for one
        control step (measured: ``r=1.0`` with ``geom_xpos z=+0.05``, only
        reaching ``-park_depth`` on the following step).  The treat bodies hang
        off ``worldbody`` with identity orientation and only translate, so the
        exact world z is available in closed form and no ``mjx.forward`` is
        needed.

        Args:
            data: Simulation data.
            collected: Boolean ``(n_treats,)`` mask.

        Returns:
            ``data`` with the treat z slide qpos and the treat rows of
            ``xpos`` / ``xipos`` / ``geom_xpos`` rewritten (idempotent).
        """
        parked_offset = -(self._park_depth + self._treat_height)
        z_offsets = jp.where(collected, parked_offset, 0.0)
        qpos = data.qpos.at[self._treat_z_qpos_idxs].set(z_offsets)

        body_z = self._treat_body_ref_z + z_offsets
        xpos = data.xpos.at[self._treat_body_ids, 2].set(body_z)
        xipos = data.xipos.at[self._treat_body_ids, 2].set(
            body_z + self._treat_body_ipos_z
        )
        geom_xpos = data.geom_xpos.at[self._treat_geom_ids, 2].set(
            body_z + self._treat_geom_local_z
        )
        return data.replace(
            qpos=qpos, xpos=xpos, xipos=xipos, geom_xpos=geom_xpos
        )

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

    def _torso_posture(self, data: mjx.Data):
        """``(torso_z, cos_tilt)`` where ``cos_tilt`` is the torso z-axis . world z.

        ``cos_tilt`` is ``xmat[2, 2]``: +1 upright, 0 on its side, -1 belly up.
        It is the quantity both posture terminations are written against,
        because it separates "reared" from "inverted" while a plain tilt
        magnitude does not.
        """
        torso = self._torso(data)
        return torso.xpos[2], torso.xmat[-1, -1]

    def _update_posture_counters(self, data: mjx.Data, info) -> None:
        """Advances the consecutive-step counters the posture terminations read.

        Each counter counts how long its condition has held WITHOUT
        interruption; a single good step resets it to zero.  That is what turns
        "is inverted right now" into "is inverted and not recovering", which is
        the actual failure we want to end an episode on.  Mutates ``info``.
        """
        torso_z, cos_tilt = self._torso_posture(data)

        crit = self._config.termination_criteria
        belly_cfg = crit.get("belly_up", {}) or {}
        low_cfg = crit.get("torso_too_low", {}) or {}
        max_tilt = float(belly_cfg.get("max_tilt_angle", 140.0))
        fallen_tilt = float(belly_cfg.get("fallen_tilt_angle", 90.0))
        fallen_max_z = float(belly_cfg.get("fallen_max_torso_z", 0.10))
        min_z = float(low_cfg.get("min_torso_z", 0.0325))

        # Two ways to be down, OR'd. Tilt alone cannot do it: a rat lying on
        # its back settles anywhere in 94-166 deg depending on how it landed,
        # and real REARING reaches 120.8 deg, so the ranges overlap. What does
        # separate them is height -- in the reference data a rat past 90 deg is
        # rearing and its torso is never below 0.1146 m, whereas every settled
        # fallen pose measured sits at 0.09 m or lower.
        inverted = jp.logical_or(
            cos_tilt < float(np.cos(np.deg2rad(max_tilt))),
            jp.logical_and(
                cos_tilt < float(np.cos(np.deg2rad(fallen_tilt))),
                torso_z < fallen_max_z,
            ),
        )
        too_low = torso_z < min_z
        info["belly_up_steps"] = jp.where(
            inverted, info["belly_up_steps"] + 1, jp.zeros_like(info["belly_up_steps"])
        )
        info["low_torso_steps"] = jp.where(
            too_low, info["low_torso_steps"] + 1, jp.zeros_like(info["low_torso_steps"])
        )

    @_registry.termination("belly_up")
    def _belly_up_termination(
        self,
        data,
        info,
        max_tilt_angle: float = 140.0,
        fallen_tilt_angle: float = 90.0,
        fallen_max_torso_z: float = 0.10,
        patience: int = 50,
    ) -> bool:
        """End the episode only when the rodent is inverted and staying that way.

        THE POINT OF THIS CRITERION is that ``fallen``'s tilt gate cannot tell a
        rearing rat from a fallen one.  Measured over the 210,500 frames of real
        rat motion in ``rodent_reference_clips.h5`` -- the data this task's
        frozen prior was trained to imitate -- torso tilt from upright is:

            median 11.5 deg,  p95 90.7 deg,  p99 98.2 deg,  max 120.8 deg
            frames >  70 deg : 14.00%   <- rodent_base `fallen` default
            frames >  85 deg :  8.13%   <- what the DMPO gap arms use
            frames > 100 deg :  0.61%
            frames > 120 deg :  0.0019%
            frames > 140 deg :  0.0000%

        Real rats spend 8% of their time past the 85 deg gate, rearing and
        nosing downward. A tilt gate anywhere in that range ends episodes on
        NORMAL behaviour. 140 deg is exceeded by zero frames of real motion, so
        it can only fire on a genuinely inverted body.

        ``cos_tilt`` (``xmat[2, 2]``) rather than a tilt magnitude is what makes
        this work: rearing pitches the torso about its lateral axis and keeps
        ``cos_tilt`` positive, while belly-up drives it to -1.

        A PURE TILT GATE IS NOT ENOUGH, and 140 deg alone missed the common
        case. Dropping the rodent at four roll angles and letting it settle
        under zero torque gives:

            init roll   settled tilt   settled z   fires at 140 deg?
                90 deg       67.7 deg     0.1140   no  (on its side)
               120 deg      129.0 deg     0.0906   yes
               150 deg       94.3 deg    -0.0042   NO  <- lies there sprawled
               180 deg      166.5 deg     0.0212   yes

        The 150 deg case never approaches 140 deg, so the episode used to run
        on until the separate height gate ended it -- the failure is silent and
        mis-attributed. Lowering the tilt threshold does not fix it either:
        fallen poses span 94-166 deg while real REARING reaches 120.8 deg, so
        the two ranges overlap and no tilt cut separates them.

        HEIGHT is what separates them. Conditioning the reference motion on
        tilt (52,625 frames):

            tilt   0-45 deg   z min 0.0354   median 0.0705
            tilt  45-90 deg   z min 0.0858   median 0.1148
            tilt  90-100 deg  z min 0.1146   median 0.1468
            tilt 100-110 deg  z min 0.1274   median 0.1547
            tilt 110-121 deg  z min 0.1329   median 0.1581

        A real rat past 90 deg is REARING, and rearing puts the torso UP: it is
        never below 0.1146 m. Every settled fallen pose is at 0.09 m or below.
        So the second arm is ``tilt > 90 deg AND z < 0.10 m``, which sits in the
        gap with ~1.5 cm of margin on the reference side and ~0.9 cm on the
        fallen side. Measured false-positive rate on the reference: 0 of 52,625
        frames.

        ``patience`` is the "and cannot recover" half. A rat that tumbles
        through an inverted pose for a few frames and rights itself never
        reaches the count; the counter resets on the first good step.

        Args:
            data: Simulation data (unused; the counter is maintained in step()).
            info: State info carrying ``belly_up_steps``.
            max_tilt_angle: Degrees from upright that count as inverted on tilt
                alone, regardless of height.
            fallen_tilt_angle: Degrees from upright for the combined arm.
            fallen_max_torso_z: Torso height below which ``fallen_tilt_angle``
                counts as down rather than reared.
            patience: Consecutive down steps required to terminate.

        Returns:
            Whether the rodent has been inverted for ``patience`` steps.
        """
        del data, max_tilt_angle, fallen_tilt_angle, fallen_max_torso_z
        return info["belly_up_steps"] >= patience

    @_registry.termination("torso_too_low")
    def _torso_too_low_termination(
        self, data, info, min_torso_z: float = 0.0325, patience: int = 50
    ) -> bool:
        """End the episode when the torso stays collapsed against the floor.

        The height half of the old ``fallen``, split out so it can be tuned
        independently of the tilt half -- they fail for different reasons and
        the tilt half was the one doing the damage.

        0.0325 m is safe by measurement, not by guess: over the 210,500
        reference frames the torso NEVER goes below 0.0354 m, so this gate
        cannot fire on any posture the reference rat adopts, while a rodent
        collapsed under zero torque settles at ~0.031 m.

        Args:
            data: Simulation data (unused; the counter is maintained in step()).
            info: State info carrying ``low_torso_steps``.
            min_torso_z: World-z below which the torso counts as collapsed.
                Read in ``_update_posture_counters``, not here.
            patience: Consecutive low steps required to terminate.

        Returns:
            Whether the torso has been below ``min_torso_z`` for ``patience`` steps.
        """
        del data, min_torso_z
        return info["low_torso_steps"] >= patience

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
        ``termination_criteria``: truncation is ``EpisodeWrapper``'s job
        (``train_config.episode_length``), and a second env-side budget is just
        a way for the two to drift apart.  It also depends entirely on
        ``info["step_count"]`` being cleared at every episode boundary -- true
        under ``wrappers.full_reset`` (or ``InfoResetOnDoneWrapper`` with
        ``step_count`` in its keys, which :data:`INFO_RESET_KEYS` has), and
        false under a bare ``BraxAutoResetWrapper(full_reset=False)``, where it
        latches past ``max_steps`` and collapses the env into one-step
        episodes.

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
        """Terminate on NaN values in the simulation state.

        Checks ``qpos`` and ``qvel`` only, NOT ``ravel_pytree(data)``.  On the
        warp backend ``data`` carries contact buffers sized O(naconmax), and
        flattening it under ``vmap`` materialises a copy of them per world: at
        2048 envs and ``naconmax=65536`` that is a single 21.7 GiB allocation
        and the run dies before its first step.  ``run_gap.py`` hit the same
        wall (PR #67) and does the same thing; this env is doubly exposed
        because ``wrappers.full_reset=true`` puts a whole ``reset()`` -- and
        therefore this check -- inside the fused training step as well.

        Checking both integrator states rather than just ``qpos`` costs
        nothing and catches a NaN one step earlier, since a NaN reaches
        ``qvel`` before it is integrated into ``qpos``.

        Args:
            data: Simulation data.
            info: State info (unused).

        Returns:
            Boolean indicating whether a NaN was detected.
        """
        del info
        return jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

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
    def cell_size(self) -> float:
        """Grid pitch in metres: ``maze_extent / (2 * maze_cells + 1)``.

        Derived, not configured (see ``_resolve_cell_size``).  This is NOT the
        corridor width -- see :attr:`corridor_width`.
        """
        return self._cell_size

    @property
    def maze_extent(self) -> Tuple[float, float]:
        """Realised outer extent ``(x, y)`` of the maze footprint, in metres.

        Read straight off the compiled grid geometry, so it is a check on
        ``config.maze_extent`` rather than a copy of it (``__init__`` raises if
        they disagree).  2.0 x 2.0 m at the defaults, for every ``maze_cells``.
        """
        half_x, half_y = self._maze_half_extent
        return (2.0 * half_x, 2.0 * half_y)

    @property
    def maze_half_extent(self) -> Tuple[float, float]:
        """Half-extent ``(x, y)`` of the maze footprint, in metres."""
        return self._maze_half_extent

    @property
    def n_free_cells(self) -> int:
        """Number of open (non-wall) grid cells -- the spawn/treat sample pool."""
        return self._n_free_cells

    @property
    def open_cell_fraction(self) -> float:
        """Fraction of the ``(2n+1)^2`` grid cells that are floor, not wall."""
        return self._n_free_cells / float(self._maze_grid.size)

    @property
    def treat_cell_fraction(self) -> float:
        """Fraction of the maze's free cells that hold a treat at reset.

        Sparsity of the exploration problem in one number: ``n_treats`` treats
        hidden among :attr:`n_free_cells` reachable cells, resampled every
        episode.  A larger ``maze_cells`` at fixed ``n_treats`` makes this
        strictly smaller, i.e. strictly harder under an already-sparse reward.
        """
        return self._n_treats / float(self._n_free_cells)

    @property
    def corridor_width(self) -> float:
        """Clear width in metres between two parallel maze walls.

        ``cell_size`` is the grid *pitch*, not this.  ``_wall_box_geometry``
        thins every single-cell-wide wall rectangle down to ``wall_thickness``,
        which frees ``cell_size - wall_thickness`` of floor on each side of the
        wall, so a nominally one-cell corridor is really
        ``2 * cell_size - wall_thickness`` wide -- 0.3336 m at the defaults
        (``cell_size = 2/11``, ``wall_thickness = 0.03``), against a 0.295 m
        long, 0.072 m wide rat.  If the thinning is disabled (a degenerate
        ``wall_thickness`` outside ``(0, cell_size)`` falls back to
        dm_control's full-cell walls) the corridor is exactly one cell.
        """
        thickness = float(self._config.wall_thickness)
        if not 0.0 < thickness < self._cell_size:
            return self._cell_size
        return 2.0 * self._cell_size - thickness

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
