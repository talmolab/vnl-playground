"""Vision-guided Y-/T-maze "spout" memory task for the virtual rodent (warp only).

The rodent runs many *trials* per *episode* in a fixed Y maze (one stem, two
goal arms), or, with ``maze_shape="T"``, in the lab's T maze at x2.5
(`spout_geometry.build_t_maze`).  One arm is baited for the whole episode
(drawn at ``reset``).  A
trial ends when the rat has visited an arm end, or on a timeout; the rat is then
teleported back to the start of the stem, facing the junction, and the episode
(and with it the policy's recurrent state) runs on.  A memoryless policy sees
the same start view on every trial, so its choice cannot depend on history and
it is held to ``1 / n_arms`` hits; doing better needs memory across trials.

Outcome signal (the only one the policy gets): a **visible treat**.  The choice
locks on the first arm whose appear zone (the last ``appear_distance`` m of the
arm) the torso enters.  If that arm is the bait and a ``pay_prob`` draw
succeeds, the arm's treat *appears* (its z slide joint is unparked) at
``treat_xy``; reaching it (torso within ``treat_reach_threshold`` in xy) pays
``+weight`` once and parks it again.  On an unbaited arm nothing appears and the
trial ends when the torso reaches the arm end (``spout_radius``).  Either way a
``dwell_steps`` dwell runs before the trial ends.  The state machine is
`spout_logic.trial_update`; the geometry is `spout_geometry.build_y_maze` or
`spout_geometry.build_t_maze`.

Maze shape (``maze_shape``, default ``"Y"``): both builders return the same
layout type, so the shape only changes construction.  ``"Y"`` uses the
``stem_length`` / ``arm_length`` / ``arm_angle_deg`` / ``corridor_width`` /
``treat_inset`` keys and the parent's ``wall_thickness``; ``"T"`` uses the
``t_*`` keys (`spout_geometry.TMazeSpec`; its own ``t_wall_thickness``).  Both
use ``wall_height``.  The T's spout point is both ``arm_end_xy`` and
``treat_xy``, and its choice locks on arm ENTRY when ``appear_distance`` is at
least every in-arm distance (0.70 in the T configs).  The wall/floor texture
pitch is shape-independent (from ``corridor_width`` and ``wall_thickness``, the
parent's physical scale).

Teleport at trial end (``spout/trial_end``):

* the root goes to ``(start_xy, spawn_height)`` with yaw ``start_yaw`` plus a
  uniform ``+-start_yaw_jitter_deg`` draw, every other joint to ``qpos0``, both
  treats are parked, and ``qvel`` / ``ctrl`` / ``act`` / ``qacc_warmstart`` are
  zeroed -- exactly the inputs a fresh ``reset`` hands to ``mjx.forward``;
* ``info["action"]`` and ``info["prev_action"]`` (and the posture counters) are
  zeroed, so no carried non-visual observation remembers the last trial;
* ``mjx.forward`` runs and the forwarded data replaces the stepped data, per
  env, through ``jnp.where(trial_end, ...)`` over every per-world ``Data``
  field (the warp contact pool, shared by all worlds, is left alone -- see
  `RodentSpoutMazeVision._where_data`);
* the observation is built afterwards, so ``BinocularVisionRenderWrapper``
  renders the start view on the teleport step itself.

With zero jitter the teleport step's observation (vision, ``task_obs`` and
proprioception) equals a fresh reset's; ``tests/test_spout_maze.py`` checks it.

Inter-trial hold and start gate (``hold_enabled``, T maze only, default off):
the lab's start-box hold.  `spout_hold` holds the state machine.

* A ``gate`` body on two slide joints (y and z) carries a colliding box
  (clear corridor width x ``wall_height`` x ``gate_thickness``,
  ``contype=0, conaffinity=1``: it collides with the rat's ``contype=1``
  geoms; an ``exclude`` with the world body keeps it off the walls) and, on its
  ``start`` (-y) face, ~1 mm in front of the box, vertical bars
  ``gate_bar_width`` wide in bright white (0.95) and dark grey (0.25), as
  separate non-colliding box geoms (the warp ray tracer shades a box with a
  single colour, so a texture cannot draw the stripes).  It is never black: the
  warp sky is near-black, so a black gate could read as an opening.  The body is
  added before the rodent, so the proprioception slices still land on the
  rodent's own dofs.  The closed gate spans exactly the walls' z range (the
  box and the bars, ``-_WALL_FLOOR_EMBED`` to ``wall_height -
  _WALL_FLOOR_EMBED``): it is sunk into the floor like the walls, so no slot is
  left under it for a finger capsule to catch in.
* The gate stands on stem boundary ``k`` (drawn per trial from
  ``gate_boundaries``; boundary ``k`` is ``layout.segments.stem_edges_y[k]``,
  between stem segments ``k - 1`` and ``k``).  Its height offset is
  `spout_hold.gate_z`: 0 (closed) during the hold, a linear slide down over
  ``gate_open_steps`` after it, then parked ``park_depth`` below the floor.
  Every step writes the gate's qpos and zeroes its qvel (contact forces never
  move it), and writes its ``xpos`` / ``xipos`` / ``geom_xpos`` in closed
  form (`_place_gate`, as `_park_collected_treats` does for the treats), so
  the frame rendered on a step shows the gate where that step put it; the
  next physics step collides with it there.
* ``reset`` and every teleport start a hold (drawn from
  ``[round(hold_min_s / ctrl_dt), round(hold_max_s / ctrl_dt)]`` steps) with
  the gate closed.  While holding, the trial state and its events are frozen
  (`spout_hold.freeze_trial`): the trial clock and timeout count only from the
  release, and no trial event or reward fires.
* Metrics ``spout/hold_active`` (the step started in the hold) and
  ``spout/gate_release``; ``info["hold"]`` is the `spout_hold.HoldState` as a
  dict.  ``critic_priv`` is unchanged (the gate is visible to the policy).

With ``hold_enabled=False`` nothing of this is built or traced: the model, the
rng streams and the step are those of the hold-free env.

Treat visibility (``treat_visible_until_teleport``, default off): the treat is
rendered only while shown and not yet eaten, i.e. from the step it appears to the
step it is eaten (~0.18 s at 100 Hz on the T).  With the flag on, the locked
arm's shown treat is also rendered through the post-eating dwell, until the
teleport on ``trial_end`` parks it, so a slow memory core ticking every 0.2 s
cannot miss the outcome.  The reward is untouched (`spout_logic.trial_update`
still pays once, on the eat step).  ``info["collected"]`` keeps its meaning
("treat ``i`` is parked"), so it is False for the eaten treat until the
teleport.

Everything below that does not depend on the grid maze is inherited from
`maze_forage_vision.MazeForageVision`: materials, treat bodies and the
closed-form treat parking, the rodent, the proprioception overrides for the
slide-joint qpos shift, ``_get_obs``, the posture counters and terminations.
``__init__`` is rewritten (it skips the parent's grid-maze construction), as are
``reset``, ``step`` and the wall-light ring.  The grid-maze config keys
(``maze_extent``, ``maze_cells``, ``cell_size``, ``maze_seed``,
``maze_loop_fraction``) are inherited for config compatibility and ignored.

Rewards and terminations live in this module's own `RewardRegistry`
(``_registry``): ``spout_treat`` plus thin wrappers over the parent's
``belly_up``, ``torso_too_low`` and ``nan_termination``.  The parent's registry
is never decorated against.

``state.info`` = the parent's per-step keys (``prev_action``, ``action``,
``step_count``, ``collected``, ``n_collected``, ``belly_up_steps``,
``low_torso_steps``) plus ``trial`` (a `spout_logic.TrialState` as a dict of
arrays), ``critic_priv`` (``f32[8]``, `spout_logic.critic_privileged_vector`;
privileged, for the critic and analysis only, never the policy) and
``spout_rng`` (the per-env key stream for the payout and jitter draws; split
every step, never reused, and kept apart from the bait draw) and ``segment``
(int32, `spout_geometry.segment_of` of the torso xy after any teleport: the
T-maze segment, for offline analysis only; -1 throughout in the Y), and, with
``hold_enabled`` only, ``hold`` (a `spout_hold.HoldState` as a dict).
``collected[i]`` is True while treat ``i`` is parked (the meaning `_park_collected_treats` gives
it); ``n_collected`` counts treats eaten this episode.

Metrics: ``rewards/treat_collected`` (the reward) and ``spout/<event>`` for each
name in :data:`SPOUT_METRICS` (plus :data:`HOLD_METRICS` with ``hold_enabled``).

``BraxAutoResetWrapper(full_reset=True)`` is mandatory, as for the parent
(``requires_per_episode_reset``): the bait and the trial state live in
``state.info`` and the start pose in ``Data``.  ``trial`` is a nested dict, so
``InfoResetOnDoneWrapper`` cannot restore it; the ``full_reset=False`` ablation
path is not supported here.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx.warp import types as mjx_warp_types
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.reward_registry import RewardRegistry
from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import maze_forage_vision as mfv
from vnl_playground.tasks.rodent import spout_geometry, spout_hold, spout_logic

_registry = RewardRegistry()

#: ``spout/<name>`` metrics written every step (events of `spout_logic.trial_update`).
SPOUT_METRICS = (
    "trial_end",
    "hit",
    "trial_after_first",
    "hit_after_first",
    "timeout",
    "choice_none",
    "show_treat",
)

#: ``spout/<name>`` metrics added with ``hold_enabled`` (events of `spout_hold.hold_update`).
HOLD_METRICS = spout_hold.HOLD_EVENTS

# The start gate (``hold_enabled``): body / joint / geom names and the striped face.
_GATE_BODY = "gate"
_GATE_BOX_GEOM = "gate_box"
_GATE_BAR_GEOM = "gate_bar_{}"
_GATE_JOINTS = ("gate_slide_y", "gate_slide_z")
_GATE_BRIGHT_RGBA = (0.95, 0.95, 0.95, 1.0)
_GATE_DARK_RGBA = (0.25, 0.25, 0.25, 1.0)
# The bars are 1 mm deep slabs whose back face is 1 mm in front of the collision
# box's start-side face.
_GATE_BAR_HALF_DEPTH = 0.0005
_GATE_BAR_GAP = 0.001

# Transient ``info`` key that carries this step's trial events into the
# ``spout_treat`` reward.  It is popped before ``step`` / ``reset`` return, so it
# never becomes part of the ``state.info`` pytree.
_EVENTS_KEY = "_spout_step_events"


def default_config() -> config_dict.ConfigDict:
    """The parent's config plus the Y-maze / T-maze / trial keys.

    Changed defaults: ``n_treats=2`` (one treat per arm), ``ctrl_dt=0.01`` (the
    rate the step counts below are written in: dwell 30 steps = 0.3 s, timeout
    800 steps = 8 s), ``reward_terms={"spout_treat": ...}`` and
    ``termination_criteria`` = the parent's ``belly_up`` / ``torso_too_low``
    thresholds plus ``nan_termination``.

    Returns:
        A fresh (unlocked) ConfigDict.
    """
    cfg = mfv.default_config().to_dict()
    spec = spout_geometry.YMazeSpec()
    t_spec = spout_geometry.TMazeSpec()
    parent_term = cfg["termination_criteria"]
    cfg.update(
        ctrl_dt=0.01,
        # --- Y maze (spout_geometry.YMazeSpec; walls use wall_height and
        # wall_thickness from the parent's keys) ---
        stem_length=spec.stem_length,
        arm_length=spec.arm_length,
        arm_angle_deg=spec.arm_angle_deg,
        corridor_width=spec.corridor_width,  # >= 0.465 m, the rat's turn width
        appear_distance=spec.appear_distance,  # last 0.5 m of an arm
        spout_radius=spec.spout_radius,
        treat_inset=spec.treat_inset,
        # --- Maze shape: "Y" (keys above) or "T" (t_* keys below; the T sets
        # appear_distance >= every in-arm distance, so the choice locks on arm
        # entry: 0.70 in its configs) ---
        maze_shape="Y",
        # --- T maze (spout_geometry.TMazeSpec; lab mouse T maze x2.5; walls
        # use wall_height and t_wall_thickness) ---
        t_corridor_width=t_spec.corridor_width,
        t_segment_length=t_spec.segment_length,
        t_n_stem_segments=t_spec.n_stem_segments,
        t_arm_length=t_spec.arm_length,
        t_spout_offset=t_spec.spout_offset,
        t_wall_thickness=t_spec.wall_thickness,
        t_entry_margin=t_spec.entry_margin,
        # --- Trials (spout_logic.SpoutParams; reach = treat_reach_threshold) ---
        n_arms=2,
        pay_prob=1.0,
        dwell_steps=30,
        trial_timeout_steps=800,
        start_yaw_jitter_deg=10.0,
        critic_priv_max_trials=20,
        # --- Inter-trial hold + striped start gate (spout_hold; T maze only;
        # all off by default).  Hold uniform over [round(hold_min_s / ctrl_dt),
        # round(hold_max_s / ctrl_dt)] steps; gate on a stem boundary drawn
        # from gate_boundaries (k = 1 is start | seg1); it slides below the
        # floor over gate_open_steps after the hold ---
        hold_enabled=False,
        hold_min_s=1.0,
        hold_max_s=3.0,
        gate_boundaries=(1,),
        gate_open_steps=20,
        gate_thickness=0.03,
        gate_bar_width=0.05,
        # --- Keep the eaten treat rendered through the post-eating dwell until
        # the teleport (default off: it is parked the step it is eaten) ---
        treat_visible_until_teleport=False,
        # --- One treat per arm ---
        n_treats=2,
        reward_terms={"spout_treat": {"weight": 1.0}},
        termination_criteria={
            "belly_up": dict(parent_term["belly_up"]),
            "torso_too_low": dict(parent_term["torso_too_low"]),
            "nan_termination": {},
        },
    )
    return config_dict.create(**cfg)


def _yaw_quat(yaw: jp.ndarray) -> jp.ndarray:
    """Unit quaternion (w, x, y, z) for a rotation of ``yaw`` rad about +z."""
    half = 0.5 * jp.asarray(yaw, dtype=jp.float32)
    zero = jp.zeros_like(half)
    return jp.stack([jp.cos(half), zero, zero, jp.sin(half)])


class RodentSpoutMazeVision(mfv.MazeForageVision):
    """Y-/T-maze spout task with a visible-treat outcome and teleported trials.

    See the module docstring for the trial structure, the teleport and the
    ``info`` / metrics contract.
    """

    _registry = _registry

    #: The bait and the trial state live in ``info`` and the start pose in
    #: ``Data``; only ``env.reset`` re-draws them (see the module docstring).
    requires_per_episode_reset = True

    #: Test hook.  When True at TRACE time, every step ends the trial (through
    #: `spout_logic.trial_update`'s own timeout path).  A jitted caller has to
    #: re-trace after flipping it (e.g. jit a fresh lambda).
    _force_trial_end: bool = False

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, str | int | list[Any]] | None = None,
    ) -> None:
        """Builds the Y or T maze, the two treats and the rodent, then compiles.

        Args:
            rng: Kept for API parity with the other rodent tasks.
            config: Configuration; a freshly built ``default_config()`` if None.
            config_overrides: Optional configuration overrides.

        Raises:
            ValueError: If ``mujoco_impl`` is not ``"warp"``, if
                ``maze_shape`` is not ``"Y"`` or ``"T"``, if
                ``aesthetic="outdoor_natural"`` (its wall facades assume
                axis-aligned walls), if ``n_arms`` / ``n_treats`` do not match
                the maze's two arms, or (``hold_enabled``) if the maze is not
                the T or a hold / gate key is invalid
                (`spout_hold.make_hold_params`, `_check_gate_config`).  All of
                these raise before the model is compiled.
        """
        if config is None:
            config = default_config()
        # Skip MazeForageVision.__init__: it builds the grid maze.
        rodent_base.RodentEnv.__init__(self, config, config_overrides)
        self._rng = rng
        cfg = self._config

        if cfg.mujoco_impl != "warp":
            raise ValueError(
                "RodentSpoutMazeVision requires mujoco_impl='warp' for rendering"
            )
        # .get: configs saved before the T maze existed have no maze_shape.
        self._maze_shape = str(cfg.get("maze_shape", "Y"))
        if self._maze_shape not in ("Y", "T"):
            raise ValueError(
                f"maze_shape must be 'Y' or 'T', got {self._maze_shape!r}."
            )
        # .get: configs saved before the hold existed have no hold_enabled.
        self._hold_enabled = bool(cfg.get("hold_enabled", False))
        if self._hold_enabled and self._maze_shape != "T":
            raise ValueError(
                "hold_enabled=True requires maze_shape='T': the start gate stands on "
                f"a T-maze stem boundary (got maze_shape={self._maze_shape!r})."
            )
        if str(cfg.get("aesthetic", "default")) == "outdoor_natural":
            raise ValueError(
                "aesthetic='outdoor_natural' is not supported: its per-face wall "
                "facades assume axis-aligned walls, and the Y-maze walls are rotated "
                "(the T maze is not supported with it either: untested)."
            )

        self._vision_width = int(cfg.vision_width)
        self._vision_height = int(cfg.vision_height)
        self._grayscale = bool(cfg.get("grayscale", False))
        self._n_treats = int(cfg.n_treats)
        self._treat_height = float(cfg.treat_height)
        self._park_depth = float(cfg.park_depth)
        self._treat_reach_threshold = float(cfg.treat_reach_threshold)
        self._spawn_height = float(cfg.spawn_height)
        self._include_origin = bool(cfg.get("include_origin", False))

        # --- Host-side maze geometry (runs once, never under trace) ---
        if self._maze_shape == "T":
            self._maze_spec = spout_geometry.TMazeSpec(
                corridor_width=float(cfg.t_corridor_width),
                segment_length=float(cfg.t_segment_length),
                n_stem_segments=int(cfg.t_n_stem_segments),
                arm_length=float(cfg.t_arm_length),
                wall_thickness=float(cfg.t_wall_thickness),
                wall_height=float(cfg.wall_height),
                spout_offset=float(cfg.t_spout_offset),
                entry_margin=float(cfg.t_entry_margin),
            )
            self._layout = spout_geometry.build_t_maze(self._maze_spec)
        else:
            self._maze_spec = spout_geometry.YMazeSpec(
                stem_length=float(cfg.stem_length),
                arm_length=float(cfg.arm_length),
                arm_angle_deg=float(cfg.arm_angle_deg),
                corridor_width=float(cfg.corridor_width),
                wall_thickness=float(cfg.wall_thickness),
                wall_height=float(cfg.wall_height),
                appear_distance=float(cfg.appear_distance),
                spout_radius=float(cfg.spout_radius),
                treat_inset=float(cfg.treat_inset),
            )
            self._layout = spout_geometry.build_y_maze(self._maze_spec)
        n_arms = int(cfg.n_arms)
        if n_arms != self._layout.arm_dir.shape[0]:
            raise ValueError(
                f"n_arms={n_arms} but the {self._maze_shape} maze has "
                f"{self._layout.arm_dir.shape[0]} arms."
            )
        if self._n_treats != n_arms:
            raise ValueError(
                f"n_treats={self._n_treats} must equal n_arms={n_arms}: treat i "
                "sits at the end of arm i."
            )
        self._params = spout_logic.SpoutParams(
            n_arms=n_arms,
            pay_prob=float(cfg.pay_prob),
            appear_distance=float(cfg.appear_distance),
            spout_radius=float(cfg.spout_radius),
            reach=self._treat_reach_threshold,
            dwell_steps=int(cfg.dwell_steps),
            trial_timeout_steps=int(cfg.trial_timeout_steps),
        )
        self._critic_priv_max_trials = int(cfg.critic_priv_max_trials)
        # .get: configs saved before this flag existed do not have it.
        self._treat_visible_until_teleport = bool(
            cfg.get("treat_visible_until_teleport", False)
        )
        if self._hold_enabled:
            self._check_gate_config()
            self._hold_params = spout_hold.make_hold_params(
                hold_min_s=float(cfg.hold_min_s),
                hold_max_s=float(cfg.hold_max_s),
                ctrl_dt=float(cfg.ctrl_dt),
                gate_boundaries=tuple(cfg.gate_boundaries),
                gate_open_steps=int(cfg.gate_open_steps),
                gate_height=float(cfg.wall_height),
                park_z=-(self._park_depth + float(cfg.wall_height)),
                n_stem_segments=int(self._maze_spec.n_stem_segments),
            )
        self._start_yaw = float(self._layout.start_yaw)
        self._start_yaw_jitter = float(np.deg2rad(float(cfg.start_yaw_jitter_deg)))
        self._treat_xy = jp.asarray(self._layout.treat_xy, dtype=jp.float32)

        extent = float(self._layout.extent)
        self._maze_half_extent = (extent, extent)
        # Texture pitch for the parent's material helpers (texrepeat = 1 /
        # cell_size).  A grid maze's corridor is 2 * cell - wall_thickness, so
        # this is the pitch of the grid maze with the same corridor width: the
        # wall and floor textures keep the parent's physical scale.  Read from
        # the Y keys for BOTH shapes, so the T's textures keep the same scale.
        self._cell_size = 0.5 * (float(cfg.corridor_width) + float(cfg.wall_thickness))
        # MuJoCo does not clamp a written qpos to the joint range, it fights it
        # with a limit constraint: the range covers every xy and the park depth.
        self._slide_range = float(
            max(extent, self._park_depth + self._treat_height)
            + float(self._maze_spec.corridor_width)
        )

        self._add_materials()
        self._build_y_maze_walls()
        # Treat bodies before the rodent, so their slide joints take the low
        # qpos/qvel addresses (the parent's proprioception overrides rely on it).
        self._build_treats()
        if self._hold_enabled:
            self._build_gate()  # also before the rodent, for the same reason

        start_x, start_y = (float(v) for v in self._layout.start_xy)
        half_yaw = 0.5 * self._start_yaw
        self.add_rodent(
            torque_actuators=cfg.torque_actuators,
            rescale_factor=cfg.rescale_factor,
            pos=[start_x, start_y, self._spawn_height],
            quat=(float(np.cos(half_yaw)), 0.0, 0.0, float(np.sin(half_yaw))),
        )

        # Same light rig as the parent: a directional key light, the
        # near-horizontal wall-light ring (the warp ray-tracer ignores ambient
        # light, so vertical walls are black without it) and the headlight.
        self._spec.worldbody.add_light(
            name="key_light",
            pos=[0, 0, 8],
            dir=[-0.1, -0.1, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[float(cfg.get("key_light_diffuse", 0.7))] * 3,
            specular=[0.3, 0.3, 0.3],
            castshadow=1,
        )
        self._add_wall_lights()
        self._spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self._spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        self._spec.visual.headlight.specular = [0.1, 0.1, 0.1]

        self.compile()

        self._cache_treat_indices()
        self._cache_rodent_qpos_layout()
        self._qpos0 = jp.asarray(self._mj_model.qpos0, dtype=jp.float32)
        # Treat slide offsets from each body's reference pose (0, 0,
        # treat_height): xy = the spout's treat point, z = parked.
        parked_z = -(self._park_depth + self._treat_height)
        self._parked_treat_offsets = jp.asarray(
            np.concatenate(
                [self._layout.treat_xy, np.full((self._n_treats, 1), parked_z)], axis=-1
            ),
            dtype=jp.float32,
        )
        if self._hold_enabled:
            self._cache_gate_indices()

    # ------------------------------------------------------------------
    # Arena construction (host-side, before compile())
    # ------------------------------------------------------------------

    def _build_y_maze_walls(self) -> None:
        """Adds the maze walls (Y or T) as rotated box geoms ``maze_wall_{i}``.

        The boxes come straight from the layout (`spout_geometry.build_y_maze`
        or `build_t_maze`), rotated by
        ``wall_yaw`` (radians) about +z and sunk into the floor by the parent's
        floor embed so no light leaks under them.  They carry the parent's wall
        material (flat-shaded: MuJoCo does not map a 2D texture onto a box).
        """
        layout = self._layout
        for i in range(layout.wall_pos.shape[0]):
            pos = np.array(layout.wall_pos[i], dtype=float)
            pos[2] -= mfv._WALL_FLOOR_EMBED
            half_yaw = 0.5 * float(layout.wall_yaw[i])
            self._spec.worldbody.add_geom(
                name=f"maze_wall_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(pos),
                size=list(np.asarray(layout.wall_halfsize[i], dtype=float)),
                quat=[float(np.cos(half_yaw)), 0.0, 0.0, float(np.sin(half_yaw))],
                material=mfv._WALL_MATERIAL,
                contype=1,
                conaffinity=1,
            )

    def _add_wall_lights(self) -> None:
        """The parent's near-horizontal wall-light ring, sized to the maze.

        Identical to `MazeForageVision._add_wall_lights` except for the ring
        radius, which the parent derives from the grid's ``maze_extent``: here
        it is twice the maze's full footprint (``4 * layout.extent``), the
        parent's ratio.  The lights are directional, so the radius only places
        them outside the maze; ``N.L`` shading depends on ``dir`` alone.
        """
        n_lights = int(self._config.get("wall_lights", 0))
        if n_lights <= 0:
            return
        radius = 4.0 * float(self._layout.extent)
        elevation = float(self._config.get("wall_light_elevation", 0.35))
        for i in range(n_lights):
            angle = 2.0 * np.pi * i / n_lights
            direction = [np.cos(angle), np.sin(angle), -elevation]
            light = self._spec.worldbody.add_light(
                name=f"wall_light_{i}",
                pos=[-direction[0] * radius, -direction[1] * radius, radius * 0.5],
                dir=direction,
                diffuse=[float(self._config.get("wall_light_diffuse", 0.45))] * 3,
                specular=[0.0, 0.0, 0.0],
                castshadow=0,
            )
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    # ------------------------------------------------------------------
    # Start gate (hold_enabled only)
    # ------------------------------------------------------------------

    def _gate_bar_edges(self) -> list[tuple[float, float]]:
        """``(x0, x1)`` of each vertical bar across the corridor, left to right (gate frame).

        Bars are ``gate_bar_width`` wide from the left wall; the last one is
        cut at the right wall if the width does not divide the corridor.
        """
        width = float(self._maze_spec.corridor_width)
        bar = float(self._config.gate_bar_width)
        n_bars = int(np.ceil(width / bar - 1e-9))
        edges = []
        for i in range(n_bars):
            x0 = -0.5 * width + i * bar
            x1 = min(x0 + bar, 0.5 * width)
            if x1 - x0 > 1e-9:
                edges.append((x0, x1))
        return edges

    def _check_gate_config(self) -> None:
        """Validates the gate's own keys (the hold keys: `spout_hold.make_hold_params`).

        Raises:
            ValueError: if ``gate_thickness`` or ``gate_bar_width`` is not
                positive, or the bars are so wide that fewer than two fit across
                the corridor (no stripes).
        """
        if float(self._config.gate_thickness) <= 0.0:
            raise ValueError(
                f"gate_thickness={self._config.gate_thickness} must be positive."
            )
        if float(self._config.gate_bar_width) <= 0.0 or len(self._gate_bar_edges()) < 2:
            raise ValueError(
                f"gate_bar_width={self._config.gate_bar_width} must be positive and "
                f"leave at least two bars across the {self._maze_spec.corridor_width} m "
                "corridor (the gate face must be striped)."
            )

    def _build_gate(self) -> None:
        """Adds the start gate body (must precede ``add_rodent``).

        One body on the stem centreline at ``(start_x, 0, 0)`` with a ``y`` and
        a ``z`` slide joint (``damping=1e8``, ``stiffness=0``, like the treats),
        so per-world placement goes through ``qpos`` (``geom_pos`` edits are
        not per world on warp).  Its geoms, in the body frame (y = 0 is the
        boundary line, z = 0 the nominal floor, from which the walls are sunk
        by ``mfv._WALL_FLOOR_EMBED``; the gate is sunk by the same amount, so
        every geom spans the walls' z range ``[-embed, wall_height - embed]``
        and no slot is left under the closed gate):

        * ``gate_box``: the collision box, clear corridor width x
          ``gate_thickness`` x ``wall_height``, centred on the boundary line,
          dark grey.  ``contype=0, conaffinity=1`` collides with the rat's
          ``contype=1`` geoms; a body ``exclude`` with the world body keeps it
          off the walls (whose inner faces it touches) and the floor.
        * ``gate_bar_{i}``: non-colliding 1 mm slabs on the ``start`` (-y) side,
          1 mm in front of the box, full height, alternating bright white and
          dark grey (left to right, starting bright).
        """
        cfg = self._config
        width = float(self._maze_spec.corridor_width)
        height = float(cfg.wall_height)
        half_t = 0.5 * float(cfg.gate_thickness)
        # Centre height of every gate geom: the walls' (layout z = height / 2,
        # less the floor embed `_build_y_maze_walls` applies).
        centre_z = 0.5 * height - mfv._WALL_FLOOR_EMBED
        slide_range = float(
            max(float(self._layout.extent), self._park_depth + height) + width
        )
        body = self._spec.worldbody.add_body(
            name=_GATE_BODY, pos=[float(self._layout.start_xy[0]), 0.0, 0.0]
        )
        for name, axis in zip(_GATE_JOINTS, ((0, 1, 0), (0, 0, 1))):
            body.add_joint(
                name=name,
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=list(axis),
                range=[-slide_range, slide_range],
                damping=1e8,
                stiffness=0,
            )
        body.add_geom(
            name=_GATE_BOX_GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * width, half_t, 0.5 * height],
            pos=[0.0, 0.0, centre_z],
            rgba=list(_GATE_DARK_RGBA),
            contype=0,
            conaffinity=1,
        )
        bar_y = -(half_t + _GATE_BAR_GAP + _GATE_BAR_HALF_DEPTH)
        for i, (x0, x1) in enumerate(self._gate_bar_edges()):
            body.add_geom(
                name=_GATE_BAR_GEOM.format(i),
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.5 * (x1 - x0), _GATE_BAR_HALF_DEPTH, 0.5 * height],
                pos=[0.5 * (x0 + x1), bar_y, centre_z],
                rgba=list(_GATE_BRIGHT_RGBA if i % 2 == 0 else _GATE_DARK_RGBA),
                contype=0,
                conaffinity=0,
            )
        self._spec.add_exclude(
            bodyname1=self._spec.worldbody.name, bodyname2=_GATE_BODY
        )

    def _cache_gate_indices(self) -> None:
        """Caches the gate's qpos / dof addresses, body and geom ids and local offsets.

        Raises:
            ValueError: If a gate joint, body or geom is missing (``mj_name2id``
                returns -1, which would silently index the last element).
        """
        model = self._mj_model

        def lookup(obj, name):
            idx = mujoco.mj_name2id(model, obj, name)
            if idx < 0:
                raise ValueError(f"'{name}' not found in the compiled model.")
            return idx

        joints = [lookup(mujoco.mjtObj.mjOBJ_JOINT, n) for n in _GATE_JOINTS]
        body_id = lookup(mujoco.mjtObj.mjOBJ_BODY, _GATE_BODY)
        box_id = lookup(mujoco.mjtObj.mjOBJ_GEOM, _GATE_BOX_GEOM)
        bar_ids = [
            lookup(mujoco.mjtObj.mjOBJ_GEOM, _GATE_BAR_GEOM.format(i))
            for i in range(len(self._gate_bar_edges()))
        ]
        geom_ids = np.asarray([box_id, *bar_ids], dtype=np.int32)

        self._gate_qpos_idxs = jp.asarray(model.jnt_qposadr[joints], dtype=jp.int32)
        self._gate_dof_idxs = jp.asarray(model.jnt_dofadr[joints], dtype=jp.int32)
        self._gate_body_id = int(body_id)
        self._gate_box_geom_id = int(box_id)
        self._gate_bar_geom_ids = np.asarray(bar_ids, dtype=np.int32)
        self._gate_geom_ids = jp.asarray(geom_ids)
        # The body hangs off worldbody with identity orientation and only
        # translates, so its world pose is the reference pose plus the slide
        # offsets, and every geom / the inertial frame keeps its body-frame
        # offset (read from the model, not assumed).
        self._gate_body_ref = jp.asarray(model.body_pos[body_id], dtype=jp.float32)
        self._gate_body_ipos = jp.asarray(model.body_ipos[body_id], dtype=jp.float32)
        self._gate_geom_local = jp.asarray(model.geom_pos[geom_ids], dtype=jp.float32)
        self._gate_edges_y = jp.asarray(
            self._layout.segments.stem_edges_y, dtype=jp.float32
        )

    def _gate_pose(self, hs: spout_hold.HoldState) -> tuple[jp.ndarray, jp.ndarray]:
        """``(y, z)`` slide offsets of the gate for ``hs``: its boundary and `spout_hold.gate_z`."""
        return self._gate_edges_y[hs.gate_k], spout_hold.gate_z(hs, self._hold_params)

    def _place_gate(self, data: mjx.Data, y: jp.ndarray, z: jp.ndarray) -> mjx.Data:
        """Writes the gate's slide ``qpos`` = ``(y, z)``, zeroes its ``qvel``, and its derived pose.

        As `_park_collected_treats` does for the treats: ``qpos`` alone would
        leave ``xpos`` / ``xipos`` / ``geom_xpos`` stale until the next physics
        step, and the renderer reads ``geom_xpos``.  The gate only translates,
        so the exact pose is available in closed form.  Zeroing ``qvel`` every
        step means contact forces never carry it anywhere.

        Args:
            data: Simulation data (a single world).
            y: Gate boundary y (world, m).
            z: Gate height offset (m; 0 = closed).

        Returns:
            ``data`` with the gate rows rewritten (idempotent).
        """
        offset = jp.stack(
            [jp.float32(0.0), jp.asarray(y, jp.float32), jp.asarray(z, jp.float32)]
        )
        qpos = data.qpos.at[self._gate_qpos_idxs].set(
            offset[1:].astype(data.qpos.dtype)
        )
        qvel = data.qvel.at[self._gate_dof_idxs].set(0.0)
        body = self._gate_body_ref + offset
        xpos = data.xpos.at[self._gate_body_id].set(body)
        xipos = data.xipos.at[self._gate_body_id].set(body + self._gate_body_ipos)
        geom_xpos = data.geom_xpos.at[self._gate_geom_ids].set(
            body[None, :] + self._gate_geom_local
        )
        return data.replace(
            qpos=qpos, qvel=qvel, xpos=xpos, xipos=xipos, geom_xpos=geom_xpos
        )

    @staticmethod
    def _hold_metrics(events: dict[str, jp.ndarray]) -> dict[str, jp.ndarray]:
        return {f"spout/{name}": events[name] for name in HOLD_METRICS}

    # ------------------------------------------------------------------
    # Start pose (shared by reset and the teleport)
    # ------------------------------------------------------------------

    def _sample_start_yaw(self, rng: jax.Array) -> jp.ndarray:
        """``start_yaw`` plus a uniform ``+-start_yaw_jitter`` draw (radians)."""
        jitter = jax.random.uniform(
            rng, (), minval=-self._start_yaw_jitter, maxval=self._start_yaw_jitter
        )
        return jp.float32(self._start_yaw) + jitter

    def _start_pose_data(
        self,
        data: mjx.Data,
        yaw: jp.ndarray,
        gate_yz: tuple[jp.ndarray, jp.ndarray] | None = None,
    ) -> mjx.Data:
        """``data`` with the start-of-trial inputs written, then forwarded.

        Writes ``qpos`` = ``qpos0`` with the root at ``(start_xy,
        spawn_height)`` facing ``yaw`` and both treats parked on their spouts
        (and, given ``gate_yz``, the gate's slide offsets), and zeroes ``qvel``,
        ``ctrl``, ``act`` and ``qacc_warmstart``.  Those are
        the inputs ``mjx.make_data`` hands a fresh ``reset``, so a forwarded
        teleport and a forwarded reset agree.  ``mjx.forward`` is mandatory:
        every ``xpos``-based obs, reward and the render read derived fields.
        """
        root = self._rodent_root_qpos
        start = jp.asarray(
            [self._layout.start_xy[0], self._layout.start_xy[1], self._spawn_height],
            dtype=jp.float32,
        )
        qpos = self._qpos0.at[root : root + 3].set(start)
        qpos = qpos.at[root + 3 : root + 7].set(_yaw_quat(yaw))
        qpos = qpos.at[self._treat_slide_qpos_idxs].set(self._parked_treat_offsets)
        if gate_yz is not None:
            qpos = qpos.at[self._gate_qpos_idxs].set(
                jp.stack(gate_yz).astype(qpos.dtype)
            )
        data = data.replace(
            qpos=qpos.astype(data.qpos.dtype),
            qvel=jp.zeros_like(data.qvel),
            ctrl=jp.zeros_like(data.ctrl),
            act=jp.zeros_like(data.act),
            qacc_warmstart=jp.zeros_like(data.qacc_warmstart),
        )
        return mjx.forward(self.mjx_model, data)

    @staticmethod
    def _where_data(
        pred: jp.ndarray, on_true: mjx.Data, on_false: mjx.Data
    ) -> mjx.Data:
        """Per-env ``jnp.where(pred, on_true, on_false)`` over every per-world ``Data`` field.

        Selects every public ``mjx.Data`` field and every field of the warp
        ``Data._impl`` EXCEPT those in ``mujoco.mjx.warp.types.DATA_NON_VMAP``,
        which keep ``on_false``'s value.  Those are buffers shared by ALL worlds
        (the contact pool, leading dim ``naconmax``, and its counters): warp's own
        vmap rule leaves them unbatched, a per-env select cannot apply to them,
        and trying leaks a batch tracer out of that rule.
        ``BraxAutoResetWrapper``'s ``where_done`` skips the same leaves.  They
        are rebuilt by the next physics step's collision pass and nothing in the
        observation or the render reads them.

        The per-world ``_impl`` fields MUST be selected too: the renderer reads
        ``_impl.light_xpos``, and the rodent carries a light, so keeping the
        stepped value shifts ~9% of the start-view pixels by one grey level.
        """

        def select(t, f):
            return jax.tree.map(lambda a, b: jp.where(pred, a, b), t, f)

        impl_t, impl_f = on_true._impl, on_false._impl
        impl = dataclasses.replace(
            impl_f,
            **{
                f.name: select(getattr(impl_t, f.name), getattr(impl_f, f.name))
                for f in dataclasses.fields(impl_f)
                if f.name not in mjx_warp_types.DATA_NON_VMAP
            },
        )
        public = {
            f.name: select(getattr(on_true, f.name), getattr(on_false, f.name))
            for f in dataclasses.fields(on_true)
            if f.name != "_impl"
        }
        return on_false.replace(_impl=impl, **public)

    def _live_treats(self, ts: spout_logic.TrialState) -> jp.ndarray:
        """``(n_treats,)`` bool: treat ``i`` is rendered (unparked).

        The locked arm's treat once it is shown and not yet eaten.  With
        ``treat_visible_until_teleport`` it also stays rendered once eaten, for
        as long as the post-eating dwell runs (``dwell >= 0``): the trial end
        resets ``choice`` / ``treat_shown`` / ``dwell``, so the teleport step
        parks it.  The flag is a Python bool, so with it off this is exactly the
        expression of the flagless env.
        """
        arm = jp.arange(self._n_treats, dtype=jp.int32)
        if self._treat_visible_until_teleport:
            return ts.treat_shown & (arm == ts.choice) & (~ts.eaten | (ts.dwell >= 0))
        return ts.treat_shown & ~ts.eaten & (arm == ts.choice)

    def _torso_segment(self, data: mjx.Data) -> jp.ndarray:
        """int32 `spout_geometry.segment_of` of the torso xy (-1 throughout in the Y)."""
        return spout_geometry.segment_of(self._layout, self._torso(data).xpos[:2])

    @staticmethod
    def _spout_metrics(events: dict[str, jp.ndarray]) -> dict[str, jp.ndarray]:
        return {f"spout/{name}": events[name] for name in SPOUT_METRICS}

    # ------------------------------------------------------------------
    # Core environment interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """New episode: draw the bait, put the rat at the start, park both treats.

        With ``hold_enabled`` the episode starts in a hold (drawn here), with
        the gate closed on its drawn boundary.

        Args:
            rng: Random number generator key.

        Returns:
            The initial environment state.
        """
        rng, bait_rng, yaw_rng = jax.random.split(rng, 3)
        ts = spout_logic.init_trial_state(bait_rng, self._params)
        hs = gate = None
        if self._hold_enabled:
            rng, hold_rng = jax.random.split(rng)
            hs = spout_hold.hold_init(hold_rng, self._hold_params)
            gate = self._gate_pose(hs)

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = self._start_pose_data(data, self._sample_start_yaw(yaw_rng), gate)
        if hs is not None:
            data = self._place_gate(data, *gate)

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "step_count": jp.array(0, dtype=jp.int32),
            "collected": jp.ones((self._n_treats,), dtype=bool),
            "n_collected": jp.array(0, dtype=jp.int32),
            "belly_up_steps": jp.array(0, dtype=jp.int32),
            "low_torso_steps": jp.array(0, dtype=jp.int32),
            "trial": ts._asdict(),
            "critic_priv": spout_logic.critic_privileged_vector(
                ts, self._critic_priv_max_trials
            ),
            # Payout / jitter key stream; the bait used its own key above.
            "spout_rng": rng,
            "segment": self._torso_segment(data),
        }
        zero_events = {name: jp.float32(0.0) for name in (*SPOUT_METRICS, "reward")}

        metrics = self._spout_metrics(zero_events)
        if hs is not None:
            info["hold"] = hs._asdict()
            metrics.update(
                self._hold_metrics(
                    {
                        "hold_active": spout_hold.holding(hs).astype(jp.float32),
                        "gate_release": jp.float32(0.0),
                    }
                )
            )
        obs = self._get_obs(data, info)
        info[_EVENTS_KEY] = zero_events
        reward = self._get_reward(data, info, metrics)
        del info[_EVENTS_KEY]
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Physics, trial update, reward, teleport on trial end, then the obs.

        Termination and reward read the stepped (pre-teleport) state.  The
        teleport, the treat show/park and the observation come last, so the
        frame rendered from ``state.data`` is the start view on a trial-end step
        and shows a treat on the step it appears.

        With ``hold_enabled``: the trial state and its events are frozen while
        the step starts in the hold; the hold is advanced (and re-armed on a
        trial end) before the teleport, the teleport closes the gate on the new
        boundary, and the gate pose is written after the teleport, before the
        treats and the obs.

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

        # --- Trial state machine (fresh keys every step) ---
        rng, pay_rng, yaw_rng = jax.random.split(info["spout_rng"], 3)
        if self._hold_enabled:
            rng, hold_rng = jax.random.split(rng)
            hs = spout_hold.HoldState(**info["hold"])
        info["spout_rng"] = rng
        ts = spout_logic.TrialState(**info["trial"])
        ts_prev = ts
        torso_xy = self._torso(data).xpos[:2]
        arm_id, dist_to_end = spout_geometry.locate(self._layout, torso_xy)
        # The treat that can be eaten is the locked arm's, or (on the step the
        # choice locks) the arm the rat is in.
        treat_arm = jp.where(
            ts.choice >= 0, ts.choice, jp.clip(arm_id, 0, self._n_treats - 1)
        )
        treat_dist = jp.linalg.norm(torso_xy - self._treat_xy[treat_arm])
        params = (
            self._params._replace(trial_timeout_steps=0)
            if self._force_trial_end
            else self._params
        )
        ts, events = spout_logic.trial_update(
            ts, arm_id, dist_to_end, treat_dist, pay_rng, params
        )
        if self._hold_enabled:
            # During the hold the trial clock, its timeout and every event are
            # frozen (the reward reads the frozen events too).
            ts, events = spout_hold.freeze_trial(
                spout_hold.holding(hs), ts_prev, ts, events
            )
        trial_end = events["trial_end"] > 0.0

        metrics = state.metrics
        metrics.update(self._spout_metrics(events))
        if self._hold_enabled:
            # Count down / animate; a trial end re-arms (new hold and boundary,
            # gate closed) for the teleport below.
            hs, hold_events = spout_hold.hold_update(
                hs, trial_end, hold_rng, self._hold_params
            )
            metrics.update(self._hold_metrics(hold_events))
        done = self._is_done(data, info, metrics)
        info[_EVENTS_KEY] = events
        reward = self._get_reward(data, info, metrics)
        del info[_EVENTS_KEY]
        reward = jp.nan_to_num(reward)
        info["n_collected"] = info["n_collected"] + events["reward"].astype(jp.int32)

        # --- Teleport on trial end: the start pose, as a fresh reset builds it ---
        gate = self._gate_pose(hs) if self._hold_enabled else None
        start = self._start_pose_data(data, self._sample_start_yaw(yaw_rng), gate)
        data = self._where_data(trial_end, start, data)
        zero_action = jp.zeros_like(info["action"])
        info["action"] = jp.where(trial_end, zero_action, info["action"])
        info["prev_action"] = jp.where(trial_end, zero_action, info["prev_action"])
        for key in ("belly_up_steps", "low_torso_steps"):
            info[key] = jp.where(trial_end, jp.zeros_like(info[key]), info[key])

        # --- The gate (closed form, every step: contacts never move it) ---
        if self._hold_enabled:
            data = self._place_gate(data, *gate)
            info["hold"] = hs._asdict()

        # --- Show / park the treats (closed form, no forward needed) ---
        parked = ~self._live_treats(ts)
        info["collected"] = parked
        data = self._park_collected_treats(data, parked)

        info["trial"] = ts._asdict()
        info["critic_priv"] = spout_logic.critic_privileged_vector(
            ts, self._critic_priv_max_trials
        )
        info["segment"] = self._torso_segment(data)  # post-teleport torso
        obs = self._get_obs(data, info)

        return state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )

    # ------------------------------------------------------------------
    # Reward and terminations (this module's registry only)
    # ------------------------------------------------------------------

    @_registry.reward("spout_treat")
    def _spout_treat_reward(self, data, info, metrics, weight) -> jp.ndarray:
        """``+weight`` on the step a shown treat is reached (`trial_update`'s ``reward``).

        Reads the step's trial events from the transient ``info`` key that
        ``step`` / ``reset`` set around ``_get_reward``.
        """
        del data
        if _EVENTS_KEY not in info:
            raise KeyError(
                f"spout_treat reads info[{_EVENTS_KEY!r}], which only exists inside "
                "RodentSpoutMazeVision.step / reset."
            )
        reward_val = weight * info[_EVENTS_KEY]["reward"]
        metrics["rewards/treat_collected"] = reward_val
        return reward_val

    @_registry.termination("belly_up")
    def _spout_belly_up_termination(self, data, info, **kwargs) -> jp.ndarray:
        """The parent's ``belly_up`` (inverted for ``patience`` steps)."""
        return mfv.MazeForageVision._belly_up_termination(self, data, info, **kwargs)

    @_registry.termination("torso_too_low")
    def _spout_torso_too_low_termination(self, data, info, **kwargs) -> jp.ndarray:
        """The parent's ``torso_too_low`` (collapsed for ``patience`` steps)."""
        return mfv.MazeForageVision._torso_too_low_termination(
            self, data, info, **kwargs
        )

    @_registry.termination("nan_termination")
    def _spout_nan_termination(self, data, info) -> jp.ndarray:
        """The parent's ``nan_termination`` (qpos / qvel only, never the whole Data)."""
        return mfv.MazeForageVision._nan_termination(self, data, info)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def layout(self) -> spout_geometry.YMazeLayout:
        """The maze geometry (walls, start pose, arm ends, treat points)."""
        return self._layout

    @property
    def maze_shape(self) -> str:
        """``"Y"`` or ``"T"``."""
        return self._maze_shape

    @property
    def spout_params(self) -> spout_logic.SpoutParams:
        """The trial state machine's parameters."""
        return self._params

    @property
    def corridor_width(self) -> float:
        """Clear width of every corridor, in metres."""
        return float(self._maze_spec.corridor_width)

    @property
    def hold_enabled(self) -> bool:
        """True when the inter-trial hold and the start gate are built."""
        return self._hold_enabled

    @property
    def hold_params(self) -> spout_hold.HoldParams:
        """The hold / gate parameters (``hold_enabled`` only)."""
        if not self._hold_enabled:
            raise AttributeError(
                "hold_params: this env was built with hold_enabled=False."
            )
        return self._hold_params

    def gate_boundary_y(self, k: int) -> float:
        """World y of stem boundary ``k`` (between stem segments ``k - 1`` and ``k``; T maze)."""
        return float(self._layout.segments.stem_edges_y[int(k)])

    def _no_grid(self, *_):
        raise AttributeError(
            "RodentSpoutMazeVision has no grid maze; use `layout` for its geometry."
        )

    maze_grid = property(_no_grid)
    maze_walls = property(_no_grid)
    n_free_cells = property(_no_grid)
    open_cell_fraction = property(_no_grid)
    treat_cell_fraction = property(_no_grid)
    free_cell_positions = property(_no_grid)
