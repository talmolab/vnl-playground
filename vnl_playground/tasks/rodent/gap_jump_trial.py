"""Discrete-trial gap-jumping task for distance estimation.

Replicates the Liska et al. mouse gap-jumping paradigm in silico.
Each episode = one trial: HOLD -> DECISION -> JUMP -> OUTCOME.

Trial phases:
  HOLD (0)     : Rodent on take-off platform, must stay still. Vision occluded.
  DECISION (1) : "Barrier lifted" - vision active. Agent estimates gap distance.
  JUMP (2)     : Rodent has left take-off platform, airborne / landing.
"""

import collections
import pathlib
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.rodent.utils.box_to_mesh import box_to_mesh_asset
from vnl_playground.tasks.reward_registry import RewardRegistry

_registry = RewardRegistry()

# Trial phase codes (integers for JAX tracing)
PHASE_HOLD = 0
PHASE_DECISION = 1
PHASE_JUMP = 2

# Trial outcome codes (PHYSICAL / execution axis: did the body land or fall)
OUTCOME_ONGOING = 0
OUTCOME_SUCCESS = 1
OUTCOME_FAILURE = 2
OUTCOME_ABORT = 3
OUTCOME_TIMEOUT = 4

# Signal-detection outcome codes (DECISION axis: jump vs withhold, scored against
# the gap's ground-truth reachability). INDEPENDENT of the physical axis above --
# a reachable gap that is jumped is a Hit here even if the body later mis-lands
# (that is the W3 execution/calibration axis). W2 go/no-go + d'/criterion use these.
SDT_ONGOING = 0
SDT_HIT = 1             # reachable gap + jumped
SDT_MISS = 2            # reachable gap + withheld
SDT_FALSE_ALARM = 3     # un-reachable gap + jumped
SDT_CORRECT_REJECT = 4  # un-reachable gap + withheld


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the GapJumpTrial environment."""
    return config_dict.create(
        walker_xml_path=consts.RODENT_NO_TAIL_COLLISION_XML,
        arena_xml_path=consts.GAP_JUMP_ARENA_XML_PATH,
        ctrl_dt=0.02,
        sim_dt=0.002,
        solver="newton",
        mujoco_impl="warp",
        naconmax=19 * 1024,
        njmax=400,
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        # Arena geometry
        takeoff_platform_length=0.3,
        takeoff_platform_width=0.3,
        takeoff_platform_thickness=0.16,
        landing_platform_depth=0.3,
        landing_platform_max_width=0.3,
        landing_height_offset=0.02,
        # Safe-landing margin (metres): shrinks the safe zone inward from the
        # landing platform's far edge. 0.0 = the whole platform top is safe;
        # larger values force the rat to touch down closer to the near edge,
        # i.e. demand more accurate distance estimation. Touching down past
        # (far_edge - margin) counts as an overjump and fails the trial.
        landing_safe_margin=0.0,
        # Success = a real four-paw touchdown ON the landing platform, not a torso
        # fly-over. Requires all four paw touch sensors (palm_L/R, sole_L/R) in
        # contact (force > landing_touch_eps) AND all four paws past the near edge
        # (on the platform, not bridging the gap), held for landing_dwell_steps
        # consecutive control steps. A paw off the platform side hangs over the
        # void -> no contact -> fails, which also supplies lateral centering.
        landing_touch_eps=1e-3,
        landing_dwell_steps=3,
        # How many paws must be down on the platform to count as a landing.
        # 4 = a full four-paw touchdown; lower (e.g. 3) to relax the criterion.
        landing_min_paws=4,
        # Success = torso crosses onto the platform and STAYS UP this many
        # consecutive control steps (300 = 3 s at ctrl_dt 0.01). The survival
        # requirement filters out fly-overs / drape-and-fall / veer-off-the-side.
        landing_survive_steps=300,
        # Graded landing reward: extra reward (on top of paws_landed.weight) once
        # ALL 4 paws are on the platform. paws_landed.weight fires at >=2 paws.
        landing_paw_bonus=0.0,
        # option-b terminate: end the episode once all 4 paws have been on the
        # landing platform for this many consecutive control steps (a confirmed
        # 4-paw landing). 30 = 0.3 s at ctrl_dt 0.01. Only active when the
        # `paw_landed` termination criterion is enabled in the config.
        landing_paw_dwell_steps=30,
        use_mesh_platforms=False,
        # Trial parameters
        # Default = easy, all-reachable gaps (Scott's Phase-0 "basic single jump"
        # baseline). W2 go/no-go needs un-reachable (> max_reachable_gap) gaps too,
        # but those are supplied per-run via config override (the eval harness sets
        # gap_distances=tuple(eval_grid); W2 training sets its own mix) -- NOT baked
        # into the code default. Keeps "config, not code" (Scott's guardrail).
        gap_distances=(0.06, 0.08, 0.10, 0.12, 0.14),
        # SDT ground-truth reachability label: a gap is "reachable" (signal
        # present) iff its distance <= this. Drives Hit/Miss vs FA/CR scoring.
        # Provisional 0.16 m (middle of the 14-18 cm ambiguous band); recalibrate
        # from the trained rat's measured max reach.
        max_reachable_gap=0.16,
        # Eval-only knob: force the gap to a specific INDEX into gap_distances
        # (-1 = random sampling, the training default). The W2 eval driver builds
        # the env with gap_distances=tuple(eval_grid) and sweeps this index, so the
        # psychometric x-axis is the REAL gap that ran. (The old eval runner
        # recorded a requested width while reset() independently sampled a random
        # gap -- making the x-axis fiction.)
        eval_fixed_gap_idx=-1,
        hold_duration=50,
        max_decision_steps=300,
        spawn_x=0.0,
        # --- Target / waypoint system ---
        target_position_mode="landing_center",  # "landing_center", "landing_round_trip", "fixed", "waypoints"
        fixed_target_position=(0.5, 0.0, 0.0),
        target_waypoints=(),
        max_waypoints=4,
        target_reach_threshold=0.05,
        auto_advance_waypoint=True,
        loop_waypoints=False,
        # Aesthetic
        aesthetic="default",  # "default" or "outdoor_natural"
        # Episode limits
        episode_length=500,
        action_repeat=1,
        # Reward terms
        reward_terms={
            "hold_stillness": {"weight": 0.3},
            "jump_success": {"weight": 100.0},
            "landing_bonus": {"weight": 50.0},
            "fall_penalty": {"weight": -10.0},
            "abort_penalty": {"weight": -20.0},
            "time_penalty": {"weight": -0.01},
        },
        # Termination criteria
        termination_criteria={
            "fallen": {"min_torso_z": -0.1},
            "trial_success": {},
            "trial_failure": {},
            "abort_dismount": {},
            "trial_timeout": {"max_steps": 500},
            "nan_termination": {},
        },
    )


def dense_config() -> config_dict.ConfigDict:
    """Returns the legacy dense-reward configuration (pre-eLife redesign)."""
    cfg = default_config()
    cfg.reward_terms = {
        "hold_stillness": {"weight": 0.3},
        "forward_displacement": {"weight": 1.0},
        "approach_velocity": {"weight": 0.5},
        "jump_success": {"weight": 10.0},
        "landing_bonus": {"weight": 5.0},
        "decision_alive": {"weight": 0.05},
        "abort_penalty": {"weight": -1.0},
        "fall_penalty": {"weight": -5.0},
    }
    cfg.termination_criteria = {
        "fallen": {"min_torso_z": -0.15},
        "trial_timeout": {"max_steps": 500},
        "nan_termination": {},
    }
    return cfg


class GapJumpTrial(rodent_base.RodentEnv):
    """Discrete-trial gap-jumping environment.

    Implements the Liska et al. paradigm: the rodent stands on a take-off
    platform, a barrier is lifted to reveal the gap, and the rodent must
    estimate the distance and jump to the landing platform.

    The landing platform has a 1-DOF slide joint so the gap distance can be
    varied per episode by setting the joint offset in qpos at reset time.
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the GapJumpTrial environment.

        Args:
            rng: Random number generator key.
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        # Default platform material (from gap_jump_arena.xml)
        self._platform_material = "platform_mat"

        # Apply aesthetic textures before building arena
        if self._config.get("aesthetic", "default") == "outdoor_natural":
            self._apply_outdoor_natural_aesthetic()

        # Build take-off and landing platforms
        self._build_arena()

        # Place rodent on take-off platform center facing forward (+x)
        init_x = self._config.spawn_x
        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[init_x, 0.0, 0.0],
            quat=(1, 0, 0, 0),
        )
        # Lighting is defined in gap_jump_arena.xml (key_light + headlight)
        self.compile()

        # The landing_slide joint prepends extra elements to qpos/qvel.
        # Record where the rodent's joints start so proprioception getters
        # slice correctly (the base class assumes qpos[7:] / qvel[6:]).
        root_jnt_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        self._rodent_qpos_start = self._mj_model.jnt_qposadr[root_jnt_id] + 7
        self._rodent_qvel_start = self._mj_model.jnt_dofadr[root_jnt_id] + 6
        self._rodent_root_dof = self._mj_model.jnt_dofadr[root_jnt_id]

        # Store joint index for landing platform slide
        self._landing_slide_qpos_idx = self._mj_model.jnt_qposadr[
            mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "landing_slide"
            )
        ]

        # Store geom IDs for contact detection
        self._takeoff_geom_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "takeoff_platform_geom"
        )
        self._landing_geom_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "landing_platform_geom"
        )

        # Precompute gap distance array as JAX array
        self._gap_distances_array = jp.array(self._config.gap_distances)
        self._max_gap = float(max(self._config.gap_distances))
        self._max_waypoints = self._config.get("max_waypoints", 4)

        # Take-off platform trailing edge x position
        self._takeoff_trailing_edge_x = self._config.takeoff_platform_length / 2.0

    def _add_mesh_for_box(self, name: str, half_extents: tuple) -> str:
        """Register a mesh asset equivalent to a box and return its name."""
        verts, faces, texcoords = box_to_mesh_asset(half_extents)
        mesh = self._spec.add_mesh()
        mesh.name = name
        mesh.uservert = verts.flatten()
        mesh.userface = faces.flatten()
        mesh.usertexcoord = texcoords.flatten()
        return name

    def _apply_outdoor_natural_aesthetic(self) -> None:
        """Apply outdoor natural aesthetic: grass platforms, blue sky, better lighting."""
        assets_dir = pathlib.Path(__file__).parent / "xmls" / "assets"
        self._spec.compiler.texturedir = str(assets_dir)

        # --- Skybox ---
        for tex in list(self._spec.textures):
            if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                tex.delete()

        self._spec.add_texture(
            name="outdoor_skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            file="OutdoorSkybox2048.png",
            gridsize=[3, 4],
            gridlayout=".U..LFRB.D..",
        )

        # --- Ground/Platform texture ---
        self._spec.add_texture(
            name="grass_texture",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file="OutdoorGrassFloorD.png",
        )

        grass_mat = self._spec.add_material(name="grass_mat")
        grass_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grass_texture"
        grass_mat.texuniform = True

        self._platform_material = "grass_mat"

        # --- Headlight for outdoor scene ---
        self._spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self._spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        self._spec.visual.headlight.specular = [0.1, 0.1, 0.1]

        # Disable dark fog
        self._spec.visual.map.fogstart = 10.0
        self._spec.visual.map.fogend = 20.0
        self._spec.visual.rgba.fog = [0.0, 0.0, 0.0, 0.0]

    def _build_arena(self) -> None:
        """Build take-off and landing platforms in the arena spec.

        The take-off platform is a static body at the origin.  The landing
        platform has a 1-DOF slide joint along x so its position can be set
        at each reset to produce different gap distances.  High damping and
        stiffness keep the platform effectively locked during simulation.
        """
        cfg = self._config
        half_thickness = cfg.takeoff_platform_thickness / 2.0
        use_mesh = cfg.get("use_mesh_platforms", False)

        # --- Take-off platform (static body) ---
        takeoff_body = self._spec.worldbody.add_body(
            name="takeoff_platform",
            pos=[0.0, 0.0, -half_thickness],
        )
        takeoff_half = (
            cfg.takeoff_platform_length / 2,
            cfg.takeoff_platform_width / 2,
            half_thickness,
        )
        if use_mesh:
            mesh_name = self._add_mesh_for_box("takeoff_mesh", takeoff_half)
            takeoff_body.add_geom(
                name="takeoff_platform_geom",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                material=self._platform_material,
                contype=1,
                conaffinity=1,
            )
        else:
            takeoff_body.add_geom(
                name="takeoff_platform_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(takeoff_half),
                material=self._platform_material,
                contype=1,
                conaffinity=1,
            )

        # --- Landing platform with slide joint ---
        max_gap = max(cfg.gap_distances)
        min_gap = min(cfg.gap_distances)
        # Place landing body at the position corresponding to the maximum gap
        landing_x = (
            cfg.takeoff_platform_length / 2 + max_gap + cfg.landing_platform_depth / 2
        )

        landing_body = self._spec.worldbody.add_body(
            name="landing_platform",
            pos=[landing_x, 0.0, -half_thickness + cfg.landing_height_offset],
        )
        # Slide joint: offset=0 -> max gap; negative offset -> smaller gap
        landing_body.add_joint(
            name="landing_slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=[-(max_gap - min_gap), 0],
            damping=1e8,
            stiffness=0,
        )
        landing_half = (
            cfg.landing_platform_depth / 2,
            cfg.landing_platform_max_width / 2,
            half_thickness,
        )
        if use_mesh:
            mesh_name = self._add_mesh_for_box("landing_mesh", landing_half)
            landing_body.add_geom(
                name="landing_platform_geom",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                material=self._platform_material,
                contype=1,
                conaffinity=1,
            )
        else:
            landing_body.add_geom(
                name="landing_platform_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(landing_half),
                material=self._platform_material,
                contype=1,
                conaffinity=1,
            )
        # Black edge strip on leading edge (visual only, no collision)
        landing_body.add_geom(
            name="landing_edge_strip",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.002, cfg.landing_platform_max_width / 2, 0.005],
            pos=[-cfg.landing_platform_depth / 2, 0, half_thickness],
            material="edge_strip_mat",
            contype=0,
            conaffinity=0,
        )

    # ---- Proprioception overrides ----
    # The landing_slide joint prepends extra elements to qpos / qvel.
    # The base-class getters assume the rodent motor joints start at
    # qpos[7:] / qvel[6:], so we override them here.

    def _get_joint_angles(self, data: mjx.Data) -> jp.ndarray:
        return data.qpos[self._rodent_qpos_start :]

    def _get_joint_ang_vels(self, data: mjx.Data) -> jp.ndarray:
        return data.qvel[self._rodent_qvel_start :]

    def _get_actuator_ctrl(self, data: mjx.Data) -> jp.ndarray:
        return data.qfrc_actuator[self._rodent_root_dof :]

    # ------------------------------------------------------------------
    # Core environment interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset with a randomly sampled gap distance.

        Args:
            rng: JAX random key.

        Returns:
            Initial environment state.
        """
        rng, gap_rng = jax.random.split(rng)

        # Gap distance: random over the configured set (training default), or
        # forced to a specific index for controlled W2 eval. The branch is on a
        # STATIC config value (read at trace time), so it is trace-safe.
        n_distances = len(self._config.gap_distances)
        fixed_idx = int(self._config.get("eval_fixed_gap_idx", -1))
        if fixed_idx >= 0:
            gap_idx = jp.array(fixed_idx % n_distances, dtype=jp.int32)
        else:
            gap_idx = jax.random.randint(
                gap_rng, shape=(), minval=0, maxval=n_distances
            )
        gap_distance = self._gap_distances_array[gap_idx]

        # Compute slide joint offset: 0 -> max gap, negative -> smaller gap
        slide_offset = -(self._max_gap - gap_distance)

        # --- Build target waypoints based on mode ---
        waypoints = jp.zeros((self._max_waypoints, 3))
        landing_leading_x = self._takeoff_trailing_edge_x + gap_distance
        landing_center_x = landing_leading_x + self._config.landing_platform_depth / 2.0
        landing_center_z = self._config.landing_height_offset

        if self._config.target_position_mode == "landing_center":
            waypoints = waypoints.at[0].set(
                jp.array([landing_center_x, 0.0, landing_center_z])
            )
            num_waypoints = jp.array(1, dtype=jp.int32)

        elif self._config.target_position_mode == "landing_round_trip":
            waypoints = waypoints.at[0].set(
                jp.array([landing_center_x, 0.0, landing_center_z])
            )
            waypoints = waypoints.at[1].set(
                jp.array([self._config.spawn_x, 0.0, 0.0])
            )
            num_waypoints = jp.array(2, dtype=jp.int32)

        elif self._config.target_position_mode == "fixed":
            waypoints = waypoints.at[0].set(
                jp.array(self._config.fixed_target_position)
            )
            num_waypoints = jp.array(1, dtype=jp.int32)

        elif self._config.target_position_mode == "waypoints":
            cfg_wps = self._config.target_waypoints
            n = min(len(cfg_wps), self._max_waypoints)
            for i in range(n):
                waypoints = waypoints.at[i].set(jp.array(cfg_wps[i]))
            num_waypoints = jp.array(n, dtype=jp.int32)

        else:
            num_waypoints = jp.array(1, dtype=jp.int32)

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "trial_phase": jp.array(PHASE_HOLD, dtype=jp.int32),
            "step_count": jp.array(0, dtype=jp.int32),
            "decision_start_step": jp.array(-1, dtype=jp.int32),
            "gap_distance": gap_distance,
            "jump_initiated": jp.array(False),
            "trial_success": jp.array(False),
            # Consecutive control steps with all four paws settled on the platform.
            "landing_dwell": jp.array(0, dtype=jp.int32),
            # DEBUG: how many paws are down on the landing platform this step.
            "n_paws_down": jp.array(0, dtype=jp.int32),
            # option-b terminate: consecutive steps with all 4 paws on the platform,
            # and the latched "confirmed 4-paw landing" flag it drives.
            "paw_dwell": jp.array(0, dtype=jp.int32),
            "paw_landed": jp.array(False),
            "trial_outcome": jp.array(OUTCOME_ONGOING, dtype=jp.int32),
            # SDT decision-axis tracking (independent of the physical outcome above)
            "gap_reachable": gap_distance <= self._config.max_reachable_gap,
            "sdt_outcome": jp.array(SDT_ONGOING, dtype=jp.int32),
            "decision_time": jp.array(0, dtype=jp.int32),
            # Waypoint / target system
            "target_waypoints": waypoints,
            "num_waypoints": num_waypoints,
            "current_waypoint_idx": jp.array(0, dtype=jp.int32),
            "target_position": waypoints[0],
            "target_reached": jp.array(False),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        # Set landing platform position via the slide joint
        new_qpos = data.qpos.at[self._landing_slide_qpos_idx].set(slide_offset)
        data = data.replace(qpos=new_qpos)
        data = mjx.forward(self.mjx_model, data)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        # Initialize trial outcome metrics so the pytree structure matches step()
        metrics["trial/outcome"] = jp.float32(0.0)
        metrics["trial/success"] = jp.float32(0.0)
        metrics["trial/failure"] = jp.float32(0.0)
        metrics["trial/abort"] = jp.float32(0.0)
        metrics["sdt/hit"] = jp.float32(0.0)
        metrics["sdt/miss"] = jp.float32(0.0)
        metrics["sdt/false_alarm"] = jp.float32(0.0)
        metrics["sdt/correct_reject"] = jp.float32(0.0)
        metrics["sdt/gap_reachable"] = jp.float32(0.0)
        metrics["sdt/decision_time"] = jp.float32(0.0)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step with trial phase management.

        Phases transition as:
          HOLD -> DECISION  (after hold_duration steps)
          DECISION -> JUMP  (torso passes take-off trailing edge)

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            Updated environment state.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action

        # Increment step count
        step_count = info["step_count"] + 1
        info["step_count"] = step_count

        # Get torso position for phase transitions
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        torso_x = torso.xpos[0]
        torso_z = torso.xpos[2]

        # Phase transition: HOLD -> DECISION (after hold_duration steps)
        phase = info["trial_phase"]
        new_phase = jp.where(
            (phase == PHASE_HOLD) & (step_count >= self._config.hold_duration),
            PHASE_DECISION,
            phase,
        )

        # Record decision start step on transition
        decision_start = jp.where(
            (phase == PHASE_HOLD) & (new_phase == PHASE_DECISION),
            step_count,
            info["decision_start_step"],
        )
        info["decision_start_step"] = decision_start

        # Phase transition: DECISION -> JUMP (torso passes take-off trailing edge)
        new_phase = jp.where(
            (new_phase == PHASE_DECISION) & (torso_x > self._takeoff_trailing_edge_x),
            PHASE_JUMP,
            new_phase,
        )
        info["jump_initiated"] = jp.where(
            new_phase == PHASE_JUMP, True, info["jump_initiated"]
        )

        info["trial_phase"] = new_phase

        # Detect landing within the SAFE landing zone. Crossing the near
        # (leading) edge alone is NOT success any more: the torso must touch
        # down BETWEEN the near edge and the far edge (minus a safety margin).
        # Overshooting past the far edge is an overjump (failed distance
        # estimation) and is treated as a trial failure, which forces the rat
        # to estimate the gap distance from vision rather than jumping maximally.
        landing_leading_x = self._takeoff_trailing_edge_x + info["gap_distance"]
        safe_margin = self._config.get("landing_safe_margin", 0.0)
        landing_far_x = (
            landing_leading_x + self._config.landing_platform_depth - safe_margin
        )
        # Four-paw touchdown: all four paw touch sensors in contact AND all four
        # paws on the landing platform (x past the near edge -> not a gap-bridge),
        # with a jump initiated. A paw veering off the 0.4 m platform side hangs
        # over the void -> no contact -> fails (this also supplies the lateral /
        # centering constraint the old torso-x band check lacked). Held for
        # landing_dwell_steps so a single impact/bounce frame is not a "landing".
        # Success = the torso fully crosses onto the landing platform and STAYS UP
        # (does not fall) for landing_survive_steps consecutive control steps
        # (300 = 3 s at ctrl_dt 0.01). A fly-over / drape-and-fall / veer-off-the-
        # side cannot hold this for 3 s, so the survival requirement enforces a real
        # landing on top of the natural torso-crossing motion -- no paw counting.
        on_platform = torso_z > -0.1
        # In the safe landing band [near, far) -- same as fwdvel's success zone.
        crossed = (
            (torso_x > landing_leading_x)
            & (torso_x < landing_far_x)
            & info["jump_initiated"]
        )
        surviving = crossed & on_platform
        dwell = jp.where(surviving, info["landing_dwell"] + 1, 0)
        info["landing_dwell"] = dwell
        # survive_steps=1 -> success/terminate the instant it crosses (= fwdvel);
        # =100 -> must stay up 1 s (@ctrl_dt 0.01) before success/terminate.
        survive_steps = int(self._config.get("landing_survive_steps", 1))
        landed = dwell >= survive_steps
        overshot = (torso_x >= landing_far_x) & info["jump_initiated"]
        info["trial_success"] = jp.where(landed, True, info["trial_success"])

        # Paw count ON the landing platform (for the graded landing reward): a paw
        # counts if its touch sensor is in contact AND it is past the near edge (on
        # the platform, not over the void). A paw that veers off the platform SIDE
        # hangs over the void -> not counted -> so this also rewards lateral centering.
        _touch = self._get_touch_sensors(data)  # [palm_L, palm_R, sole_L, sole_R]
        _teps = self._config.get("landing_touch_eps", 1e-3)
        _paw_x = jp.array(
            [
                data.bind(self.mjx_model, self._spec.body(f"{b}{self._suffix}")).xpos[0]
                for b in ("hand_L", "hand_R", "foot_L", "foot_R")
            ]
        )
        _paw_down = (_touch.reshape(-1) > _teps) & (_paw_x > landing_leading_x)
        info["n_paws_down"] = jp.sum(_paw_down.astype(jp.int32))

        # option-b terminate: count consecutive steps with all 4 paws on the
        # platform; latch paw_landed once that dwell reaches landing_paw_dwell_steps
        # (a confirmed 4-paw landing). Drives the `paw_landed` termination.
        paw_dwell = jp.where(info["n_paws_down"] >= 4, info["paw_dwell"] + 1, 0)
        info["paw_dwell"] = paw_dwell
        paw_dwell_steps = int(self._config.get("landing_paw_dwell_steps", 30))
        info["paw_landed"] = jp.where(
            paw_dwell >= paw_dwell_steps, True, info["paw_landed"]
        )

        # --- Trial outcome tracking ---
        is_ongoing = info["trial_outcome"] == OUTCOME_ONGOING
        info["trial_outcome"] = jp.where(
            is_ongoing & landed, OUTCOME_SUCCESS, info["trial_outcome"]
        )
        torso_fallen = torso_z < -0.1
        # Failure = fell into the gap (underjump) OR overshot the far edge
        # (overjump). Either way the rat misjudged the jump distance.
        info["trial_outcome"] = jp.where(
            is_ongoing & (torso_fallen | overshot) & ~landed,
            OUTCOME_FAILURE,
            info["trial_outcome"],
        )
        behind_platform = torso_x < -self._config.takeoff_platform_length / 2.0
        past_hold = new_phase >= PHASE_DECISION
        info["trial_outcome"] = jp.where(
            is_ongoing & behind_platform & past_hold,
            OUTCOME_ABORT,
            info["trial_outcome"],
        )

        # --- Signal-detection (DECISION axis) scoring ---
        # Decision = jump vs withhold, scored against ground-truth reachability,
        # INDEPENDENT of the physical land/fall outcome above. Logic verified in
        # scratchpad/sdt_env_logic_prototype.py before porting.
        reachable = info["gap_reachable"]
        jumped = info["jump_initiated"]
        decision_start = info["decision_start_step"]
        # Deliberation time: accrues while in DECISION and not yet committed,
        # frozen once the jump is initiated (raw data for T3 SPRT/dwell).
        in_decision_window = (new_phase == PHASE_DECISION) & ~jumped
        info["decision_time"] = jp.where(
            in_decision_window & (decision_start >= 0),
            step_count - decision_start,
            info["decision_time"],
        )
        # A jump resolves the decision immediately; a withhold resolves once the
        # decision window (max_decision_steps) has elapsed without a jump.
        sdt_ongoing = info["sdt_outcome"] == SDT_ONGOING
        withhold_resolved = step_count >= self._config.max_decision_steps
        info["sdt_outcome"] = jp.where(
            sdt_ongoing & jumped & reachable, SDT_HIT, info["sdt_outcome"]
        )
        info["sdt_outcome"] = jp.where(
            sdt_ongoing & jumped & ~reachable, SDT_FALSE_ALARM, info["sdt_outcome"]
        )
        info["sdt_outcome"] = jp.where(
            sdt_ongoing & ~jumped & withhold_resolved & reachable,
            SDT_MISS,
            info["sdt_outcome"],
        )
        info["sdt_outcome"] = jp.where(
            sdt_ongoing & ~jumped & withhold_resolved & ~reachable,
            SDT_CORRECT_REJECT,
            info["sdt_outcome"],
        )

        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        # --- Waypoint advancement (after reward) ---
        if self._config.get("auto_advance_waypoint", True):
            dist_to_target = jp.linalg.norm(torso.xpos - info["target_position"])
            at_target = dist_to_target < self._config.get(
                "target_reach_threshold", 0.05
            )

            if self._config.get("loop_waypoints", False):
                should_advance = at_target
                new_idx = jp.where(
                    should_advance,
                    (info["current_waypoint_idx"] + 1) % info["num_waypoints"],
                    info["current_waypoint_idx"],
                )
                info["target_reached"] = jp.array(False)
            else:
                can_advance = info["current_waypoint_idx"] < (
                    info["num_waypoints"] - 1
                )
                should_advance = at_target & can_advance
                new_idx = jp.where(
                    should_advance,
                    info["current_waypoint_idx"] + 1,
                    info["current_waypoint_idx"],
                )
                info["target_reached"] = at_target & ~can_advance

            info["current_waypoint_idx"] = new_idx
            info["target_position"] = info["target_waypoints"][new_idx]

        # --- Outcome metrics ---
        metrics = state.metrics
        metrics["trial/outcome"] = info["trial_outcome"].astype(float)
        metrics["trial/success"] = (info["trial_outcome"] == OUTCOME_SUCCESS).astype(
            float
        )
        metrics["trial/failure"] = (info["trial_outcome"] == OUTCOME_FAILURE).astype(
            float
        )
        metrics["trial/abort"] = (info["trial_outcome"] == OUTCOME_ABORT).astype(float)
        metrics["sdt/hit"] = (info["sdt_outcome"] == SDT_HIT).astype(float)
        metrics["sdt/miss"] = (info["sdt_outcome"] == SDT_MISS).astype(float)
        metrics["sdt/false_alarm"] = (
            info["sdt_outcome"] == SDT_FALSE_ALARM
        ).astype(float)
        metrics["sdt/correct_reject"] = (
            info["sdt_outcome"] == SDT_CORRECT_REJECT
        ).astype(float)
        metrics["sdt/gap_reachable"] = info["gap_reachable"].astype(float)
        metrics["sdt/decision_time"] = info["decision_time"].astype(float)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> collections.OrderedDict:
        """Get observations with body-state signals and phase indicator as task_obs.

        ``task_obs`` contains body-state signals (prev_action, kinematic
        sensors, touch sensors, origin) concatenated with a 3-dim phase
        indicator one-hot vector.  This richer signal is compatible with the
        vision+task_obs fusion architecture used in ``RunGapVision``.

        The agent observation does NOT include gap_distance -- the agent must
        rely on vision (or the privileged state) to infer the gap width.

        Args:
            data: Simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with ``state`` and ``privileged_state`` keys.
        """
        phase = info.get("trial_phase", jp.array(PHASE_HOLD, dtype=jp.int32))
        phase_indicator = jax.nn.one_hot(phase, 3)

        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        # Egocentric vector to target position
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        target_pos = info.get("target_position", jp.zeros(3))
        rel_target_world = target_pos - torso.xpos
        ego_target = jp.dot(rel_target_world, torso.xmat)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
                phase_indicator,
                ego_target,
            ]
        )

        proprioception = self._get_proprioception(data, info, flatten=False)

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=proprioception,
        )

        privileged_obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=proprioception,
            gap_distance=jp.array(info.get("gap_distance", 0.0)).reshape(1),
            target_position=info.get("target_position", jp.zeros(3)),
        )

        return collections.OrderedDict(
            state=obs,
            privileged_state=privileged_obs,
        )

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    @_registry.reward("hold_stillness")
    def _hold_stillness_reward(self, data, info, metrics, weight):
        """Reward for staying still during HOLD phase.

        Uses an exponential kernel on the torso velocity magnitude so that
        near-zero velocity yields a reward close to ``weight``.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        velocity_magnitude = jp.sqrt(jp.sum(torso.subtree_linvel**2))
        stillness = jp.exp(-10.0 * velocity_magnitude)
        is_hold = (info["trial_phase"] == PHASE_HOLD).astype(jp.float32)
        reward_val = weight * stillness * is_hold
        metrics["rewards/hold_stillness"] = reward_val
        return reward_val

    @_registry.reward("jump_success")
    def _jump_success_reward(self, data, info, metrics, weight):
        """Sparse reward for successful landing on the landing platform."""
        reward_val = weight * info["trial_success"].astype(jp.float32)
        metrics["rewards/jump_success"] = reward_val
        return reward_val

    @_registry.reward("decision_alive")
    def _decision_alive_reward(self, data, info, metrics, weight):
        """Small per-step bonus during DECISION phase to encourage survival."""
        is_decision = (info["trial_phase"] == PHASE_DECISION).astype(jp.float32)
        reward_val = weight * is_decision
        metrics["rewards/decision_alive"] = reward_val
        return reward_val

    @_registry.reward("abort_penalty")
    def _abort_penalty_reward(self, data, info, metrics, weight):
        """Penalty for moving backward off the take-off platform.

        Fires when the torso x-position falls behind the take-off platform
        trailing edge (negative x direction).  The ``weight`` config value
        should be negative to act as a penalty.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        behind = (torso.xpos[0] < -self._config.takeoff_platform_length / 2.0).astype(
            jp.float32
        )
        reward_val = weight * behind  # weight is negative
        metrics["rewards/abort_penalty"] = reward_val
        return reward_val

    @_registry.reward("forward_displacement")
    def _forward_displacement_reward(self, data, info, metrics, weight):
        """Dense reward for forward displacement during DECISION phase.

        Rewards the rodent's x-position relative to its spawn position,
        normalized by the distance to the gap edge. Only active during
        DECISION and JUMP phases.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        torso_x = torso.xpos[0]

        # Normalize displacement by takeoff platform length
        # (rodent starts at spawn_x=0, platform edge is at takeoff_platform_length/2)
        platform_edge = self._takeoff_trailing_edge_x
        normalized_progress = jp.clip(torso_x / platform_edge, 0.0, 2.0)

        is_active = (info["trial_phase"] >= PHASE_DECISION).astype(jp.float32)
        reward_val = weight * normalized_progress * is_active
        metrics["rewards/forward_displacement"] = reward_val
        return reward_val

    @_registry.reward("approach_velocity")
    def _approach_velocity_reward(self, data, info, metrics, weight):
        """Reward for forward velocity during DECISION phase.

        Encourages the rodent to build momentum toward the gap.
        Clamped to [0, 1] relative to a target approach speed.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        forward_vel = torso.subtree_linvel[0]

        target_speed = 0.3  # m/s approach speed target
        vel_reward = jp.clip(forward_vel / target_speed, 0.0, 1.0)

        is_active = (info["trial_phase"] >= PHASE_DECISION).astype(jp.float32)
        reward_val = weight * vel_reward * is_active
        metrics["rewards/approach_velocity"] = reward_val
        return reward_val

    @_registry.reward("edge_proximity")
    def _edge_proximity_reward(self, data, info, metrics, weight):
        """Dense reward for approaching the gap edge during DECISION phase.

        Provides a smooth, increasing signal as the torso gets closer to
        the takeoff trailing edge. Uses a quadratic ramp that saturates
        at the edge, giving the agent a strong gradient toward the gap.
        Active only during DECISION phase (not JUMP, to avoid rewarding
        overshoot before the agent commits).
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        torso_x = torso.xpos[0]

        # Distance from spawn (0) to edge, normalized to [0, 1]
        edge_x = self._takeoff_trailing_edge_x
        progress = jp.clip(torso_x / edge_x, 0.0, 1.0)

        # Quadratic ramp: gentle at start, steep near edge
        proximity = progress**2

        is_decision = (info["trial_phase"] == PHASE_DECISION).astype(jp.float32)
        reward_val = weight * proximity * is_decision
        metrics["rewards/edge_proximity"] = reward_val
        return reward_val

    @_registry.reward("target_proximity")
    def _target_proximity_reward(self, data, info, metrics, weight):
        """Dense reward based on inverse distance to landing platform center.

        Provides a continuous gradient toward the landing target. Uses an
        exponential kernel so the reward increases sharply as the rodent
        approaches the target, giving a strong signal near the landing zone.

        Active during DECISION and JUMP phases.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        torso_x = torso.xpos[0]

        # Target is the center of the landing platform
        landing_leading_x = self._takeoff_trailing_edge_x + info["gap_distance"]
        target_x = landing_leading_x + self._config.landing_platform_depth / 2.0

        # Distance to target, clamped to avoid negative rewards
        dist = jp.abs(target_x - torso_x)

        # Exponential proximity: 1.0 at target, decays with distance
        # length_scale controls how quickly reward drops off. Larger scale = a
        # broader, flatter pull that reaches the rat from farther back (and raises
        # the cumulative value, so pair it with a lower weight). Config-overridable.
        length_scale = self._config.get("target_proximity_length_scale", 0.3)
        proximity = jp.exp(-dist / length_scale)

        is_active = (info["trial_phase"] >= PHASE_DECISION).astype(jp.float32)
        reward_val = weight * proximity * is_active
        metrics["rewards/target_proximity"] = reward_val
        return reward_val

    @_registry.reward("go_to_target")
    def _go_to_target_reward(self, data, info, metrics, weight):
        """Dense reward for moving toward the active target position.

        Active during DECISION and JUMP phases (gated by trial phase).
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        target_pos = info.get("target_position", jp.zeros(3))
        dist = jp.linalg.norm(torso.xpos - target_pos)

        length_scale = 0.3
        proximity = jp.exp(-dist / length_scale)

        is_active = (info["trial_phase"] >= PHASE_DECISION).astype(jp.float32)
        reward_val = weight * proximity * is_active
        metrics["rewards/go_to_target"] = reward_val
        metrics["rewards/target_distance"] = dist
        return reward_val

    @_registry.reward("landing_bonus")
    def _landing_bonus_reward(self, data, info, metrics, weight):
        """Bonus reward for landing that scales with gap distance.

        Harder gaps (larger distance) yield higher reward. This encourages
        the agent to attempt challenging jumps rather than only easy ones.
        """
        gap_dist = info["gap_distance"]
        max_gap = self._max_gap
        # Scale: min_gap -> 1.0x, max_gap -> 2.0x
        difficulty_scale = 1.0 + (gap_dist / max_gap)
        reward_val = (
            weight * difficulty_scale * info["trial_success"].astype(jp.float32)
        )
        metrics["rewards/landing_bonus"] = reward_val
        return reward_val

    @_registry.reward("fall_penalty")
    def _fall_penalty_reward(self, data, info, metrics, weight):
        """Penalty when the torso drops below the platform surface.

        Provides negative reward signal when the agent falls into the gap,
        helping it learn to avoid the gap edge without sufficient momentum.
        The weight should be negative.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        fallen = (torso.xpos[2] < -0.1).astype(jp.float32)
        reward_val = weight * fallen
        metrics["rewards/fall_penalty"] = reward_val
        return reward_val

    @_registry.reward("time_penalty")
    def _time_penalty_reward(self, data, info, metrics, weight):
        """Small per-step penalty during DECISION phase."""
        is_decision = (info["trial_phase"] == PHASE_DECISION).astype(jp.float32)
        reward_val = weight * is_decision
        metrics["rewards/time_penalty"] = reward_val
        return reward_val

    @_registry.reward("energy_cost")
    def _energy_cost_reward(self, data, info, metrics, weight):
        """Metabolic cost proxy = muscular effort = sum |actuator torque| over the
        ACTUATED DOFs (skips the root free joint, so forward translation is NOT
        penalised -- only effort). Uses torque magnitude (not mechanical work
        tau*omega) ON PURPOSE: holding a static reared/reaching pose against gravity
        costs sustained torque but ~0 work, so a work penalty would NOT discourage
        it -- a torque penalty does. weight should be NEGATIVE. Shapes an efficient,
        natural low-effort gait (cost-of-transport) and penalises effortful poses.
        """
        dof = self._rodent_root_dof
        effort = jp.sum(jp.abs(data.qfrc_actuator[dof:]))
        reward_val = weight * effort
        metrics["rewards/energy_cost"] = reward_val
        return reward_val

    @_registry.reward("paws_landed")
    def _paws_landed_reward(self, data, info, metrics, weight):
        """Per-paw landing reward (Keming's design): give `weight` for EACH paw on
        the landing platform, so more paws -> more reward
        (0,1,2,3,4 paws -> 0,w,2w,3w,4w). A smooth monotonic gradient toward a full
        four-paw landing. n_paws_down only counts paws on the platform, so a rat
        veering off the SIDE loses paws -> this also pushes lateral centering.
        Per step, so staying landed keeps paying (fights falling off).
        """
        n = info["n_paws_down"].astype(jp.float32)
        reward_val = weight * n
        metrics["rewards/paws_landed"] = reward_val
        return reward_val

    # ------------------------------------------------------------------
    # Termination criteria
    # ------------------------------------------------------------------

    @_registry.termination("fallen")
    def _fallen_termination(self, data, info, min_torso_z=-0.15):
        """Terminate if torso drops below minimum height."""
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        return torso.xpos[2] < min_torso_z

    @_registry.termination("trial_timeout")
    def _trial_timeout(self, data, info, max_steps=500):
        """Terminate if step count exceeds maximum."""
        return info["step_count"] >= max_steps

    @_registry.termination("trial_success")
    def _trial_success_termination(self, data, info):
        """Terminate immediately when the rodent successfully lands."""
        return info.get("trial_success", jp.array(False))

    @_registry.termination("paw_landed")
    def _paw_landed_termination(self, data, info):
        """option-b terminate: end the episode once all 4 paws have been on the
        landing platform for landing_paw_dwell_steps consecutive steps (a confirmed
        4-paw landing). Unlike trial_success (which fires the instant the torso
        crosses the near edge), this waits for a real settled touchdown, giving the
        per-paw landing reward time to shape the final paws-down pose."""
        return info.get("paw_landed", jp.array(False))

    @_registry.termination("trial_failure")
    def _trial_failure_termination(self, data, info):
        """Terminate immediately when the trial is marked a failure.

        A failure is either an underjump (torso fell into the gap) or an
        overjump (torso overshot past the landing platform's far edge). Both
        are encoded as OUTCOME_FAILURE in step(); terminating here ends the
        episode the moment the rat misjudges the jump distance.
        """
        outcome = info.get("trial_outcome", jp.array(OUTCOME_ONGOING, dtype=jp.int32))
        return outcome == OUTCOME_FAILURE

    @_registry.termination("abort_dismount")
    def _abort_dismount_termination(self, data, info):
        """Terminate when the rodent walks backward off the take-off platform."""
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        behind = torso.xpos[0] < -self._config.takeoff_platform_length / 2.0
        past_hold = info["trial_phase"] >= PHASE_DECISION
        return behind & past_hold

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info):
        """Terminate on NaN values in simulation data."""
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        return jp.sum(jp.isnan(flattened_vals)) > 0

    @_registry.termination("reached_target")
    def _reached_target_termination(self, data, info):
        """Terminate when all waypoints are completed."""
        return info.get("target_reached", jp.array(False))

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def null_action(self) -> jp.ndarray:
        """Return zero action vector."""
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0])

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        return jax.tree_util.tree_map(
            lambda x: jp.prod(jp.array(x.shape)), abstract_state.obs
        )
