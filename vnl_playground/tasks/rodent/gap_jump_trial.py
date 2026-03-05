"""Discrete-trial gap-jumping task for distance estimation.

Replicates the Liska et al. mouse gap-jumping paradigm in silico.
Each episode = one trial: HOLD -> DECISION -> JUMP -> OUTCOME.

Trial phases:
  HOLD (0)     : Rodent on take-off platform, must stay still. Vision occluded.
  DECISION (1) : "Barrier lifted" - vision active. Agent estimates gap distance.
  JUMP (2)     : Rodent has left take-off platform, airborne / landing.
"""

import collections
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
from vnl_playground.tasks.task_registry import TaskRegistry

_registry = TaskRegistry()

# Trial phase codes (integers for JAX tracing)
PHASE_HOLD = 0
PHASE_DECISION = 1
PHASE_JUMP = 2

# Trial outcome codes
OUTCOME_ONGOING = 0
OUTCOME_SUCCESS = 1
OUTCOME_FAILURE = 2
OUTCOME_ABORT = 3
OUTCOME_TIMEOUT = 4


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
        use_mesh_platforms=False,
        # Trial parameters
        gap_distances=(0.06, 0.08, 0.10, 0.12, 0.14),
        hold_duration=50,
        max_decision_steps=300,
        spawn_x=0.0,
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
                material="platform_mat",
                contype=1,
                conaffinity=1,
            )
        else:
            takeoff_body.add_geom(
                name="takeoff_platform_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(takeoff_half),
                material="platform_mat",
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
                material="platform_mat",
                contype=1,
                conaffinity=1,
            )
        else:
            landing_body.add_geom(
                name="landing_platform_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(landing_half),
                material="platform_mat",
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

        # Sample gap distance uniformly from the configured set
        n_distances = len(self._config.gap_distances)
        gap_idx = jax.random.randint(gap_rng, shape=(), minval=0, maxval=n_distances)
        gap_distance = self._gap_distances_array[gap_idx]

        # Compute slide joint offset: 0 -> max gap, negative -> smaller gap
        slide_offset = -(self._max_gap - gap_distance)

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "trial_phase": jp.array(PHASE_HOLD, dtype=jp.int32),
            "step_count": jp.array(0, dtype=jp.int32),
            "decision_start_step": jp.array(-1, dtype=jp.int32),
            "gap_distance": gap_distance,
            "jump_initiated": jp.array(False),
            "trial_success": jp.array(False),
            "trial_outcome": jp.array(OUTCOME_ONGOING, dtype=jp.int32),
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

        # Detect landing success: torso past landing leading edge and not fallen
        landing_leading_x = self._takeoff_trailing_edge_x + info["gap_distance"]
        landed = (
            (torso_x > landing_leading_x) & (torso_z > -0.1) & info["jump_initiated"]
        )
        info["trial_success"] = jp.where(landed, True, info["trial_success"])

        # --- Trial outcome tracking ---
        is_ongoing = info["trial_outcome"] == OUTCOME_ONGOING
        info["trial_outcome"] = jp.where(
            is_ongoing & landed, OUTCOME_SUCCESS, info["trial_outcome"]
        )
        torso_fallen = torso_z < -0.1
        info["trial_outcome"] = jp.where(
            is_ongoing & torso_fallen & ~landed, OUTCOME_FAILURE, info["trial_outcome"]
        )
        behind_platform = torso_x < -self._config.takeoff_platform_length / 2.0
        past_hold = new_phase >= PHASE_DECISION
        info["trial_outcome"] = jp.where(
            is_ongoing & behind_platform & past_hold,
            OUTCOME_ABORT,
            info["trial_outcome"],
        )

        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

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

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
                phase_indicator,
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
        # length_scale controls how quickly reward drops off
        length_scale = 0.3  # ~0.3m characteristic distance
        proximity = jp.exp(-dist / length_scale)

        is_active = (info["trial_phase"] >= PHASE_DECISION).astype(jp.float32)
        reward_val = weight * proximity * is_active
        metrics["rewards/target_proximity"] = reward_val
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
