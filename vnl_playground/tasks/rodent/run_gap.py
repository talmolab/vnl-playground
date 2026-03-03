"""RunGap corridor task for virtual rodent.

The rodent must run forward (+x direction) across platforms separated by gaps.
Platforms are procedurally generated box geoms added to a corridor arena that
has side walls but no floor.

Reward is based on forward velocity along the corridor (+x direction).

Termination occurs if:
- Torso becomes too tilted or falls below the platforms (fallen)
- Agent fails to make minimum forward displacement over patience window (stale_location)
- NaN detected in simulation data
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
from vnl_playground.tasks.task_registry import TaskRegistry

_registry = TaskRegistry()

_WALL_THICKNESS = 0.16


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the RunGap environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        walker_xml_path=consts.RODENT_NO_TAIL_COLLISION_XML,
        arena_xml_path=consts.CORRIDOR_ARENA_XML_PATH,
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
        corridor_length=4.0,
        corridor_width=2.0,
        platform_length_range=(0.3, 0.6),
        gap_length_range=(0.03, 0.12),
        n_platforms=10,
        episode_length=2000,
        action_repeat=1,
        spawn_x=0.5,
        randomize_gaps=True,
        aesthetic="default",
        use_mesh_platforms=False,
        reward_terms={
            "forward_velocity": {"weight": 1.0, "target_speed": 0.3},
            "termination_penalty": {"weight": 10.0},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.01, "max_torso_angle": 70},
            # "stale_location": {"patience": 100, "min_displacement": 0.03},
            "nan_termination": {},
        },
    )


class RunGap(rodent_base.RodentEnv):
    """RunGap corridor environment.

    The rodent must run forward (+x direction) across platforms separated by
    gaps (voids). Platforms are procedurally generated and added as box geoms
    to a corridor arena.
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the RunGap environment.

        Args:
            rng: Random number generator key.
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        # Default platform material (from corridor_arena.xml)
        self._platform_material = "platform_mat"

        # Apply aesthetic textures before building corridor so platforms use
        # the correct material from the start
        if self._config.aesthetic == "outdoor_natural":
            self._apply_outdoor_natural_aesthetic()

        # Build the corridor platforms before adding the rodent
        self._build_corridor()

        # Initialize rodent on the starting platform facing forward (+x)
        init_x = self._config.spawn_x
        init_y = 0.0
        init_z = 0.0
        init_quat = (1, 0, 0, 0)

        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
        )

        # Lighting — uses directional light for consistent illumination
        # in both CPU and warp renderers (warp ignores light_diffuse/specular
        # and has no distance attenuation for directional lights)
        self._spec.worldbody.add_light(
            name="key_light",
            pos=[2, 1, 8],
            dir=[-0.2, -0.1, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
            castshadow=1,
        )

        if self._config.aesthetic != "outdoor_natural":
            # outdoor_natural already sets headlight in _apply_outdoor_natural_aesthetic
            self._spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
            self._spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
            self._spec.visual.headlight.specular = [0.1, 0.1, 0.1]
        self.compile()

        # Stale location termination config (read at init to avoid tracing issues)
        stale_cfg = dict(self._config.termination_criteria.get("stale_location", {}))
        self._stale_patience = stale_cfg.get("patience", 100)
        self._stale_min_displacement = stale_cfg.get("min_displacement", 0.03)

        if self._config.randomize_gaps:
            # Store slide joint qpos indices and platform body IDs for reset
            self._platform_slide_qpos_idxs = []
            for i in range(self._config.n_platforms):
                jnt_id = mujoco.mj_name2id(
                    self._mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"platform_{i}_slide",
                )
                self._platform_slide_qpos_idxs.append(
                    self._mj_model.jnt_qposadr[jnt_id]
                )
            self._platform_slide_qpos_idxs = jp.array(self._platform_slide_qpos_idxs)

        # Store platform body IDs for reading xpos in observations
        self._platform_body_ids = []
        for i in range(self._config.n_platforms):
            bid = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_BODY, f"platform_{i}"
            )
            self._platform_body_ids.append(bid)
        self._platform_body_ids = jp.array(self._platform_body_ids)

        self._start_platform_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "platform_start"
        )

        if self._config.randomize_gaps:
            # The slide joints shift qpos/qvel layout. Record where the
            # rodent's joints start so proprioception getters slice correctly
            # (the base class assumes qpos[7:] / qvel[6:]).
            root_jnt_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root"
            )
            # Motor joints start right after the 7-element free joint
            self._rodent_qpos_start = self._mj_model.jnt_qposadr[root_jnt_id] + 7
            self._rodent_qvel_start = self._mj_model.jnt_dofadr[root_jnt_id] + 6
            # Root joint DOF address (for qfrc_actuator slicing)
            self._rodent_root_dof = self._mj_model.jnt_dofadr[root_jnt_id]

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

        # Point MjSpec compiler at the assets directory so file= paths resolve
        self._spec.compiler.texturedir = str(assets_dir)

        # --- Skybox ---
        # Remove existing procedural/builtin skybox textures if any
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

        # --- Headlight for outdoor scene (matching dm_control) ---
        self._spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self._spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        self._spec.visual.headlight.specular = [0.1, 0.1, 0.1]

        # Disable dark fog
        self._spec.visual.map.fogstart = 10.0
        self._spec.visual.map.fogend = 20.0
        self._spec.visual.rgba.fog = [0.0, 0.0, 0.0, 0.0]

    def _build_corridor(self) -> None:
        """Procedurally build corridor platforms with gaps.

        Creates a starting platform followed by alternating gaps and platforms.
        When ``randomize_gaps`` is enabled, each platform gets a 1-DOF slide
        joint along x so that gap distances can be varied per episode at reset
        time.  High damping and stiffness lock the joints during simulation.

        When ``randomize_gaps`` is disabled, platforms are placed at fixed
        positions using a deterministic random seed (legacy behaviour).
        """
        half_width = self._config.corridor_width / 2.0
        half_thickness = _WALL_THICKNESS / 2.0
        gap_length_range = self._config.gap_length_range
        platform_length_range = self._config.platform_length_range
        n_platforms = self._config.n_platforms
        randomize = self._config.randomize_gaps

        # Starting platform (always static)
        start_length = 2.0
        x_cursor = 0.0

        body = self._spec.worldbody.add_body(
            name="platform_start",
            pos=[x_cursor, 0.0, -half_thickness],
        )
        if self._config.get("use_mesh_platforms", False):
            mesh_name = self._add_mesh_for_box(
                "platform_start_mesh",
                (start_length / 2.0, half_width, half_thickness),
            )
            body.add_geom(
                name="platform_start_geom",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                material=self._platform_material,
                contype=1,
                conaffinity=1,
            )
        else:
            body.add_geom(
                name="platform_start_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[start_length / 2.0, half_width, half_thickness],
                material=self._platform_material,
                contype=1,
                conaffinity=1,
            )
        self._start_platform_half_length = start_length / 2.0
        x_cursor = start_length / 2.0  # trailing edge of start platform

        if randomize:
            # Place platforms at maximum-spacing reference positions with
            # slide joints.  At reset, negative slide offsets pull platforms
            # leftward to create the sampled (smaller) gaps.
            max_gap = gap_length_range[1]
            max_plat = platform_length_range[1]
            self._platform_half_length = max_plat / 2.0

            # Slide range: each joint can pull its platform leftward by up to
            # the difference between the max and min gap for that single slot.
            # But because offsets are cumulative (shifting platform i also
            # shifts the reference frame for platform i+1), we set a generous
            # per-joint range that accommodates the full cumulative shift.
            max_cumulative = n_platforms * (max_gap - gap_length_range[0])
            self._reference_positions = []

            for i in range(n_platforms):
                ref_center_x = x_cursor + max_gap + max_plat / 2.0
                self._reference_positions.append(ref_center_x)

                plat_body = self._spec.worldbody.add_body(
                    name=f"platform_{i}",
                    pos=[ref_center_x, 0.0, -half_thickness],
                )
                plat_body.add_joint(
                    name=f"platform_{i}_slide",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[1, 0, 0],
                    range=[-max_cumulative, 0],
                    damping=1e8,
                    stiffness=0,
                )
                if self._config.get("use_mesh_platforms", False):
                    mesh_name = self._add_mesh_for_box(
                        f"platform_{i}_mesh",
                        (max_plat / 2.0, half_width, half_thickness),
                    )
                    plat_body.add_geom(
                        name=f"platform_{i}_geom",
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=mesh_name,
                        material=self._platform_material,
                        contype=1,
                        conaffinity=1,
                    )
                else:
                    plat_body.add_geom(
                        name=f"platform_{i}_geom",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[max_plat / 2.0, half_width, half_thickness],
                        material=self._platform_material,
                        contype=1,
                        conaffinity=1,
                    )
                x_cursor = ref_center_x + max_plat / 2.0

            self._reference_positions = jp.array(self._reference_positions)
            self._n_gaps = n_platforms
        else:
            # Legacy: deterministic layout with fixed seed
            rng = np.random.RandomState(42)
            self._platform_positions = [(-(start_length / 2.0), x_cursor)]

            for i in range(n_platforms):
                gap_length = rng.uniform(*gap_length_range)
                x_cursor += gap_length

                plat_length = rng.uniform(*platform_length_range)
                plat_center_x = x_cursor + plat_length / 2.0

                plat_body = self._spec.worldbody.add_body(
                    name=f"platform_{i}",
                    pos=[plat_center_x, 0.0, -half_thickness],
                )
                if self._config.get("use_mesh_platforms", False):
                    mesh_name = self._add_mesh_for_box(
                        f"platform_{i}_mesh",
                        (plat_length / 2.0, half_width, half_thickness),
                    )
                    plat_body.add_geom(
                        name=f"platform_{i}_geom",
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=mesh_name,
                        material=self._platform_material,
                        contype=1,
                        conaffinity=1,
                    )
                else:
                    plat_body.add_geom(
                        name=f"platform_{i}_geom",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[plat_length / 2.0, half_width, half_thickness],
                        material=self._platform_material,
                        contype=1,
                        conaffinity=1,
                    )
                self._platform_positions.append((x_cursor, x_cursor + plat_length))
                x_cursor += plat_length

            # Precompute static gap arrays for legacy mode
            gap_starts, gap_ends, gap_lengths = [], [], []
            for i in range(1, len(self._platform_positions)):
                prev_end = self._platform_positions[i - 1][1]
                curr_start = self._platform_positions[i][0]
                gap_starts.append(prev_end)
                gap_ends.append(curr_start)
                gap_lengths.append(curr_start - prev_end)

            self._static_gap_starts = jp.array(gap_starts)
            self._static_gap_ends = jp.array(gap_ends)
            self._static_gap_lengths = jp.array(gap_lengths)
            self._static_platform_trailing_edges = jp.array(
                [pos[1] for pos in self._platform_positions]
            )
            self._n_gaps = len(gap_starts)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment state.

        When ``randomize_gaps`` is enabled, samples fresh gap lengths from the
        configured range and positions each platform by setting its slide joint
        qpos.  Each episode therefore sees a different corridor layout.

        Args:
            rng: Random number generator state.

        Returns:
            mjx_env.State: The initial environment state after reset.
        """
        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "stale_ref_x": jp.array(self._config.spawn_x, dtype=jp.float32),
            "stale_steps": jp.array(0, dtype=jp.int32),
            "gaps_crossed": jp.array(0, dtype=jp.int32),
            "just_crossed_gap": jp.array(False),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        if self._config.randomize_gaps:
            rng, gap_rng = jax.random.split(rng)
            n = self._config.n_platforms
            max_gap = self._config.gap_length_range[1]
            max_plat = self._config.platform_length_range[1]

            # Sample random gap lengths for this episode
            gap_lengths = jax.random.uniform(
                gap_rng,
                shape=(n,),
                minval=self._config.gap_length_range[0],
                maxval=max_gap,
            )

            # Compute where each platform center should actually be
            start_trailing = self._start_platform_half_length

            # Build actual center positions by scanning forward
            def _scan_positions(x_cursor, gap_len):
                center = x_cursor + gap_len + max_plat / 2.0
                next_cursor = center + max_plat / 2.0
                return next_cursor, center

            _, actual_centers = jax.lax.scan(
                _scan_positions, start_trailing, gap_lengths
            )

            # Slide offset = actual_center - reference_center
            offsets = actual_centers - self._reference_positions

            # Set slide joint qpos values
            new_qpos = data.qpos
            new_qpos = new_qpos.at[self._platform_slide_qpos_idxs].set(offsets)
            data = data.replace(qpos=new_qpos)

            # Run forward kinematics so xpos reflects the new joint positions
            data = mjx.forward(self.mjx_model, data)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward by one timestep.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            mjx_env.State: The new environment state after stepping.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        obs = self._get_obs(data, info)

        info["prev_action"] = info["action"]
        info["action"] = action

        # Stale location tracking: check forward displacement over patience window
        stale_steps = info["stale_steps"] + 1
        torso_body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        current_x = torso_body.xpos[0]
        ref_x = info["stale_ref_x"]

        # At checkpoint: reset counter & reference if agent made enough progress
        at_checkpoint = stale_steps >= self._stale_patience
        made_progress = (current_x - ref_x) >= self._stale_min_displacement
        reset_stale = at_checkpoint & made_progress

        info["stale_steps"] = jp.where(
            reset_stale, jp.array(0, dtype=jp.int32), stale_steps
        )
        info["stale_ref_x"] = jp.where(reset_stale, current_x, ref_x)

        # Gap-crossing detection
        if self._config.randomize_gaps:
            plat_centers = data.xpos[self._platform_body_ids, 0]
            gap_ends = plat_centers - self._platform_half_length
        else:
            gap_ends = self._static_gap_ends
        new_gaps_crossed = jp.sum(current_x > gap_ends).astype(jp.int32)
        info["just_crossed_gap"] = new_gaps_crossed > info["gaps_crossed"]
        info["gaps_crossed"] = new_gaps_crossed

        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    # ---- Proprioception overrides ----
    # When randomize_gaps is enabled, slide joints prepend extra elements to
    # qpos / qvel.  The base-class getters assume the rodent motor joints
    # start at qpos[7:] / qvel[6:], so we override them here.

    def _get_joint_angles(self, data: mjx.Data) -> jp.ndarray:
        if self._config.randomize_gaps:
            return data.qpos[self._rodent_qpos_start :]
        return super()._get_joint_angles(data)

    def _get_joint_ang_vels(self, data: mjx.Data) -> jp.ndarray:
        if self._config.randomize_gaps:
            return data.qvel[self._rodent_qvel_start :]
        return super()._get_joint_ang_vels(data)

    def _get_actuator_ctrl(self, data: mjx.Data) -> jp.ndarray:
        if self._config.randomize_gaps:
            return data.qfrc_actuator[self._rodent_root_dof :]
        return super()._get_actuator_ctrl(data)

    def _get_gap_features(self, data: mjx.Data) -> jp.ndarray:
        """Compute handcrafted gap-aware features for the current state.

        Returns a flat array of features about upcoming gaps relative to the
        rodent's position, plus velocity, height, lateral position, and heading.
        These serve as the encoder input (imitation_target) for the intention
        network - encoding "what terrain lies ahead" into a latent intention.

        Feature layout (16 values):
          [0-2]  gap_distances: distance to leading edge of next 3 gaps
          [3-5]  gap_lengths: lengths of next 3 gaps
          [6-8]  gap_far_edge_distances: distance to far edge (end) of next 3 gaps
          [9]    forward_vel: x velocity
          [10]   lateral_vel: y velocity
          [11]   body_height: height above platform surface
          [12]   lateral_pos: lateral deviation from corridor center
          [13-14] heading: (heading_x, heading_y) from torso rotation
          [15]   platform_edge_dist: distance to trailing edge of current platform

        When ``randomize_gaps`` is enabled, platform positions are read
        dynamically from ``data.xpos`` (reflecting the current slide-joint
        configuration).  Otherwise the precomputed static arrays are used.

        Args:
            data: Current simulation data.

        Returns:
            Flat array of 16 gap-aware feature values.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        rodent_x = torso.xpos[0]
        rodent_y = torso.xpos[1]
        rodent_z = torso.xpos[2]

        n = self._n_gaps
        sentinel_dist = 10.0  # large distance for "no gap ahead"
        sentinel_len = 0.0  # zero length for "no gap"

        if self._config.randomize_gaps:
            # Read actual platform positions from simulation state
            plat_centers = data.xpos[self._platform_body_ids, 0]
            half_plat = self._platform_half_length

            plat_starts = plat_centers - half_plat  # leading edges
            plat_ends = plat_centers + half_plat  # trailing edges

            # Start platform trailing edge
            start_end = (
                data.xpos[self._start_platform_body_id, 0]
                + self._start_platform_half_length
            )

            # Gap geometry from trailing/leading edges
            all_trailing_edges = jp.concatenate([start_end.reshape(1), plat_ends])
            gap_starts_arr = all_trailing_edges[:-1]
            gap_ends_arr = plat_starts
            gap_lengths_arr = gap_ends_arr - gap_starts_arr
        else:
            gap_starts_arr = self._static_gap_starts
            gap_ends_arr = self._static_gap_ends
            gap_lengths_arr = self._static_gap_lengths
            all_trailing_edges = self._static_platform_trailing_edges

        # Find the first gap whose end (= leading edge of next platform)
        # is ahead of the rodent
        next_idx = jp.searchsorted(gap_ends_arr, rodent_x)

        # Distances to leading edge (start) of next 3 gaps
        gap_distances = jp.array(
            [
                jp.where(
                    next_idx + i < n,
                    gap_starts_arr[jp.clip(next_idx + i, 0, n - 1)] - rodent_x,
                    sentinel_dist,
                )
                for i in range(3)
            ]
        )

        # Lengths of next 3 gaps
        gap_lengths = jp.array(
            [
                jp.where(
                    next_idx + i < n,
                    gap_lengths_arr[jp.clip(next_idx + i, 0, n - 1)],
                    sentinel_len,
                )
                for i in range(3)
            ]
        )

        # Distance to far edge (end) of next 3 gaps
        # Replaces former platform_lengths feature (which was always constant in randomized mode)
        gap_far_edge_distances = jp.array(
            [
                jp.where(
                    next_idx + i < n,
                    gap_ends_arr[jp.clip(next_idx + i, 0, n - 1)] - rodent_x,
                    sentinel_dist,
                )
                for i in range(3)
            ]
        )

        # Velocity features
        forward_vel = torso.subtree_linvel[0]  # x velocity
        lateral_vel = torso.subtree_linvel[1]  # y velocity

        # Height above platform surface (surface is at z=0)
        body_height = rodent_z

        # Lateral deviation from corridor center (center is y=0)
        lateral_pos = rodent_y

        # Heading direction (x-y plane components from rotation matrix)
        # xmat row 0 gives the forward direction of the torso
        heading_x = torso.xmat[0, 0]  # cos of heading
        heading_y = torso.xmat[0, 1]  # sin of heading

        # Distance to trailing edge of current platform
        n_trailing = all_trailing_edges.shape[0]
        plat_idx = jp.searchsorted(all_trailing_edges, rodent_x)
        platform_edge_dist = jp.where(
            plat_idx < n_trailing,
            all_trailing_edges[jp.clip(plat_idx, 0, n_trailing - 1)] - rodent_x,
            sentinel_dist,
        )

        return jp.concatenate(
            [
                gap_distances,  # (3,)
                gap_lengths,  # (3,)
                gap_far_edge_distances,  # (3,)
                forward_vel.reshape(1),  # (1,)
                lateral_vel.reshape(1),  # (1,)
                body_height.reshape(1),  # (1,)
                lateral_pos.reshape(1),  # (1,)
                jp.array([heading_x, heading_y]),  # (2,)
                platform_edge_dist.reshape(1),  # (1,)
            ]
        )  # Total: 16

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> collections.OrderedDict:
        """Get the current observation from the simulation data.

        Observation structure for the intention encoder-decoder architecture:
        - imitation_target: Handcrafted gap features (16 values) -> encoder input
          The encoder compresses terrain-ahead information into a latent "intention"
        - proprioception: Body state (joint angles, sensors, etc.) -> decoder input
          Combined with the latent intention to produce actions

        Args:
            data: The simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with state and privileged_state keys, each containing
            imitation_target and proprioception.
        """
        # Gap features -> encoder input (what terrain lies ahead)
        gap_features = self._get_gap_features(data)

        # Body state -> decoder input (combined with latent intention)
        proprioception = self._get_proprioception(data, info, flatten=False)

        obs = collections.OrderedDict(
            imitation_target=gap_features,
            proprioception=proprioception,
        )
        return collections.OrderedDict(
            state=obs,
            privileged_state=obs,
        )

    # ---- Reward functions ----

    @_registry.reward("forward_velocity")
    def _forward_velocity_reward(
        self, data, info, metrics, weight, target_speed=0.3
    ) -> float:
        """Reward for forward velocity along the corridor (+x direction).

        Linear reward proportional to x-velocity, clamped to [0, 1].
        Zero reward for standing still or moving backward, max reward at
        target_speed. Circling yields ~0 average reward since x-velocity
        averages out over a full loop.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Reward weight multiplier.

        Returns:
            Weighted forward velocity reward.
        """
        del info

        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        forward_vel = body.subtree_linvel[0]

        reward_value = jp.clip(forward_vel / target_speed, 0.0, 1.0)

        weighted_reward = reward_value * weight
        metrics["rewards/forward_velocity"] = weighted_reward

        return weighted_reward

    @_registry.reward("termination_penalty")
    def _termination_penalty(self, data, info, metrics, weight) -> float:
        """Negative reward applied on the timestep when the episode terminates.

        Discourages the agent from optimizing for shorter episodes to game
        the speed reward. The penalty is applied once on the terminal step.

        Args:
            data: Simulation data (unused).
            info: State info (unused).
            metrics: Metrics dict; reads terminations/any set by _is_done().
            weight: Penalty magnitude (positive number, applied as negative reward).

        Returns:
            Weighted termination penalty (negative on terminal step, 0 otherwise).
        """
        del data, info
        terminated = metrics.get("terminations/any", 0.0)
        penalty = -weight * terminated
        metrics["rewards/termination_penalty"] = penalty
        return penalty

    @_registry.reward("gap_crossing_bonus")
    def _gap_crossing_bonus(self, data, info, metrics, weight) -> float:
        """Sparse bonus awarded each time the agent crosses a new gap.

        The crossing flag is computed in step() by comparing the agent's
        x-position against gap far-edges (leading edges of landing platforms).

        Args:
            data: Simulation data (unused).
            info: State info containing just_crossed_gap flag.
            metrics: Metrics dict for logging.
            weight: Bonus magnitude per gap crossed.

        Returns:
            Weighted bonus (weight if gap just crossed, 0 otherwise).
        """
        del data
        bonus = jp.where(info.get("just_crossed_gap", False), weight, 0.0)
        metrics["rewards/gap_crossing_bonus"] = bonus
        return bonus

    # ---- Termination criteria ----

    @_registry.termination("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = -0.05,
        max_torso_angle: float = 70,
    ) -> bool:
        """Check if rodent has fallen.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum z height threshold.
            max_torso_angle: Maximum angle from vertical in degrees.

        Returns:
            Boolean indicating if fallen.
        """
        del info

        torso_body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        torso_z = torso_body.xpos[2]

        below_ground = torso_z < min_torso_z

        # xmat is 3x3 rotation matrix, [-1, -1] is element (2,2) = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_ground, too_tilted)

    @_registry.termination("stale_location")
    def _stale_location_termination(
        self,
        data: mjx.Data,
        info,
        patience: int = 100,
        min_displacement: float = 0.03,
    ) -> bool:
        """Terminate if agent hasn't displaced forward enough over patience window.

        This catches cases where the agent gets stuck (e.g., feet wedged in a
        gap) and jitters in place without making net forward progress, which
        would not trigger the height-based fallen termination.

        The actual displacement check and counter management happen in step().
        This function just reads the counter - if it reached patience without
        being reset (meaning insufficient displacement), we terminate.

        Args:
            data: Simulation data (unused, state is tracked in info).
            info: State info containing stale_steps counter.
            patience: Steps before checking displacement (counter reset on progress).
            min_displacement: Minimum forward displacement required over patience
                window to avoid termination (in meters).

        Returns:
            Boolean indicating if agent has been stale too long.
        """
        del data
        return info["stale_steps"] >= patience

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Check for NaN values in simulation data.

        Args:
            data: Simulation data.
            info: State info (unused).

        Returns:
            Boolean indicating if NaN detected.
        """
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    # ---- Utility methods ----

    def null_action(self) -> jp.ndarray:
        """Return zero action."""
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0])

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)
