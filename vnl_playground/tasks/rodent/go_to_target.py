"""Go-to-target navigation task for rodent.

A simple task where the rodent navigates to a target position in the world.
Supports single targets, multi-waypoint sequences, and random targets.
No trial phases — the reward is always active.

Usage::

    env = GoToTarget(config=cfg)
    state = env.reset(rng)
    state = env.step(state, action)
"""

import collections
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.task_registry import TaskRegistry

_registry = TaskRegistry()


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the GoToTarget environment."""
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
        spawn_x=0.0,
        # Episode limits
        episode_length=300,
        action_repeat=1,
        # --- Target / waypoint system ---
        target_position_mode="fixed",  # "fixed", "random", "waypoints"
        fixed_target_position=(0.5, 0.0, 0.0),
        target_waypoints=(),  # tuple of (x,y,z) tuples for "waypoints" mode
        max_waypoints=4,
        target_reach_threshold=0.05,  # 5cm
        auto_advance_waypoint=True,
        loop_waypoints=False,
        # Random mode bounds
        target_random_x_range=(0.1, 0.8),
        target_random_y_range=(-0.15, 0.15),
        target_random_z=0.0,
        # Aesthetic
        aesthetic="default",  # "default" or "outdoor_natural"
        # Reward terms
        reward_terms={
            "go_to_target": {"weight": 5.0},
        },
        # Termination criteria
        termination_criteria={
            "fallen": {"min_torso_z": -0.15},
            "reached_target": {},
            "timeout": {"max_steps": 300},
            "nan_termination": {},
        },
    )


class GoToTarget(rodent_base.RodentEnv):
    """Go-to-target navigation task.

    The rodent spawns on a flat ground plane and must navigate to a target
    position. Supports single targets, multi-waypoint sequences (round-trip),
    random targets per episode, and external target override during rollouts.

    No trial phases — the go_to_target reward is always active.
    """

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._rng = rng

        # Default platform material
        self._platform_material = "platform_mat"

        # Apply aesthetic textures before building arena
        if self._config.get("aesthetic", "default") == "outdoor_natural":
            self._apply_outdoor_natural_aesthetic()

        # Add a flat ground plane to the arena
        self._build_arena()

        # Place rodent at spawn position facing forward (+x)
        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[self._config.spawn_x, 0.0, 0.0],
            quat=(1, 0, 0, 0),
        )
        self.compile()

        self._max_waypoints = self._config.get("max_waypoints", 4)

    def _apply_outdoor_natural_aesthetic(self) -> None:
        """Apply outdoor natural aesthetic: grass platforms, blue sky, better lighting."""
        import pathlib

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
        """Add a flat ground plane to the arena."""
        self._spec.worldbody.add_geom(
            name="ground_plane",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[5, 5, 0.1],
            pos=[0, 0, 0],
            contype=1,
            conaffinity=1,
        )

    # ------------------------------------------------------------------
    # Core environment interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset with a target position based on config mode."""
        rng, target_rng = jax.random.split(rng)

        # --- Build target waypoints based on mode ---
        waypoints = jp.zeros((self._max_waypoints, 3))

        if self._config.target_position_mode == "fixed":
            waypoints = waypoints.at[0].set(
                jp.array(self._config.fixed_target_position)
            )
            num_waypoints = jp.array(1, dtype=jp.int32)

        elif self._config.target_position_mode == "random":
            x_lo, x_hi = self._config.target_random_x_range
            y_lo, y_hi = self._config.target_random_y_range
            target_rng_x, target_rng_y = jax.random.split(target_rng)
            rand_x = jax.random.uniform(target_rng_x, minval=x_lo, maxval=x_hi)
            rand_y = jax.random.uniform(target_rng_y, minval=y_lo, maxval=y_hi)
            rand_z = self._config.target_random_z
            waypoints = waypoints.at[0].set(jp.array([rand_x, rand_y, rand_z]))
            num_waypoints = jp.array(1, dtype=jp.int32)

        elif self._config.target_position_mode == "waypoints":
            cfg_wps = self._config.target_waypoints
            n = min(len(cfg_wps), self._max_waypoints)
            for i in range(n):
                waypoints = waypoints.at[i].set(jp.array(cfg_wps[i]))
            num_waypoints = jp.array(n, dtype=jp.int32)

        else:
            # Fallback: zero target (for external override)
            num_waypoints = jp.array(1, dtype=jp.int32)

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "step_count": jp.array(0, dtype=jp.int32),
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
        data = mjx.forward(self.mjx_model, data)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step physics and advance waypoints."""
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action
        info["step_count"] = info["step_count"] + 1

        # Compute obs, done, reward using CURRENT target_position
        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        # --- Waypoint advancement (AFTER reward) ---
        if self._config.get("auto_advance_waypoint", True):
            torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
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
        """Observations: proprioception + egocentric target vector.

        task_obs = [prev_action, kinematic_sensors, touch, origin, ego_target]
        No phase indicator — this task has no phases.
        """
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
            target_position=info.get("target_position", jp.zeros(3)),
        )

        return collections.OrderedDict(
            state=obs,
            privileged_state=privileged_obs,
        )

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    @_registry.reward("go_to_target")
    def _go_to_target_reward(self, data, info, metrics, weight):
        """Dense reward for moving toward the active target position.

        Always active (no phase gating). Exponential proximity kernel.
        """
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        target_pos = info.get("target_position", jp.zeros(3))
        dist = jp.linalg.norm(torso.xpos - target_pos)

        length_scale = 0.3
        proximity = jp.exp(-dist / length_scale)

        reward_val = weight * proximity
        metrics["rewards/go_to_target"] = reward_val
        metrics["rewards/target_distance"] = dist
        return reward_val

    # ------------------------------------------------------------------
    # Termination criteria
    # ------------------------------------------------------------------

    @_registry.termination("reached_target")
    def _reached_target_termination(self, data, info):
        """Terminate when all waypoints are completed."""
        return info.get("target_reached", jp.array(False))

    @_registry.termination("fallen")
    def _fallen_termination(self, data, info, min_torso_z=-0.15):
        """Terminate if torso drops below minimum height."""
        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        return torso.xpos[2] < min_torso_z

    @_registry.termination("timeout")
    def _timeout_termination(self, data, info, max_steps=300):
        """Terminate if step count exceeds maximum."""
        return info["step_count"] >= max_steps

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info):
        """Terminate on NaN values in simulation data."""
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        return jp.sum(jp.isnan(flattened_vals)) > 0

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0]
        )

    @property
    def observation_size(self) -> int:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        return jax.tree_util.tree_map(
            lambda x: jp.prod(jp.array(x.shape)), abstract_state.obs
        )
