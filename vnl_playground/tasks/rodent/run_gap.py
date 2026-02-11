"""RunGap corridor task for virtual rodent.

The rodent must run forward (+x direction) across platforms separated by gaps.
Platforms are procedurally generated box geoms added to a corridor arena that
has side walls but no floor.

Reward is based on forward velocity (tolerance function), with optional penalties
for lateral movement. An alive bonus is provided each step.

Termination occurs if:
- Torso becomes too tilted or falls below the platforms (fallen)
- NaN detected in simulation data
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
from mujoco_playground._src import reward as reward_fns

from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts
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
        naconmax=90 * 1024,
        njmax=1200,
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
        target_speed=0.3,
        episode_length=2000,
        action_repeat=1,
        spawn_x=0.5,
        reward_terms={
            "forward_velocity": {"weight": 1.0},
            "lateral_velocity": {"weight": -0.1},
            "alive": {"weight": 0.1},
        },
        termination_criteria={
            "fallen": {"min_torso_z": -0.05, "max_torso_angle": 70},
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
        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def _build_corridor(self) -> None:
        """Procedurally build corridor platforms with gaps.

        Creates a starting platform followed by alternating gaps and platforms.
        Platforms are box geoms with collision enabled. Uses a fixed random seed
        for deterministic layout.
        """
        rng = np.random.RandomState(42)

        half_width = self._config.corridor_width / 2.0
        half_thickness = _WALL_THICKNESS / 2.0
        platform_length_range = self._config.platform_length_range
        gap_length_range = self._config.gap_length_range

        self._platform_positions = []  # (x_start, x_end) for each platform

        # Starting platform
        start_length = 2.0
        x_cursor = 0.0
        x_start = x_cursor - start_length / 2.0
        x_end = x_cursor + start_length / 2.0

        body = self._spec.worldbody.add_body(
            name="platform_start",
            pos=[x_cursor, 0.0, -half_thickness],
        )
        body.add_geom(
            name="platform_start_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[start_length / 2.0, half_width, half_thickness],
            material="platform_mat",
            contype=1,
            conaffinity=1,
        )
        self._platform_positions.append((x_start, x_end))
        x_cursor = x_end

        # Alternating gaps and platforms
        for i in range(self._config.n_platforms):
            # Gap
            gap_length = rng.uniform(*gap_length_range)
            x_cursor += gap_length

            # Platform
            plat_length = rng.uniform(*platform_length_range)
            plat_center_x = x_cursor + plat_length / 2.0

            body = self._spec.worldbody.add_body(
                name=f"platform_{i}",
                pos=[plat_center_x, 0.0, -half_thickness],
            )
            body.add_geom(
                name=f"platform_{i}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[plat_length / 2.0, half_width, half_thickness],
                material="platform_mat",
                contype=1,
                conaffinity=1,
            )
            self._platform_positions.append(
                (x_cursor, x_cursor + plat_length)
            )
            x_cursor += plat_length

        self._corridor_end_x = x_cursor

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment state.

        Args:
            rng: Random number generator state.

        Returns:
            mjx_env.State: The initial environment state after reset.
        """
        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

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

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> collections.OrderedDict:
        """Get the current observation from the simulation data.

        Args:
            data: The simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with state and privileged_state keys.
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

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(
            state=obs,
            privileged_state=obs,
        )

    # ---- Reward functions ----

    @_registry.reward("forward_velocity")
    def _forward_velocity_reward(self, data, info, metrics, weight) -> float:
        """Reward for maintaining target forward velocity in +x direction.

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

        target_speed = self._config.target_speed

        reward_value = reward_fns.tolerance(
            forward_vel,
            bounds=(target_speed, target_speed),
            margin=target_speed,
            sigmoid="linear",
            value_at_margin=0.0,
        )

        weighted_reward = reward_value * weight
        metrics["rewards/forward_velocity"] = weighted_reward

        return weighted_reward

    @_registry.reward("lateral_velocity")
    def _lateral_velocity_cost(self, data, info, metrics, weight) -> float:
        """Cost for lateral (y-direction) velocity to encourage straight-line motion.

        Args:
            data: Simulation data.
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Cost weight multiplier (negative value = penalty).

        Returns:
            Weighted lateral velocity cost.
        """
        del info
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        lateral_vel = body.subtree_linvel[1]  # y-direction velocity
        cost = weight * jp.square(lateral_vel)
        metrics["rewards/lateral_velocity"] = cost
        return cost

    @_registry.reward("alive")
    def _alive_reward(self, data, info, metrics, weight) -> float:
        """Constant alive bonus per step.

        Args:
            data: Simulation data (unused).
            info: State info (unused).
            metrics: Metrics dict for logging.
            weight: Reward weight (constant bonus per step).

        Returns:
            Alive bonus.
        """
        del data, info
        metrics["rewards/alive"] = weight
        return weight

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
