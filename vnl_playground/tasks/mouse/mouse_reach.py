"""Mouse forelimb reaching task following rodent task patterns."""

import collections
from typing import Any, Dict, Mapping, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from vnl_playground.tasks.mouse import consts
from vnl_playground.tasks.mouse.base import (
    MouseBaseEnv,
    default_config as base_default_config,
)


def default_config() -> config_dict.ConfigDict:
    """Default config for mouse reaching task.

    Returns:
        config_dict.ConfigDict: Configuration with target size/margin, control
            cost weight, target sampling mode, and volume bounds.
    """
    cfg = base_default_config()
    cfg.target_size = 0.001  # reaching radius tolerance
    cfg.target_margin = 0.003  # margin for reward shaping
    cfg.ctrl_cost_weight = 0.001  # penalty on action magnitude
    # Target sampling: "fixed_list" or "random_volume"
    cfg.target_mode = "random_volume"
    # Volume bounds for random sampling (min and max corners)
    # Derived from fixed target positions: x in [-0.003, 0.007], y ~ 0.010, z in [-0.011, -0.001]
    cfg.target_volume_min = (-0.003, 0.010, -0.011)
    cfg.target_volume_max = (0.007, 0.010, -0.001)
    return cfg


class MouseReach(MouseBaseEnv):
    """Mouse reaching env: reward for moving wrist marker to target position."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the mouse reaching environment.

        Adds a fixed-base mouse arm, a mocap target sphere, compiles
        the model, and caches body/geom IDs.

        Args:
            config: Configuration dictionary with reaching task parameters.
            config_overrides: Optional overrides for config fields.
        """
        super().__init__(config, config_overrides)

        # Add mouse model (no freejoint - fixed base arm)
        # Spawn at origin to match target positions from original XML
        self.add_mouse(freejoint=False, pos=(0.0, 0.0, 0.0))

        # Add mocap body for target (allows dynamic positioning via data.mocap_pos)
        target_body = self._spec.worldbody.add_body(
            name="target",
            mocap=True,
            pos=[0.002, 0.010, -0.006],  # default position
        )
        target_body.add_geom(
            name="target_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.001, 0, 0],
            rgba=[0, 1, 0, 0.5],
            contype=0,
            conaffinity=0,
        )

        # Compile model
        self.compile()

        # Cache body/geom IDs after compilation (names have "-mouse" suffix from add_mouse)
        self._wrist_body_id = self._mj_model.body("wrist_body-mouse").id
        self._wrist_marker_geom_id = self._mj_model.geom("wrist_marker-mouse").id
        self._target_mocap_id = self._mj_model.body("target").mocapid[0]

    @staticmethod
    def get_target_positions() -> jp.ndarray:
        """Return a fixed set of reachable target positions.

        Returns:
            jp.ndarray: Array of shape (8, 3) with target (x, y, z) positions.
        """
        return jp.array(
            [
                [0.007, 0.010, -0.006],
                [0.0055355, 0.010, -0.0024645],
                [0.002, 0.010, -0.001],
                [-0.0015355, 0.010, -0.0024645],
                [-0.003, 0.010, -0.006],
                [-0.0015355, 0.010, -0.0095355],
                [0.002, 0.010, -0.011],
                [0.0055355, 0.010, -0.0095355],
            ],
            dtype=jp.float32,
        )

    def _sample_target_from_list(self, rng: jax.Array) -> jp.ndarray:
        """Sample a target from the fixed list of positions.

        Args:
            rng: JAX random key.

        Returns:
            jp.ndarray: Sampled target position of shape (3,).
        """
        target_positions = self.get_target_positions()
        idx = jax.random.randint(rng, (), 0, target_positions.shape[0])
        return target_positions[idx]

    def _sample_target_from_volume(self, rng: jax.Array) -> jp.ndarray:
        """Sample a target uniformly from the configured cubic volume.

        Args:
            rng: JAX random key.

        Returns:
            jp.ndarray: Sampled target position of shape (3,).
        """
        vol_min = jp.array(self._config.target_volume_min, dtype=jp.float32)
        vol_max = jp.array(self._config.target_volume_max, dtype=jp.float32)
        return jax.random.uniform(rng, shape=(3,), minval=vol_min, maxval=vol_max)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment and sample a target for this episode.

        Args:
            rng: JAX random key for target sampling.

        Returns:
            mjx_env.State: Initial environment state with sampled target.
        """
        # Sample a target position based on mode
        rng, key = jax.random.split(rng)
        if self._config.target_mode == "random_volume":
            target_position = self._sample_target_from_volume(key)
        else:  # "fixed_list"
            target_position = self._sample_target_from_list(key)

        # Initialize physics data (pass impl for warp/jax compatibility)
        data = mjx.make_data(self.mj_model, impl=self._config.mujoco_impl)

        # Set mocap target position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._target_mocap_id].set(target_position)
        )

        # Run forward kinematics to update xpos from mocap_pos
        data = mjx.forward(self.mjx_model, data)

        # Build observation
        obs = self._get_obs(data, target_position)

        reward_val, done = jp.zeros(2)
        metrics = {}

        info = {
            "target_position": target_position,
            "prev_action": jp.zeros(self.action_size),
        }

        return mjx_env.State(data, obs, reward_val, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take one control step (with physics substeps).

        Args:
            state: Current environment state.
            action: Action array of shape (action_size,).

        Returns:
            mjx_env.State: Updated environment state after stepping.
        """
        target_position = state.info["target_position"]

        # Ensure mocap position is set before physics step
        data = state.data.replace(
            mocap_pos=state.data.mocap_pos.at[self._target_mocap_id].set(
                target_position
            )
        )

        # Step physics (n_steps = ctrl_dt / sim_dt for proper substep integration)
        # mjx_env.step calls forward internally, so xpos will be updated
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, data, action, n_steps)

        # Get observation
        obs = self._get_obs(data, target_position)

        # Compute reward
        rew = self._get_reward(data, target_position, action)

        # Check termination
        done = self._get_termination(data)

        # Update info (direct assignment, no .copy() - JAX friendly)
        info = state.info
        info["prev_action"] = action

        return state.replace(data=data, obs=obs, reward=rew, done=done, info=info)

    def _get_obs(
        self, data: mjx.Data, target_position: jp.ndarray
    ) -> Mapping[str, jp.ndarray]:
        """Build observation dictionary.

        Args:
            data: MJX simulation data.
            target_position: Target position of shape (3,).

        Returns:
            Mapping[str, jp.ndarray]: OrderedDict with 'task_obs' (direction
                to target) and 'proprioception' (qpos, qvel, wrist position).
        """
        wrist_pos = data.xpos[self._wrist_body_id]
        to_target = target_position - wrist_pos

        return collections.OrderedDict(
            task_obs=to_target,
            proprioception=jp.concatenate(
                [
                    data.qpos,
                    data.qvel,
                    wrist_pos,
                ]
            ),
        )

    def _get_reward(
        self, data: mjx.Data, target_position: jp.ndarray, action: jax.Array
    ) -> jp.ndarray:
        """Distance-based tolerance reward from wrist marker to target.

        Args:
            data: MJX simulation data.
            target_position: Target position of shape (3,).
            action: Applied action array.

        Returns:
            jp.ndarray: Scalar reward (tolerance reward minus control cost).
        """
        wrist_marker_pos = data.geom_xpos[self._wrist_marker_geom_id]
        dist = jp.linalg.norm(wrist_marker_pos - target_position)
        distance_reward = reward.tolerance(
            dist,
            bounds=(0, self._config.target_size),
            margin=self._config.target_margin,
            sigmoid="hyperbolic",
        )

        # Control cost: penalize large actions
        ctrl_cost = self._config.ctrl_cost_weight * jp.sum(jp.square(action))

        return jp.asarray(distance_reward - ctrl_cost, dtype=jp.float32)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """Check for early termination conditions.

        Args:
            data: MJX simulation data.

        Returns:
            jax.Array: Scalar float, 0.0 (no early termination by default).
        """
        return jp.zeros((), dtype=jp.float32)

    @property
    def observation_size(self) -> int:
        """Compute observation size from model dimensions.

        Observation = [to_target(3), qpos(nq), qvel(nv), wrist_pos(3)]
        """
        # task_obs: to_target (3)
        # proprio_obs: qpos (nq) + qvel (nv) + wrist_pos (3)
        nq = self._mj_model.nq
        nv = self._mj_model.nv
        return 3 + nq + nv + 3

    @property
    def non_flattened_observation_size(self) -> Dict[str, int]:
        """Get observation sizes by component.

        Returns:
            Dict[str, int]: Mapping of observation component names to their sizes.
        """
        nq = self._mj_model.nq
        nv = self._mj_model.nv
        return {
            "task_obs": 3,  # to_target
            "proprioception": nq + nv + 3,  # qpos + qvel + wrist_pos
        }
