"""Mouse forelimb reaching task following rodent task patterns."""

from typing import Any, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jp
from jax import flatten_util
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from vnl_playground.tasks.mouse import consts
from vnl_playground.tasks.mouse.base import MouseBaseEnv, default_config as base_default_config


def default_config() -> config_dict.ConfigDict:
    """Default config for mouse reaching task."""
    cfg = base_default_config()
    cfg.target_size = 0.003  # reaching radius tolerance
    cfg.target_margin = 0.006  # margin for reward shaping
    return cfg


class MouseReach(MouseBaseEnv):
    """Mouse reaching env: reward for moving wrist marker to target position."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
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
            size=[0.003, 0, 0],
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
        """Return a fixed set of reachable target positions."""
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

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment and sample a target for this episode."""
        # Sample a target position
        target_positions = self.get_target_positions()
        rng, key = jax.random.split(rng)
        idx = jax.random.randint(key, (), 0, target_positions.shape[0])
        target_position = target_positions[idx]

        # Initialize physics data
        data = mjx.make_data(self.mjx_model)

        # Set mocap target position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._target_mocap_id].set(target_position)
        )

        # Run forward kinematics to update xpos from mocap_pos
        data = mjx.forward(self.mjx_model, data)

        # Build observation
        task_obs, proprio_obs = self._get_obs(data, target_position)
        obs = jp.concatenate([task_obs, proprio_obs])

        reward_val, done = jp.zeros(2)
        metrics = {}

        info = {
            "target_position": target_position,
            "prev_action": jp.zeros(self.action_size),
            "task_obs_size": task_obs.shape[0],
            "proprio_obs_size": proprio_obs.shape[0],
        }

        return mjx_env.State(data, obs, reward_val, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take one physics step."""
        target_position = state.info["target_position"]

        # Ensure mocap position is set before physics step
        data = state.data.replace(
            mocap_pos=state.data.mocap_pos.at[self._target_mocap_id].set(target_position)
        )

        # Step physics (n_steps = ctrl_dt / sim_dt for proper substep integration)
        # mjx_env.step calls forward internally, so xpos will be updated
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, data, action, n_steps)

        # Get observation
        task_obs, proprio_obs = self._get_obs(data, target_position)
        obs = jp.concatenate([task_obs, proprio_obs])

        # Compute reward
        rew = self._get_reward(data, target_position)

        # Check termination
        done = self._get_termination(data)

        # Update info (direct assignment, no .copy() - JAX friendly)
        info = state.info
        info["prev_action"] = action

        return state.replace(data=data, obs=obs, reward=rew, done=done, info=info)

    def _get_obs(
        self, data: mjx.Data, target_position: jp.ndarray
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        """Build observation vector.

        Returns:
            task_obs: task-specific observations (target direction)
            proprio_obs: proprioceptive observations (qpos, qvel, wrist pos)
        """
        wrist_pos = data.xpos[self._wrist_body_id]
        to_target = target_position - wrist_pos

        # Task obs: direction to target
        task_obs = to_target

        # Proprioceptive obs: joint angles, velocities, wrist position
        proprio_obs = jp.concatenate([
            data.qpos,
            data.qvel,
            wrist_pos,
        ])

        return task_obs, proprio_obs

    def _get_reward(self, data: mjx.Data, target_position: jp.ndarray) -> jp.ndarray:
        """Distance-based tolerance reward from wrist marker to target."""
        wrist_marker_pos = data.geom_xpos[self._wrist_marker_geom_id]
        dist = jp.linalg.norm(wrist_marker_pos - target_position)
        return jp.asarray(
            reward.tolerance(
                dist,
                bounds=(0, self._config.target_size),
                margin=self._config.target_margin,
                sigmoid="hyperbolic",
            ),
            dtype=jp.float32,
        )

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """No early termination by default."""
        return jp.zeros((), dtype=jp.float32)

    @property
    def observation_size(self) -> int:
        """Compute observation size using jax.eval_shape (no actual execution)."""
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> Dict[str, int]:
        """Get observation sizes without executing reset."""
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)
