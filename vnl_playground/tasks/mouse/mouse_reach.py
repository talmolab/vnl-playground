"""Mouse forelimb reaching task following rodent task patterns."""

import collections
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_fns

from vnl_playground.tasks import math_utils
from vnl_playground.tasks.mouse.base import (
    MouseBaseEnv,
)
from vnl_playground.tasks.mouse.base import (
    default_config as base_default_config,
)
from vnl_playground.tasks.reward_registry import RewardRegistry


def default_config() -> config_dict.ConfigDict:
    """Default config for mouse reaching task.

    Returns:
        config_dict.ConfigDict: Configuration with target size/margin, control
            cost weight, target sampling mode, and volume bounds.
    """
    cfg = base_default_config()
    cfg.target_size = 0.001  # reaching radius tolerance
    cfg.target_margin = 0.003  # margin for reward shaping
    # Target sampling: "fixed_list" or "random_volume"
    cfg.target_mode = "random_volume"
    # Volume bounds for random sampling (min and max corners)
    # Derived from fixed target positions: x in [-0.003, 0.007], y ~ 0.010, z in [-0.011, -0.001]
    cfg.target_volume_min = (-0.003, 0.010, -0.011)
    cfg.target_volume_max = (0.007, 0.010, -0.001)
    cfg.reward_terms = {
        "distance": {"weight": 1.0},
        "control_cost": {"weight": 0.001},
    }
    cfg.termination_criteria = {
        "nan_termination": {},
    }
    return cfg


_registry = RewardRegistry()


class MouseReach(MouseBaseEnv):
    """Mouse reaching env: reward for moving wrist marker to target position."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict[str, str | int | list[Any]] | None = None,
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

        info = {
            "target_position": target_position,
            "prev_action": jp.zeros(self.action_size),
            "action": jp.zeros(self.action_size),
        }

        obs = self._get_obs(data, info)
        metrics = {}
        reward_val = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(
            data, obs, reward_val, jp.astype(done, float), metrics, info
        )

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

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        rew = self._get_reward(data, info, state.metrics)
        rew = jp.nan_to_num(rew)

        return state.replace(
            data=data, obs=obs, reward=rew, done=done.astype(float), info=info
        )

    def _get_obs(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Build observation dictionary.

        Args:
            data: MJX simulation data.
            info: Episode info dict containing 'target_position'.

        Returns:
            Mapping[str, jp.ndarray]: OrderedDict with state wrapper containing
                'task_obs' (direction to target) and 'proprioception'.
        """
        wrist_pos = data.xpos[self._wrist_body_id]
        to_target = info["target_position"] - wrist_pos

        obs = collections.OrderedDict(
            task_obs=to_target,
            proprioception=jp.concatenate(
                [
                    data.qpos,
                    data.qvel,
                    wrist_pos,
                ]
            ),
        )
        return collections.OrderedDict(state=obs)

    @_registry.reward("distance")
    def _distance_reward(self, data, info, metrics, weight) -> float:
        wrist_marker_pos = data.geom_xpos[self._wrist_marker_geom_id]
        dist = jp.linalg.norm(wrist_marker_pos - info["target_position"])
        reward_value = reward_fns.tolerance(
            dist,
            bounds=(0, self._config.target_size),
            margin=self._config.target_margin,
            sigmoid="hyperbolic",
        )
        weighted = weight * reward_value
        metrics["rewards/distance"] = weighted
        return weighted

    @_registry.reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        cost = -weight * math_utils.squared_l2_norm(info["action"])
        metrics["rewards/control_cost"] = cost
        return cost

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        return jp.any(jp.isnan(data.qpos))
