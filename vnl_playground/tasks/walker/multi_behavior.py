"""Multi-behavior conditional walker environment.

Step 1 environment: the policy takes observations + one-hot behavior mode
and learns to exhibit 4 distinct behaviors: stand, walk_slow, run, knee_down.

The behavior mode can change within an episode to train smooth transitions.

Observation: [orientations(14), height(1), velocity(9), joint_angles(6),
              prev_action(6), behavior_mode(4)] = 40 dims

Action: 6 motor torques in [-1, 1]
"""

from typing import Any, Dict, Mapping, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_utils

from vnl_playground.tasks.walker import consts
from vnl_playground.tasks.walker.base import WalkerEnv


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        sim_dt=0.0025,
        ctrl_dt=0.025,
        episode_length=1000,
        mujoco_impl="jax",
        nconmax=50_000,
        njmax=100,
        # Behavior mode scheduling
        mode_switch_prob=0.005,  # Probability of mode switch per step
        fixed_mode=None,  # If set, fix to this mode index (no switching)
        # Reward weights
        stand_height_weight=3.0,
        upright_weight=1.0,
        move_weight=5.0,
        knee_bend_weight=2.0,
        knee_height_weight=2.0,
        control_cost_weight=0.01,
    )


class MultiBehaviorWalker(WalkerEnv):
    """PlanarWalker with 4 conditional behavior modes.

    Behavior modes (one-hot encoded in observation):
      0 = stand:      Stay upright at full height, no horizontal movement
      1 = walk_slow:  Walk forward at 0.5 m/s while staying upright
      2 = run:        Run forward at 8 m/s while staying upright
      3 = knee_down:  Kneel down (bend both knees) at lower torso height
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_qpos, rng_joints, rng_mode = jax.random.split(rng, 4)

        # Randomize initial joint positions (same as PlanarWalker)
        qpos = jp.zeros(self.mjx_model.nq)
        # Randomize root y-rotation
        qpos = qpos.at[2].set(
            jax.random.uniform(rng_qpos, (), minval=-jp.pi, maxval=jp.pi)
        )
        # Randomize joint angles within limits
        qpos = qpos.at[consts.N_ROOT_QPOS:].set(
            jax.random.uniform(
                rng_joints,
                (consts.N_JOINTS,),
                minval=self._joint_lowers,
                maxval=self._joint_uppers,
            )
        )

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Select initial behavior mode
        if self._config.fixed_mode is not None:
            mode_idx = jp.array(self._config.fixed_mode, dtype=jp.int32)
        else:
            mode_idx = jax.random.randint(rng_mode, (), 0, consts.N_BEHAVIOR_MODES)
        behavior_mode = jax.nn.one_hot(mode_idx, consts.N_BEHAVIOR_MODES)

        info = {
            "rng": rng,
            "behavior_mode": behavior_mode,
            "mode_idx": mode_idx,
            "prev_action": jp.zeros(self.action_size),
            "action": jp.zeros(self.action_size),
        }

        metrics = {
            "reward/standing": jp.zeros(()),
            "reward/upright": jp.zeros(()),
            "reward/stand": jp.zeros(()),
            "reward/move": jp.zeros(()),
            "reward/knee_bend": jp.zeros(()),
            "reward/knee_height": jp.zeros(()),
            "reward/control_cost": jp.zeros(()),
            "reward/total": jp.zeros(()),
            "mode_idx": jp.float32(mode_idx),
        }

        obs = self._get_obs(data, info)
        reward_val = jp.zeros(())
        done = jp.zeros(())
        return mjx_env.State(data, obs, reward_val, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

        # Possibly switch behavior mode
        info = state.info
        rng, rng_switch, rng_mode = jax.random.split(info["rng"], 3)
        info["rng"] = rng
        info["prev_action"] = state.info["action"]
        info["action"] = action

        if self._config.fixed_mode is None:
            should_switch = (
                jax.random.uniform(rng_switch) < self._config.mode_switch_prob
            )
            new_mode_idx = jax.random.randint(rng_mode, (), 0, consts.N_BEHAVIOR_MODES)
            mode_idx = jp.where(should_switch, new_mode_idx, info["mode_idx"])
            info["mode_idx"] = mode_idx
            info["behavior_mode"] = jax.nn.one_hot(mode_idx, consts.N_BEHAVIOR_MODES)

        reward_val = self._compute_reward(data, action, info, state.metrics)
        obs = self._get_obs(data, info)

        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        state.metrics["mode_idx"] = jp.float32(info["mode_idx"])
        state.metrics["reward/total"] = reward_val

        return mjx_env.State(data, obs, reward_val, done, state.metrics, info)

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> jax.Array:
        """Observations: proprioception + behavior mode one-hot.

        Layout: [orientations(14), height(1), velocity(9),
                 joint_angles(6), prev_action(6), behavior_mode(4)] = 40
        """
        orientations = self._get_orientations(data)
        height = self._get_body_height(data).reshape(1)
        velocity = data.qvel
        joint_angles = self._get_joint_angles(data)
        prev_action = info.get("prev_action", jp.zeros(self.action_size))
        behavior_mode = info["behavior_mode"]

        return jp.concatenate([
            orientations,
            height,
            velocity,
            joint_angles,
            prev_action,
            behavior_mode,
        ])

    def _compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Mapping[str, Any],
        metrics: dict,
    ) -> jax.Array:
        """Compute reward based on current behavior mode."""
        mode_idx = info["mode_idx"]

        # Shared components
        torso_height = data.xpos[self._torso_id, 2]
        torso_upright = data.xmat[self._torso_id, 2, 2]
        horizontal_vel = self._get_horizontal_velocity(data)

        standing = reward_utils.tolerance(
            torso_height,
            bounds=(consts.STAND_HEIGHT, float("inf")),
            margin=consts.STAND_HEIGHT / 2,
        )
        upright = (1 + torso_upright) / 2
        metrics["reward/standing"] = standing
        metrics["reward/upright"] = upright

        control_cost = -self._config.control_cost_weight * jp.sum(jp.square(action))
        metrics["reward/control_cost"] = control_cost

        # Stand reward: be tall and upright
        stand_reward = (
            (self._config.stand_height_weight * standing
             + self._config.upright_weight * upright)
            / (self._config.stand_height_weight + self._config.upright_weight)
        )
        metrics["reward/stand"] = stand_reward

        # Move reward component (for walk_slow and run)
        def _move_reward(target_speed):
            move_r = reward_utils.tolerance(
                horizontal_vel,
                bounds=(target_speed, float("inf")),
                margin=target_speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
            return stand_reward * (
                self._config.move_weight * move_r + 1
            ) / (self._config.move_weight + 1)

        # Knee-down reward components
        right_knee = data.qpos[consts.N_ROOT_QPOS + 1]  # right_knee joint
        left_knee = data.qpos[consts.N_ROOT_QPOS + 4]   # left_knee joint
        min_knee, max_knee = consts.KNEE_DOWN_ANGLE_RANGE
        right_knee_bent = reward_utils.tolerance(
            right_knee, bounds=(min_knee, max_knee), margin=0.3,
        )
        left_knee_bent = reward_utils.tolerance(
            left_knee, bounds=(min_knee, max_knee), margin=0.3,
        )
        knee_bend_r = (right_knee_bent + left_knee_bent) / 2

        min_h, max_h = consts.KNEE_DOWN_HEIGHT_RANGE
        knee_height_r = reward_utils.tolerance(
            torso_height, bounds=(min_h, max_h), margin=0.2,
        )
        knee_no_move = reward_utils.tolerance(
            jp.abs(horizontal_vel), bounds=(0.0, 0.1), margin=0.2,
        )
        metrics["reward/knee_bend"] = knee_bend_r
        metrics["reward/knee_height"] = knee_height_r

        # Mode-specific rewards
        def stand_mode(_):
            return stand_reward + control_cost

        def walk_slow_mode(_):
            move_r = _move_reward(consts.WALK_SLOW_SPEED)
            return move_r + control_cost

        def run_mode(_):
            move_r = _move_reward(consts.RUN_SPEED)
            return move_r + control_cost

        def knee_down_mode(_):
            knee_r = (
                (self._config.knee_bend_weight * knee_bend_r
                 + self._config.knee_height_weight * knee_height_r
                 + self._config.upright_weight * upright
                 + knee_no_move)
                / (self._config.knee_bend_weight + self._config.knee_height_weight
                   + self._config.upright_weight + 1.0)
            )
            return knee_r + control_cost

        reward = jax.lax.switch(
            mode_idx,
            [stand_mode, walk_slow_mode, run_mode, knee_down_mode],
            None,
        )
        return reward

    @property
    def observation_size(self) -> int:
        # orientations(14) + height(1) + velocity(9) + joints(6)
        # + prev_action(6) + mode(4) = 40
        return 14 + 1 + 9 + 6 + 6 + consts.N_BEHAVIOR_MODES
