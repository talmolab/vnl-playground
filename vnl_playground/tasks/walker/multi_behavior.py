"""Multi-behavior conditional walker environment.

Step 1 environment: the policy takes observations + soft behavior mode vector
and learns to exhibit 8 distinct behaviors with smooth transitions between them.

Behavior modes:
  0 = stand:          Stay upright at full height, no horizontal movement
  1 = walk_slow:      Walk forward at 0.5 m/s
  2 = run:            Run forward at 8 m/s
  3 = knee_down:      Kneel down (bend both knees) at lower torso height
  4 = walk_backward:  Walk backward at 0.5 m/s
  5 = hop:            Jump in place (periodic vertical motion)
  6 = walk_fast:      Walk forward at 3 m/s
  7 = tiptoe:         Stand tall on extended ankles

Transitions use linear blending over a configurable window so the mode
vector interpolates smoothly (e.g., [1,0,...] -> [0,0,1,...] over 40 steps).
The reward is a weighted sum of both source and target mode rewards during
the transition.

Observation: [orientations(14), height(1), velocity(9), joint_angles(6),
              prev_action(6), behavior_mode(8)] = 44 dims

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
        transition_steps=consts.DEFAULT_TRANSITION_STEPS,  # Steps to blend
        # Reward weights
        stand_height_weight=3.0,
        upright_weight=1.0,
        move_weight=5.0,
        knee_bend_weight=2.0,
        knee_height_weight=2.0,
        control_cost_weight=0.01,
    )


class MultiBehaviorWalker(WalkerEnv):
    """PlanarWalker with 8 conditional behavior modes and smooth transitions.

    When a mode switch is triggered, the behavior_mode vector linearly
    interpolates from the current mode to the target over ``transition_steps``
    timesteps.  During the blend, rewards are computed as the weighted sum
    of all mode rewards scaled by the current mode vector, so the policy
    receives a smooth gradient signal throughout the transition.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_qpos, rng_joints, rng_mode = jax.random.split(rng, 4)

        # Randomize initial joint positions
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[2].set(
            jax.random.uniform(rng_qpos, (), minval=-jp.pi, maxval=jp.pi)
        )
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
            # Transition state: source/target mode vectors and progress counter
            "transition_source": behavior_mode,
            "transition_target": behavior_mode,
            "transition_progress": jp.array(self._config.transition_steps, dtype=jp.int32),
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

        info = state.info
        rng, rng_switch, rng_mode = jax.random.split(info["rng"], 3)
        info["rng"] = rng
        info["prev_action"] = state.info["action"]
        info["action"] = action

        if self._config.fixed_mode is None:
            # Check if we should start a new transition
            should_switch = (
                jax.random.uniform(rng_switch) < self._config.mode_switch_prob
            )
            new_mode_idx = jax.random.randint(rng_mode, (), 0, consts.N_BEHAVIOR_MODES)
            new_target = jax.nn.one_hot(new_mode_idx, consts.N_BEHAVIOR_MODES)

            # Only start a new transition if the current one is complete
            transition_done = info["transition_progress"] >= self._config.transition_steps
            start_new = should_switch & transition_done

            # If starting new transition: source = current mode, target = new mode, reset counter
            info["transition_source"] = jp.where(
                start_new, info["behavior_mode"], info["transition_source"]
            )
            info["transition_target"] = jp.where(
                start_new, new_target, info["transition_target"]
            )
            info["transition_progress"] = jp.where(
                start_new, jp.int32(0), info["transition_progress"] + 1
            )
            info["mode_idx"] = jp.where(start_new, new_mode_idx, info["mode_idx"])

            # Compute blended mode vector
            alpha = jp.clip(
                info["transition_progress"] / self._config.transition_steps, 0.0, 1.0
            )
            info["behavior_mode"] = (
                (1.0 - alpha) * info["transition_source"]
                + alpha * info["transition_target"]
            )

        reward_val = self._compute_reward(data, action, info, state.metrics)
        obs = self._get_obs(data, info)

        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        state.metrics["mode_idx"] = jp.float32(info["mode_idx"])
        state.metrics["reward/total"] = reward_val

        return mjx_env.State(data, obs, reward_val, done, state.metrics, info)

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> jax.Array:
        """Observations: proprioception + behavior mode soft vector.

        Layout: [orientations(14), height(1), velocity(9),
                 joint_angles(6), prev_action(6), behavior_mode(8)] = 44
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
        """Compute reward as weighted sum of all mode rewards using mode vector."""
        behavior_mode = info["behavior_mode"]

        # --- Shared components ---
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

        # Stand reward (shared by many modes)
        stand_reward = (
            (self._config.stand_height_weight * standing
             + self._config.upright_weight * upright)
            / (self._config.stand_height_weight + self._config.upright_weight)
        )
        metrics["reward/stand"] = stand_reward

        # --- Move reward helper ---
        def _move_reward(target_speed):
            move_r = reward_utils.tolerance(
                horizontal_vel,
                bounds=(target_speed, float("inf")),
                margin=max(abs(target_speed) / 2, 0.1),
                value_at_margin=0.5,
                sigmoid="linear",
            )
            return stand_reward * (
                self._config.move_weight * move_r + 1
            ) / (self._config.move_weight + 1)

        def _backward_move_reward(target_speed):
            """Reward for backward walking (negative velocity)."""
            move_r = reward_utils.tolerance(
                -horizontal_vel,  # flip sign: reward negative velocity
                bounds=(-target_speed, float("inf")),
                margin=-target_speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
            return stand_reward * (
                self._config.move_weight * move_r + 1
            ) / (self._config.move_weight + 1)

        # --- Knee-down components ---
        right_knee = data.qpos[consts.N_ROOT_QPOS + 1]
        left_knee = data.qpos[consts.N_ROOT_QPOS + 4]
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

        # --- Hop components ---
        vertical_vel = data.qvel[0]  # rootz velocity
        hop_height_r = reward_utils.tolerance(
            torso_height,
            bounds=(consts.HOP_MIN_HEIGHT, float("inf")),
            margin=0.3,
        )
        hop_upward_r = jp.clip(vertical_vel * consts.HOP_VERTICAL_VEL_REWARD_SCALE, 0.0, 1.0)
        hop_no_lateral = reward_utils.tolerance(
            jp.abs(horizontal_vel), bounds=(0.0, 0.2), margin=0.3,
        )

        # --- Tiptoe components ---
        right_ankle = data.qpos[consts.N_ROOT_QPOS + 2]
        left_ankle = data.qpos[consts.N_ROOT_QPOS + 5]
        min_ankle, max_ankle = consts.TIPTOE_ANKLE_RANGE
        right_ankle_ext = reward_utils.tolerance(
            right_ankle, bounds=(min_ankle, max_ankle), margin=0.2,
        )
        left_ankle_ext = reward_utils.tolerance(
            left_ankle, bounds=(min_ankle, max_ankle), margin=0.2,
        )
        ankle_ext_r = (right_ankle_ext + left_ankle_ext) / 2
        tiptoe_height_r = reward_utils.tolerance(
            torso_height,
            bounds=(consts.TIPTOE_HEIGHT, float("inf")),
            margin=0.2,
        )
        tiptoe_no_move = reward_utils.tolerance(
            jp.abs(horizontal_vel), bounds=(0.0, 0.1), margin=0.2,
        )

        # --- Per-mode rewards ---
        # 0: stand
        r_stand = stand_reward + control_cost

        # 1: walk_slow
        r_walk_slow = _move_reward(consts.WALK_SLOW_SPEED) + control_cost

        # 2: run
        r_run = _move_reward(consts.RUN_SPEED) + control_cost

        # 3: knee_down
        r_knee_down = (
            (self._config.knee_bend_weight * knee_bend_r
             + self._config.knee_height_weight * knee_height_r
             + self._config.upright_weight * upright
             + knee_no_move)
            / (self._config.knee_bend_weight + self._config.knee_height_weight
               + self._config.upright_weight + 1.0)
        ) + control_cost

        # 4: walk_backward
        r_walk_backward = _backward_move_reward(consts.WALK_BACKWARD_SPEED) + control_cost

        # 5: hop
        r_hop = (
            (hop_height_r + hop_upward_r + upright + hop_no_lateral) / 4.0
        ) + control_cost

        # 6: walk_fast
        r_walk_fast = _move_reward(consts.WALK_FAST_SPEED) + control_cost

        # 7: tiptoe
        r_tiptoe = (
            (2.0 * ankle_ext_r + tiptoe_height_r + upright + tiptoe_no_move) / 5.0
        ) + control_cost

        # Stack all mode rewards and compute weighted sum
        mode_rewards = jp.array([
            r_stand, r_walk_slow, r_run, r_knee_down,
            r_walk_backward, r_hop, r_walk_fast, r_tiptoe,
        ])
        metrics["reward/move"] = _move_reward(consts.WALK_SLOW_SPEED)

        reward = jp.dot(behavior_mode, mode_rewards)
        return reward

    @property
    def observation_size(self) -> int:
        # orientations(14) + height(1) + velocity(9) + joints(6)
        # + prev_action(6) + mode(8) = 44
        return 14 + 1 + 9 + 6 + 6 + consts.N_BEHAVIOR_MODES
