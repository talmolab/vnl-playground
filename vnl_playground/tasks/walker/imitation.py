"""Walker imitation environment with online reference generation.

Step 2 environment: uses a trained multi-behavior policy (from Step 1)
to generate reference trajectories on-the-fly. The imitation task
provides future trajectory observations to the encoder, which produces
latent representations for topological analysis.

Observation structure (matches rodent imitation pattern):
{
    "state": {
        "task_obs": {root, quat, joint, body},  # Future trajectory targets
        "proprioception": {orientations, height, ...},
    },
    "privileged_state": { ... same as state ... },
}
"""

import collections
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import brax.math
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.walker import consts
from vnl_playground.tasks.walker.base import WalkerEnv
from vnl_playground.tasks.walker.online_reference import (
    OnlineReferenceGenerator,
    WalkerTrajectory,
)
from vnl_playground.tasks.reward_registry import RewardRegistry

_registry = RewardRegistry()


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        sim_dt=0.0025,
        ctrl_dt=0.025,
        episode_length=1000,
        mujoco_impl="jax",
        nconmax=50_000,
        njmax=100,
        # Reference generation
        reference_length=5,        # Future frames to include in observation
        trajectory_length=200,     # Total frames per generated trajectory
        mocap_hz=40,               # Frame rate = 1 / ctrl_dt
        mode_duration_mean=150,
        mode_duration_min=60,
        warmup_frames=0,               # Standing warmup before actual behavior (0 = disabled)
        warmup_transition_frames=40,   # Smooth blend from standing to first mode
        # Reward terms
        reward_terms={
            "root_pos": {"exp_scale": 0.1, "weight": 1.0},
            "root_angle": {"exp_scale": 20.0, "weight": 1.0},
            "joints": {"exp_scale": 1.4, "weight": 1.0},
            "joints_vel": {"exp_scale": 1.0, "weight": 1.0},
            "bodies_pos": {"exp_scale": 0.25, "weight": 1.0},
            "control_cost": {"weight": 0.02},
            "energy_cost": {"max_value": 50.0, "weight": 0.01},
        },
        # Termination criteria
        termination_criteria={
            "root_too_far": {"max_distance": 0.3},
            "pose_error": {"max_l2_error": 4.5},
            "nan_termination": {},
        },
    )


class WalkerImitation(WalkerEnv):
    """Walker imitation with online reference trajectory generation.

    Each episode generates a fresh reference trajectory using the trained
    Step 1 policy. The imitation target (future trajectory) is provided
    as task_obs to the encoder, matching the rodent imitation pattern.
    """

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[
            Dict[str, Union[str, int, list[Any], dict]]
        ] = None,
        generator: Optional[OnlineReferenceGenerator] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        if generator is None:
            raise ValueError(
                "WalkerImitation requires an OnlineReferenceGenerator. "
                "Pass it via the `generator` argument."
            )
        self.generator = generator
        self._mocap_hz = self._config.mocap_hz

    def reset(
        self,
        rng: jax.Array,
        behavior_schedule: Optional[jp.ndarray] = None,
    ) -> mjx_env.State:
        """Reset: generate a fresh reference trajectory and initialize walker.

        Args:
            rng: Random key.
            behavior_schedule: Optional (trajectory_length, N_BEHAVIOR_MODES)
                one-hot array. If None, samples a random schedule.
        """
        rng, gen_rng, reset_rng = jax.random.split(rng, 3)

        # Generate behavior schedule if not provided
        if behavior_schedule is None:
            behavior_schedule = OnlineReferenceGenerator.sample_behavior_schedule(
                gen_rng,
                n_frames=self._config.trajectory_length,
                mode_duration_mean=self._config.get(
                    "mode_duration_mean", 150
                ),
                mode_duration_min=self._config.get(
                    "mode_duration_min", 60
                ),
            )

        # Generate reference trajectory
        reference = self.generator.generate(gen_rng, behavior_schedule)

        # Initialize walker to first frame of reference
        start_frame = 0
        data = mjx_env.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(qpos=reference.qpos[start_frame])
        data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
        data = mjx.forward(self.mjx_model, data)

        info = {
            "start_frame": jp.array(start_frame, dtype=jp.int32),
            "reference": reference,
            "behavior_labels": reference.behavior_labels,
            "prev_action": jp.zeros(self.action_size),
            "action": jp.zeros(self.action_size),
        }

        last_valid_frame = (
            self._config.trajectory_length
            - self._config.reference_length
            - 1
        )
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jp.astype(truncated, float)

        metrics = {
            "current_frame": jp.float32(self._get_cur_frame(data, info)),
            "behavior_mode": jp.float32(
                jp.argmax(reference.behavior_labels[start_frame])
            ),
        }

        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(
            data, obs, reward, jp.astype(done, float), metrics, info
        )

    def soft_reset(self, state: mjx_env.State) -> mjx_env.State:
        """Reset agent to qpos[0] of the *existing* reference trajectory.

        Unlike ``reset()``, this does NOT generate a new trajectory.
        The agent is placed back at the start of the same reference,
        making it cheap enough to call on every episode termination.
        """
        reference = state.info["reference"]
        start_frame = 0

        data = state.data.replace(
            qpos=reference.qpos[start_frame],
            qvel=jp.zeros(self.mjx_model.nv),
            time=jp.float32(0),
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            **state.info,
            "start_frame": jp.array(start_frame, dtype=jp.int32),
            "prev_action": jp.zeros(self.action_size),
            "action": jp.zeros(self.action_size),
            "truncated": jp.float32(0),
        }

        metrics = {
            **state.metrics,
            "current_frame": jp.float32(start_frame),
            "behavior_mode": jp.float32(
                jp.argmax(reference.behavior_labels[start_frame])
            ),
        }

        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = jp.float32(0)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(
        self, state: mjx_env.State, action: jax.Array
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        last_valid_frame = (
            self._config.trajectory_length
            - self._config.reference_length
            - 1
        )
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jp.astype(truncated, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        terminated = self._is_done(data, info, state.metrics)
        done = jp.logical_or(terminated, info["truncated"])
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        cur_frame = self._get_cur_frame(data, info)
        state.metrics["current_frame"] = jp.float32(cur_frame)
        state.metrics["behavior_mode"] = jp.float32(
            jp.argmax(info["behavior_labels"][cur_frame])
        )
        return state

    # -------------------------------------------------------------------------
    # Observation (matches rodent imitation pattern)
    # -------------------------------------------------------------------------

    def _get_obs(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        obs = collections.OrderedDict(
            task_obs=self._get_imitation_target(data, info),
            proprioception=self._get_proprioception(
                data, info, flatten=False
            ),
        )
        return collections.OrderedDict(
            state=obs,
            privileged_state=obs,
        )

    def _get_cur_frame(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> int:
        time_in_frames = data.time * self._mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> dict:
        """Get reference data at the current frame."""
        frame = self._get_cur_frame(data, info)
        ref = info["reference"]
        return {
            "qpos": ref.qpos[frame],
            "qvel": ref.qvel[frame],
            "xpos": ref.xpos[frame],
            "xquat": ref.xquat[frame],
        }

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> dict:
        """Get future reference frames for the observation (task_obs)."""
        start = self._get_cur_frame(data, info) + 1
        length = self._config.reference_length
        ref = info["reference"]

        return {
            "qpos": jax.lax.dynamic_slice(
                ref.qpos, (start, 0), (length, ref.qpos.shape[1])
            ),
            "qvel": jax.lax.dynamic_slice(
                ref.qvel, (start, 0), (length, ref.qvel.shape[1])
            ),
            "xpos": jax.lax.dynamic_slice(
                ref.xpos,
                (start, 0, 0),
                (length, ref.xpos.shape[1], ref.xpos.shape[2]),
            ),
            "xquat": jax.lax.dynamic_slice(
                ref.xquat,
                (start, 0, 0),
                (length, ref.xquat.shape[1], ref.xquat.shape[2]),
            ),
        }

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Compute egocentric imitation targets from future reference frames.

        Returns same structure as rodent imitation: root, quat, joint, body.
        """
        future_ref = self._get_imitation_reference(data, info)

        # Current walker state
        cur_root_pos = data.xpos[self._torso_id]  # (3,)
        cur_root_xmat = data.xmat[self._torso_id]  # (3, 3)
        cur_root_quat = _xmat_to_quat(cur_root_xmat)

        # Future root positions (egocentric)
        future_root_pos = future_ref["xpos"][
            :, self._torso_id, :
        ]  # (ref_len, 3)
        root_targets = jax.vmap(
            lambda ref_pos: brax.math.rotate(
                ref_pos - cur_root_pos, cur_root_quat
            )
        )(future_root_pos)

        # Future root quaternions (relative)
        future_root_quat = future_ref["xquat"][
            :, self._torso_id, :
        ]  # (ref_len, 4)
        quat_targets = jax.vmap(
            lambda ref_quat: brax.math.relative_quat(
                ref_quat, cur_root_quat
            )
        )(future_root_quat)

        # Future joint angles (difference from current)
        cur_joints = self._get_joint_angles(data)
        future_joints = future_ref["qpos"][
            :, consts.N_ROOT_QPOS:
        ]  # (ref_len, 6)
        joint_targets = future_joints - cur_joints

        # Future body positions (egocentric)
        cur_body_pos = jp.array(
            [data.xpos[self._body_ids[name]] for name in consts.BODIES]
        )  # (n_bodies, 3)
        future_body_pos = jp.array(
            [
                future_ref["xpos"][:, self._body_ids[name], :]
                for name in consts.BODIES
            ]
        )  # (n_bodies, ref_len, 3)
        body_rel = (
            future_body_pos - cur_body_pos[:, None, :]
        )  # (n_bodies, ref_len, 3)
        to_ego = jax.vmap(
            lambda diff: brax.math.rotate(diff, cur_root_quat)
        )
        body_targets = jax.vmap(to_ego)(body_rel)  # (n_bodies, ref_len, 3)

        return collections.OrderedDict(
            root=root_targets,
            quat=quat_targets,
            joint=joint_targets,
            body=body_targets,
        )

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------

    @_registry.reward("root_pos")
    def _root_pos_reward(
        self, data, info, metrics, weight, exp_scale
    ) -> float:
        target = self._get_current_target(data, info)
        root_pos = data.xpos[self._torso_id]
        target_root_pos = target["xpos"][self._torso_id]
        distance = jp.linalg.norm(target_root_pos - root_pos)
        metrics["root_pos_distance"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/root_pos"] = reward
        return reward

    @_registry.reward("root_angle")
    def _root_angle_reward(
        self, data, info, metrics, weight, exp_scale
    ) -> float:
        """Reward for matching root orientation (y-axis rotation)."""
        target = self._get_current_target(data, info)
        cur_angle = data.qpos[2]  # rooty
        target_angle = target["qpos"][2]
        angle_diff = jp.abs(cur_angle - target_angle)
        angle_diff_degrees = jp.rad2deg(angle_diff)
        metrics["root_angular_error"] = angle_diff_degrees
        reward = weight * jp.exp(
            -((angle_diff_degrees / exp_scale) ** 2) / 2
        )
        metrics["rewards/root_angle"] = reward
        return reward

    @_registry.reward("joints")
    def _joints_reward(
        self, data, info, metrics, weight, exp_scale
    ) -> float:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        target_joints = target["qpos"][consts.N_ROOT_QPOS:]
        distance = jp.linalg.norm(target_joints - joints)
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_registry.reward("joints_vel")
    def _joints_vel_reward(
        self, data, info, metrics, weight, exp_scale
    ) -> float:
        target = self._get_current_target(data, info)
        joint_vels = self._get_joint_ang_vels(data)
        target_vels = target["qvel"][consts.N_ROOT_QVEL:]
        distance = jp.linalg.norm(target_vels - joint_vels)
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    @_registry.reward("bodies_pos")
    def _body_pos_reward(
        self, data, info, metrics, weight, exp_scale
    ) -> float:
        target = self._get_current_target(data, info)
        total_dist_sqr = 0.0
        for name in consts.BODIES:
            cur_pos = data.xpos[self._body_ids[name]]
            target_pos = target["xpos"][self._body_ids[name]]
            dist_sqr = jp.sum((cur_pos - target_pos) ** 2)
            metrics[f"body_errors/{name}"] = jp.sqrt(dist_sqr)
            total_dist_sqr += dist_sqr
        total_dist = jp.sqrt(total_dist_sqr)
        metrics["body_errors/total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_registry.reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        ctrl_sqr = jp.sum(jp.square(info["action"]))
        metrics["ctrl_sqr"] = ctrl_sqr
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = -cost
        return -cost

    @_registry.reward("energy_cost")
    def _energy_cost(
        self, data, info, metrics, weight, max_value
    ) -> float:
        energy_use = jp.sum(
            jp.abs(data.qvel) * jp.abs(data.qfrc_actuator)
        )
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = -cost
        return -cost

    # -------------------------------------------------------------------------
    # Termination
    # -------------------------------------------------------------------------

    @_registry.termination("root_too_far")
    def _root_too_far(self, data, info, max_distance) -> bool:
        target = self._get_current_target(data, info)
        root_pos = data.xpos[self._torso_id]
        target_root = target["xpos"][self._torso_id]
        distance = jp.linalg.norm(target_root - root_pos)
        return distance > max_distance

    @_registry.termination("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        target_joints = target["qpos"][consts.N_ROOT_QPOS:]
        pose_error = jp.linalg.norm(target_joints - joints)
        return pose_error > max_l2_error

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    # -------------------------------------------------------------------------
    # Reward/termination dispatch (using registry)
    # -------------------------------------------------------------------------

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: dict
    ) -> float:
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += self._registry.rewards[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def _is_done(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: dict
    ) -> bool:
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            terminated = self._registry.terminations[name](
                self, data, info, **kwargs
            )
            any_terminated = jp.logical_or(any_terminated, terminated)
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            flatten_util.ravel_pytree(
                obs_size["state"]["proprioception"]
            )[0]
        )

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self) -> int:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(
            lambda x: jp.prod(jp.array(x.shape)), obs
        )


def _xmat_to_quat(xmat: jp.ndarray) -> jp.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    trace = xmat[0, 0] + xmat[1, 1] + xmat[2, 2]
    w = jp.sqrt(jp.maximum(0, 1 + trace)) / 2
    x = jp.sqrt(
        jp.maximum(0, 1 + xmat[0, 0] - xmat[1, 1] - xmat[2, 2])
    ) / 2
    y = jp.sqrt(
        jp.maximum(0, 1 - xmat[0, 0] + xmat[1, 1] - xmat[2, 2])
    ) / 2
    z = jp.sqrt(
        jp.maximum(0, 1 - xmat[0, 0] - xmat[1, 1] + xmat[2, 2])
    ) / 2
    x = jp.copysign(x, xmat[2, 1] - xmat[1, 2])
    y = jp.copysign(y, xmat[0, 2] - xmat[2, 0])
    z = jp.copysign(z, xmat[1, 0] - xmat[0, 1])
    return jp.array([w, x, y, z])
