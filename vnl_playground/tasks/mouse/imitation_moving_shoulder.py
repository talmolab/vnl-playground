"""Mouse arm imitation with IK-driven shoulder translation.

The shoulder_tx/ty/tz DOFs are overwritten from the STAC v16 IK reference
after every env step. The muscle policy learns only the 4 hinge-joint
actuations (sh_rotation, sh_extension, sh_elv, elbow). The three IK-driven
dims are masked out of the `joints` and `joints_vel` rewards and the
`pose_error` termination.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from vnl_playground.tasks.mouse.consts import (
    JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH,
    MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH,
)
from vnl_playground.tasks.mouse.imitation import (
    MouseImitation,
    default_config as imitation_default_config,
    _registry as _parent_registry,
)
from vnl_playground.tasks.mouse.reference_clips import MouseReferenceClips
from vnl_playground.tasks.reward_registry import RewardRegistry


def default_config() -> config_dict.ConfigDict:
    """Moving-shoulder defaults: muscle xml + v16 IK clips + 3 IK-driven dims."""
    cfg = imitation_default_config()
    cfg.walker_xml_path = JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH
    cfg.reference_data_path = str(MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH)
    cfg.recompute_kinematics = False  # ref was STAC-fit with same kinematic chain
    cfg.ik_driven_dims = 3  # leading qpos/qvel dims to snap + mask
    # Moving-shoulder XML has 'wrist' (not 'wrist_body') and no 'radius'.
    cfg.tracked_bodies = ["scapula", "humerus", "ulna", "wrist"]
    cfg.end_effector = "wrist"
    return cfg


# Build a registry that inherits all parent entries, then override specific ones.
_registry = RewardRegistry()
_registry.rewards.update(_parent_registry.rewards)
_registry.terminations.update(_parent_registry.terminations)


class MouseImitationMovingShoulder(MouseImitation):
    """MouseImitation variant that snaps leading qpos dims to IK every step."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list, dict]]] = None,
        clips: Optional[MouseReferenceClips] = None,
    ) -> None:
        super().__init__(config, config_overrides, clips)

        n_ik = int(self._config.ik_driven_dims)
        assert n_ik >= 0, n_ik
        if n_ik > 0:
            # Sanity: reference qpos leading dims stay inside the slide-joint range.
            # reference_clips.qpos shape: (n_clips, n_frames, nq)
            lead = self.reference_clips.qpos[:, :, :n_ik]
            max_abs = float(jp.max(jp.abs(lead)))
            assert max_abs < 0.01, (
                f"IK-driven qpos leading dims exceed slide-joint range "
                f"(max |q|={max_abs:.4f} >= 0.01). Widen the XML range or "
                f"rescale the IK before training."
            )

    def _override_ik_dims(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> mjx.Data:
        """Snap the leading qpos/qvel dims to the IK reference for the current frame."""
        n = int(self._config.ik_driven_dims)
        if n <= 0:
            return data
        cur_frame = self._get_cur_frame(data, info)
        last_valid = self._clip_length() - 1
        cur_frame_clamped = jp.minimum(cur_frame, last_valid)
        ref = self.reference_clips.at(
            clip=info["reference_clip"], frame=cur_frame_clamped
        )
        data = data.replace(
            qpos=data.qpos.at[:n].set(ref.qpos[:n]),
            qvel=data.qvel.at[:n].set(ref.qvel[:n]),
        )
        # Refresh xpos/xquat so downstream consumers (rewards, obs) see the
        # snapped base pose in world coordinates.
        data = mjx.forward(self.mjx_model, data)
        return data

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        state = super().reset(rng, clip_idx, start_frame)
        # Reference qpos already matches at start, but the snap is idempotent and
        # guards against any rounding drift.
        data = self._override_ik_dims(state.data, state.info)
        obs = self._get_obs(data, state.info)
        return state.replace(data=data, obs=obs)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        # Snap IK-driven dims BEFORE computing obs/reward/termination so that
        # every downstream consumer sees the kinematically correct pose.
        data = self._override_ik_dims(data, info)

        last_valid_frame = self._clip_length() - self._config.reference_length - 1
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
        current_frame = self._get_cur_frame(data, info)
        state.metrics["current_frame"] = jp.astype(current_frame, float)
        return state

    # ---- Reward / termination overrides: mask leading IK-driven dims ----

    @_registry.reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        n = int(self._config.ik_driven_dims)
        distance = jp.linalg.norm(target.joints[n:] - data.qpos[n:])
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_registry.reward("joints_vel")
    def _joints_vel_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        n = int(self._config.ik_driven_dims)
        distance = jp.linalg.norm(target.joints_velocity[n:] - data.qvel[n:])
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    @_registry.termination("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        n = int(self._config.ik_driven_dims)
        pose_error = jp.linalg.norm(target.joints[n:] - data.qpos[n:])
        return pose_error > max_l2_error
