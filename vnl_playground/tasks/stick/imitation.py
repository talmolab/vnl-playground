"""Imitation task for stick bug.

Multi-clip imitation environment where the stick bug must track motion
capture reference data. Uses the legacy H5 format (qpos/qvel/xpos/xquat)
loaded via the unified ReferenceClips class.
"""

import collections
import tqdm
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import brax.math
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from jax import flatten_util

from .. import utils
from . import base as stick_base
from . import consts
from vnl_playground.tasks.reference_clips import ReferenceClips
from vnl_playground.tasks.reward_registry import RewardRegistry


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.STICK_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        mujoco_impl="jax",
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="newton",
        iterations=5,
        ls_iterations=5,
        naconmax=256,
        njmax=600,
        noslip_iterations=0,
        torque_actuators=False,
        rescale_factor=1.0,  # Mesh STAC fit was done with SCALE_FACTOR=1.
        reference_data_path=consts.IMITATION_REFERENCE_PATH,
        mocap_hz=50,
        clip_length=100,
        clip_set="all",
        reference_length=5,
        reference_stride=1,
        start_frame_range=[0, 44],
        qvel_init="zeros",
        keep_clips_idx=None,
        # CGS units (cm/g/s), v3 algebra-derived scales — see stick-ppo-imitation.yaml.
        reward_terms={
            "root_pos": {"exp_scale": 0.31, "weight": 1.0},
            "root_quat": {"exp_scale": 5.0, "weight": 1.0},
            "joints": {"exp_scale": 1.4, "weight": 2.0},
            "joints_vel": {"exp_scale": 1.0, "weight": 0.0},
            "bodies_pos": {"exp_scale": 2.2, "weight": 0.0},
            "end_eff": {"exp_scale": 4.4, "weight": 3.0},
            "leg_joints": {"exp_scale": 8.8, "weight": 3.0},
            "torso_z_range": {"healthy_z_range": (0.0, 10.0), "weight": 0.3},
            "control_cost": {"weight": 0.05},
            "control_diff_cost": {"weight": 0.1},
            "energy_cost": {"max_value": 50.0, "weight": 0.005},
        },
        termination_criteria={
            # root_too_far now uses xy-only distance (see _root_too_far)
            "root_too_far": {"max_distance": 5.0},
            "root_too_rotated": {"max_degrees": 90.0},
            "pose_error": {"max_l2_error": 15.0},
            "nan_termination": {},
        },
    )


_registry = RewardRegistry()


class Imitation(stick_base.StickBugEnv):
    """Multi-clip imitation environment for stick bug."""

    _registry = _registry

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
        clips: Optional[ReferenceClips] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        # rgba=None → keep the native chitin texture from sungaya_mat (set
        # in sungaya_inexpectata_mesh.xml). Recoloring would flatten the
        # mesh material to a solid color and lose the texture detail.
        self.add_stick(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
            rgba=None,
        )
        self.compile()
        if clips is not None:
            self.reference_clips = clips
        else:
            self.reference_clips = ReferenceClips(
                self._config.reference_data_path,
                self._config.clip_length,
                self._config.keep_clips_idx,
            )
        # CGS conversion: H5 stores positions in SI meters, but the compiled
        # model now lives in CGS centimeters (see base.py _apply_cgs_rescaling).
        # Scale every length field × 100. qvel is empty (qvel_init="zeros").
        _CGS_L = 100.0
        _FLOOR_Z_CGS = -0.9  # cm, matches arena.xml floor pos × CGS rescale
        clips_data = self.reference_clips._data_arrays

        # Scale xpos first so we can compute the z-shift from it.
        if "xpos" in clips_data:
            xpos = np.array(clips_data["xpos"], copy=True) * _CGS_L
        else:
            xpos = None

        # Z-shift: the STAC fit produced a reference clip where every claw
        # floats 1.5-2.4 mm above the floor across all frames. Under gravity
        # the policy can't physically realize that trajectory — it has to
        # fall to make ground contact. Shift the whole reference DOWN so the
        # lowest claw across the clip just touches the floor. This grounds
        # the imitation target so feet-on-ground is the natural attractor.
        if xpos is not None:
            min_body_z = float(xpos[..., 2].min())
            z_shift = _FLOOR_Z_CGS - min_body_z   # negative → shifts down
            xpos[..., 2] += z_shift
            clips_data["xpos"] = jp.array(xpos)
        else:
            z_shift = 0.0

        if "qpos" in clips_data:
            qpos = np.array(clips_data["qpos"], copy=True)
            qpos[..., 0:3] *= _CGS_L
            qpos[..., 2] += z_shift  # apply same z-shift to the free-joint root
            clips_data["qpos"] = jp.array(qpos)

        if "qvel" in clips_data and clips_data["qvel"].size > 0:
            qvel = np.array(clips_data["qvel"], copy=True)
            qvel[..., 0:3] *= _CGS_L
            clips_data["qvel"] = jp.array(qvel)
        max_n_clips = self.reference_clips.qpos.shape[0]
        if self._config.clip_set == "all":
            self._clip_set = max_n_clips
        elif isinstance(self._config.clip_set, (list, tuple, jp.ndarray, np.ndarray)):
            self._clip_set = jp.array(self._config.clip_set)
        else:
            raise ValueError(
                "config.clip_set must be 'all' or a list of clip indices."
                f" Got {self._config.clip_set}."
            )

        if (
            self.reference_clips._config is not None
            and "model" in self.reference_clips._config
            and self._config.rescale_factor
            != self.reference_clips._config["model"]["SCALE_FACTOR"]
        ):
            warnings.warn(
                f"Environment `rescale_factor` ({self._config.rescale_factor})"
                f" does not match the reference data `SCALE_FACTOR`"
                f" ({self.reference_clips._config['model']['SCALE_FACTOR']})."
            )

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        start_rng, clip_rng = jax.random.split(rng)
        if clip_idx is None:
            clip_idx = jax.random.choice(clip_rng, self._clip_set)
        if start_frame is None:
            start_frame = jax.random.randint(
                start_rng, (), *self._config.start_frame_range
            )
        data = self._reset_data(clip_idx, start_frame)
        info: dict[str, Any] = {
            "start_frame": start_frame,
            "reference_clip": clip_idx,
        }
        truncated = self._get_cur_frame(data, info) > self._last_valid_frame()
        info["truncated"] = jp.astype(truncated, float)
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()

        metrics = {
            "current_frame": jp.astype(self._get_cur_frame(data, info), float),
        }
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        truncated = self._get_cur_frame(data, info) > self._last_valid_frame()
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

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = collections.OrderedDict(
            task_obs=self._get_imitation_target(data, info),
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(state=obs)

    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            njmax=self._config.njmax,
            naconmax=self._config.naconmax,
        )
        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)
        _assert_all_are_prefix(
            reference.joint_names,
            self.get_joint_names(),
            "reference joints",
            "model joints",
        )
        data = data.replace(qpos=reference.qpos)
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel=reference.qvel)
        data = mjx.forward(self.mjx_model, data)
        return data

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    def _clip_length(self):
        return self.reference_clips.qpos.shape[1]

    def _last_valid_frame(self):
        return (
            self._clip_length()
            - (self._config.reference_length - 1) * self._config.reference_stride
            - 2
        )

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        time_in_frames = data.time * self._config.mocap_hz
        return jp.floor(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        return self.reference_clips.at(
            clip=info["reference_clip"], frame=self._get_cur_frame(data, info)
        )

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        return self.reference_clips.slice(
            clip=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.reference_length,
            stride=self._config.reference_stride,
        )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        reference = self._get_imitation_reference(data, info)

        root_pos = self.root_body(data).xpos
        root_quat = self.root_body(data).xquat
        root_targets = jax.vmap(
            lambda ref_pos: brax.math.rotate(ref_pos - root_pos, root_quat)
        )(reference.root_position)
        quat_targets = jax.vmap(
            lambda ref_quat: brax.math.relative_quat(ref_quat, root_quat)
        )(reference.root_quaternion)

        _assert_all_are_prefix(
            reference.joint_names,
            self.get_joint_names(),
            "reference joints",
            "model joints",
        )
        joint_targets = reference.joints - self._get_joint_angles(data)

        bodies_pos = self._get_bodies_pos(data, flatten=False)
        body_rel_pos = jp.array(
            [reference.body_xpos(name) - bodies_pos[name] for name in bodies_pos]
        )
        to_egocentric = jax.vmap(lambda diff_vec: brax.math.rotate(diff_vec, root_quat))
        body_targets = jax.vmap(to_egocentric)(body_rel_pos)

        return collections.OrderedDict(
            root=root_targets,
            quat=quat_targets,
            joint=joint_targets,
            body=body_targets,
        )

    # Rewards
    @_registry.reward("root_pos")
    def _root_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        distance = jp.linalg.norm(target.root_position - root_pos)
        metrics["root_pos_distance"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/root_pos"] = reward
        return reward

    @_registry.reward("root_quat")
    def _root_quat_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        rot_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        ang_dist_degrees = jp.rad2deg(rot_dist)
        metrics["root_angular_error"] = ang_dist_degrees
        reward = weight * jp.exp(-((ang_dist_degrees / exp_scale) ** 2) / 2)
        metrics["rewards/root_quat"] = reward
        return reward

    @_registry.reward("joints")
    def _joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(target.joints - joints)
        metrics["joint_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints"] = reward
        return reward

    @_registry.reward("joints_vel")
    def _joint_vels_reward(self, data, info, metrics, weight, exp_scale) -> float:
        target = self._get_current_target(data, info)
        joint_vels = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(target.joints_velocity - joint_vels)
        metrics["joint_vel_l2_error"] = distance
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        metrics["rewards/joints_vel"] = reward
        return reward

    def _get_bodies_dist(self, data, info, metrics, bodies=consts.BODIES) -> float:
        target = self._get_current_target(data, info)
        body_pos = self._get_bodies_pos(data, flatten=False)
        total_dist_sqr = 0.0
        for body_name in bodies:
            dist_sqr = jp.sum((body_pos[body_name] - target.body_xpos(body_name)) ** 2)
            metrics["body_errors/" + body_name] = jp.sqrt(dist_sqr)
            total_dist_sqr += dist_sqr
        return jp.sqrt(total_dist_sqr)

    @_registry.reward("bodies_pos")
    def _body_pos_reward(self, data, info, metrics, weight, exp_scale) -> float:
        total_dist = self._get_bodies_dist(data, info, metrics, consts.BODIES)
        metrics["body_errors/total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_registry.reward("end_eff")
    def _end_eff_reward(self, data, info, metrics, weight, exp_scale) -> float:
        total_dist = self._get_bodies_dist(data, info, metrics, consts.END_EFFECTORS)
        metrics["body_errors/end_eff_total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/end_eff"] = reward
        return reward

    @_registry.reward("leg_joints")
    def _leg_joints_reward(self, data, info, metrics, weight, exp_scale) -> float:
        """Tracks the 24 non-claw leg segment bodies — the joints of the
        six legs (hip, knee, ankle, tarsal). Complements `end_eff`, which
        only tracks the 6 claw tips, by also requiring the full leg pose
        to match the reference (not just the foot placement)."""
        total_dist = self._get_bodies_dist(data, info, metrics, consts.LEG_JOINTS)
        metrics["body_errors/leg_joints_total"] = total_dist
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        metrics["rewards/leg_joints"] = reward
        return reward

    @_registry.reward("torso_z_range")
    def _torso_z_range_reward(
        self, data, info, metrics, weight, healthy_z_range
    ) -> float:
        metrics["torso_z"] = torso_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        metrics["in_range"] = in_range.astype(float)
        reward = weight * in_range
        metrics["rewards/torso_z_range"] = reward
        return reward

    @_registry.reward("control_cost")
    def _control_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_sqr"] = ctrl_sqr = jp.sum(jp.square(info["action"]))
        cost = weight * ctrl_sqr
        metrics["rewards/control_cost"] = -cost
        return -cost

    @_registry.reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, weight) -> float:
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = jp.sum(
            jp.square(info["action"] - info["prev_action"])
        )
        cost = weight * ctrl_diff_sqr
        metrics["rewards/control_diff_cost"] = -cost
        return -cost

    @_registry.reward("energy_cost")
    def _energy_cost(self, data, info, metrics, weight, max_value) -> float:
        energy_use = jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator))
        metrics["energy_use"] = energy_use
        cost = weight * jp.minimum(energy_use, max_value)
        metrics["rewards/energy_cost"] = -cost
        return -cost

    # Termination
    @_registry.termination("root_too_far")
    def _root_too_far(self, data, info, max_distance) -> bool:
        """Horizontal (xy) root drift from reference. We deliberately
        ignore z because under early-training policies the bug can't
        support its weight, falls under gravity, and the z-component
        dominates the 3D distance — causing every episode to terminate
        in ~2 control steps before PPO can learn anything. Horizontal
        drift is the actually-meaningful signal for "policy lost track
        of the reference location"."""
        target = self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        # x, y components only — z drift (falling) is allowed.
        distance = jp.linalg.norm((target.root_position - root_pos)[:2])
        return distance > max_distance

    @_registry.termination("root_too_rotated")
    def _root_too_rotated(self, data, info, max_degrees) -> bool:
        target = self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        ang_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        return ang_dist > jp.deg2rad(max_degrees)

    @_registry.termination("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        pose_error = jp.linalg.norm(target.joints - joints)
        return pose_error > max_l2_error

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        return jp.any(jp.isnan(data.qpos))

    def _compile_with_ghost(self) -> mujoco.MjModel:
        """Compile a new MjModel with an attached transparent ghost stick."""
        spec = self._spec.copy()
        ghost_stick = mujoco.MjSpec.from_file(self._walker_xml_path)
        ghost_rescale = self._config.rescale_factor
        if (
            self.reference_clips._config is not None
            and "model" in self.reference_clips._config
        ):
            ghost_rescale = self.reference_clips._config["model"]["SCALE_FACTOR"]
        if ghost_rescale != 1.0:
            ghost_stick = utils.scale_spec(
                ghost_stick, ghost_rescale, root_body="reference_base"
            )
        for body in ghost_stick.worldbody.bodies:
            utils._recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])
        spawn_frame = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
        spawn_frame.attach_body(
            ghost_stick.body("reference_base"), "", suffix="-ghost"
        )
        ghost_model = spec.compile()
        # Mirror the SI rescaling applied in base.compile() so eval videos
        # use the same physics — without this, the trackcom camera sees
        # stale subtree_com and fails to follow the bug across frames.
        self._apply_si_rescaling(ghost_model)
        return ghost_model

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        add_labels=False,
        termination_extra_frames=0,
        render_ghost: bool = True,
    ) -> Sequence[np.ndarray]:
        """Renders a sequence of states with optional ghost stick bug."""
        if render_ghost:
            mj_model = self._compile_with_ghost()
        else:
            mj_model = self.mj_model
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        mj_data = mujoco.MjData(mj_model)

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        if camera is None:
            camera = self._default_render_camera

        rendered_frames = []
        for i, state in enumerate(trajectory):
            time_in_frames = state.data.time * self._config.mocap_hz
            frame = jp.floor(time_in_frames + state.info["start_frame"]).astype(int)
            clip = state.info["reference_clip"]
            ref = self.reference_clips.at(clip=clip, frame=frame)

            if render_ghost:
                mj_data.qpos = jp.concatenate((state.data.qpos, ref.qpos))
                mj_data.qvel = jp.concatenate((state.data.qvel, ref.qvel))
            else:
                mj_data.qpos = state.data.qpos
                mj_data.qvel = state.data.qvel
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
            if modify_scene_fns is not None:
                modify_scene_fns[i](renderer.scene)
            rendered_frame = renderer.render()
            rendered_frames.append(rendered_frame)
            if state.done:
                for t in range(termination_extra_frames):
                    rel_t = t / termination_extra_frames
                    fade_factor = 1 / (1 + np.exp(10 * (rel_t - 0.5)))
                    faded_frame = (rendered_frame * fade_factor).astype(np.uint8)
                    rendered_frames.append(faded_frame)
        return rendered_frames

    def render_optimized(
        self,
        rollout_source: Any,
        height: int = 480,
        width: int = 640,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        render_ghost: bool = True,
    ) -> List[np.ndarray]:
        """Render from precomputed qposes (track-mjx eval path)."""
        if isinstance(rollout_source, Mapping) and "qposes_rollout" in rollout_source:
            qposes_rollout = np.asarray(rollout_source["qposes_rollout"])
            qposes_ref = (
                np.asarray(rollout_source["qposes_ref"])
                if render_ghost and "qposes_ref" in rollout_source
                else None
            )
        else:
            qposes_rollout = np.asarray(rollout_source.data.qpos)
            qposes_ref = None
            if render_ghost:
                clip_idx = int(
                    np.asarray(rollout_source.info["reference_clip"]).reshape(-1)[0]
                )
                start_frame = np.asarray(rollout_source.info["start_frame"])
                times = np.asarray(rollout_source.data.time)
                frame_indices = np.floor(
                    times * float(self._config.mocap_hz) + start_frame
                ).astype(np.int32)
                ref_qpos = np.asarray(self.reference_clips.qpos[clip_idx])
                qposes_ref = ref_qpos[frame_indices]

        if render_ghost:
            mj_model = self._compile_with_ghost()
            qpos_list = [
                np.concatenate((qroll, qref))
                for qroll, qref in zip(qposes_rollout, qposes_ref, strict=False)
            ]
        else:
            mj_model = self.mj_model
            qpos_list = qposes_rollout

        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

        if camera is None:
            camera = self._default_render_camera
        if scene_option is None:
            scene_option = mujoco.MjvOption()
            scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]

        frames = []
        for qpos in tqdm.tqdm(qpos_list, desc="Rendering"):
            mj_data.qpos = qpos
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
            frames.append(renderer.render())

        renderer.close()
        return frames

    def verify_reference_data(self, atol: float = 5e-3) -> bool:
        """Check that env-from-qpos reproduces the reference body positions."""
        def test_frame(clip_idx: int, frame: int) -> dict[str, bool]:
            data = self._reset_data(clip_idx, frame)
            reference = self.reference_clips.at(clip=clip_idx, frame=frame)
            checks = collections.OrderedDict()
            checks["root_pos"] = jp.allclose(
                self.root_body(data).xpos, reference.root_position, atol=atol
            )
            checks["root_quat"] = jp.allclose(
                self.root_body(data).xquat, reference.root_quaternion, atol=atol
            )
            checks["joints"] = jp.allclose(
                self._get_joint_angles(data), reference.joints, atol=atol
            )
            body_pos = self._get_bodies_pos(data, flatten=False)
            for body_name, bp in body_pos.items():
                checks[f"body_xpos/{body_name}"] = jp.allclose(
                    bp, reference.body_xpos(body_name), atol=atol
                )
            return checks

        @jax.jit
        def test_clip(clip_idx: int):
            return jax.vmap(test_frame, in_axes=(None, 0))(
                clip_idx, jp.arange(self._clip_length())
            )

        _assert_all_are_prefix(
            self.reference_clips.joint_names,
            self.get_joint_names(),
            "reference joints",
            "model joints",
        )
        if isinstance(self._clip_set, int):
            clip_idxs = jp.arange(self._clip_set)
        else:
            clip_idxs = self._clip_set

        any_failed = False
        for clip in clip_idxs:
            if clip < 0 or clip >= self.reference_clips.qpos.shape[0]:
                raise ValueError(
                    f"Clip index {clip} is out of range. Reference"
                    f"data has {self.reference_clips.qpos.shape[0]} clips."
                )
            test_result = test_clip(clip)
            for name, result in test_result.items():
                n_failed = jp.sum(np.logical_not(result))
                if n_failed > 0:
                    first_failed_frame = jp.argmax(np.logical_not(result))
                    warnings.warn(
                        f"Reference data verification failed for {n_failed}"
                        f" frames for check '{name}' for clip {clip}."
                        f" First failure at frame {first_failed_frame}."
                    )
                    any_failed = True
        return not any_failed


def _assert_all_are_prefix(a, b, a_name="a", b_name="b"):
    if isinstance(a, map):
        a = list(a)
    if isinstance(b, map):
        b = list(b)
    if len(a) != len(b):
        raise AssertionError(
            f"{a_name} has length {len(a)} but {b_name} has length {len(b)}."
        )
    for a_el, b_el in zip(a, b):
        if not b_el.startswith(a_el):
            raise AssertionError(
                f"Comparing {a_name} and {b_name}. Expected {a_el} to match {b_el}."
            )
