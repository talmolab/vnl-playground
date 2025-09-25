"""C. elegans imitation learning environment.

This module implements an imitation learning environment for C. elegans that
allows training agents to mimic reference motion clips.
"""

import collections
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import brax.math
import cv2
import jax
import jax.flatten_util
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.celegans.reference_clips import ReferenceClips

from . import base as worm_base
from . import consts


def default_config() -> config_dict.ConfigDict:
    """Create default configuration for the imitation environment.

    Returns:
        Configuration dictionary with default parameters for imitation learning.
    """
    return config_dict.create(
        walker_xml_path=str(consts.CELEGANS_XML_PATH),
        arena_xml_path=str(consts.ARENA_XML_PATH),
        root_body=consts.ROOT,
        joints=consts.JOINTS,
        bodies=consts.BODIES,
        end_effectors=consts.END_EFFECTORS,
        touch_sensors=consts.TOUCH_SENSORS,
        sensors=consts.SENSORS,
        mujoco_impl="jax",
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="cg",
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        nconmax=256,
        njmax=256,
        torque_actuators=False,
        rescale_factor=1.0,
        dim=3,
        friction=(1, 1, 0.005, 0.0001, 0.0001),
        solimp=(0.9, 0.95, 0.001, 0.5, 2),
        mocap_hz=20,
        reference_clips=ReferenceClips(
            data_path=consts.REFERENCE_H5_PATH, n_frames_per_clip=250
        ),
        clip_set="all",
        reference_length=5,
        start_frame_range=[0, 44],
        qvel_init="zeros",
        with_ghost=False,
        reward_terms={
            # Imitation rewards
            "root_pos": {"exp_scale": 0.05, "weight": 1.0},  # Meters
            "root_quat": {"exp_scale": 0.0625, "weight": 1.0},  # Degrees
            "joints": {"exp_scale": 16, "weight": 1.0},  # Joint-space L2 distance
            "joints_vel": {
                "exp_scale": 0.02,
                "weight": 1.0,
            },  # Joint velocity-space L2 distance
            "bodies_pos": {
                "exp_scale": 0.125,
                "weight": 1.0,
            },  # Distance in concatenated euclidean space
            "end_eff": {
                "exp_scale": 0.002,
                "weight": 1.0,
            },  # Distance in concatenated euclidean space
            "upright": {"healthy_z_range": (0.0, 1.0), "weight": 0.0},
        },
        # Costs / regularizers
        cost_terms={
            "control": {"weight": 0.02},
            "control_diff": {"weight": 0.02},
            "energy": {"max_value": 50.0, "weight": 0.0},
        },
        termination_criteria={
            "fall": {"healthy_z_range": (0.0, 1.0)},
            "root_too_far": {"max_distance": 0.01},  # Meters
            "root_too_rotated": {"max_degrees": 60.0},  # Degrees
            "pose_error": {"max_l2_error": 4.5},  # Joint-space L2 distance
        },
    )


_REWARD_FCN_REGISTRY: Dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: Dict[str, Callable] = {}
_COST_FCN_REGISTRY: Dict[str, Callable] = {}


class Imitation(worm_base.CelegansEnv):
    """Multi-clip imitation environment for C. elegans.

    This environment enables training agents to imitate reference motion clips
    by providing rewards based on how closely the agent's motion matches the
    reference data.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[
            Dict[str, Union[str, int, List[Any], Dict[str, Any]]]
        ] = None,
    ) -> None:
        """Initialize the imitation environment.

        Args:
            config: Configuration dictionary for the environment.
            config_overrides: Optional overrides for the configuration.
        """
        super().__init__(config, config_overrides)

        self.add_worm(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
            dim=self._config.dim,
            friction=self._config.friction,
            solimp=self._config.solimp,
        )
        if self._config.with_ghost:
            self.add_ghost_worm(
                rescale_factor=self._config.rescale_factor, dim=self._config.dim
            )

        self.compile()
        self.reference_clips = (
            self._config.reference_clips
        )  # ReferenceClips(self._config.reference_data_path,
        # self._config.clip_length)
        max_n_clips = self.reference_clips.qpos.shape[0]
        if self._config.clip_set == "all":
            self._clip_set = max_n_clips
        else:
            raise NotImplementedError("'all' is the only implemented set of clips.")

        self._mocap_dt = 1 / self._config.mocap_hz
        self._steps_for_cur_frame = self._mocap_dt / self._config.ctrl_dt
        self._n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        if self._n_steps == 0:
            warnings.warn(
                f"Simulation will not advance! Please increase `ctrl_dt` from {self._config.ctrl_dt} to at least {self._config.sim_dt}."
            )

    def reset(
        self,
        rng: jax.Array,
        clip_idx: Optional[int] = None,
        start_frame: Optional[int] = None,
    ) -> mjx_env.State:
        """Reset the environment to initial state.

        Args:
            rng: Random number generator key.
            clip_idx: Specific clip index to use. If None, randomly selected.
            start_frame: Specific start frame. If None, randomly selected.

        Returns:
            Initial environment state.
        """
        start_rng, clip_rng = jax.random.split(rng)
        if clip_idx is None:
            clip_idx = jax.random.choice(clip_rng, self._num_clips())
        if start_frame is None:
            start_frame = jax.random.randint(
                start_rng, (), *self._config.start_frame_range
            )
        data = self._reset_data(clip_idx, start_frame)
        info: dict[str, Any] = {
            "start_frame": start_frame,
            "reference_clip": clip_idx,
            "prev_ctrl": jp.zeros((self.action_size,)),
        }

        last_valid_frame = self._clip_length() - self._config.reference_length - 1
        info["truncated"] = jp.astype(
            self._get_cur_frame(data, info) > last_valid_frame, float
        )
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()

        obs = self._get_obs(data, info)
        # Used to initialize our intention network
        info["reference_obs_size"] = jax.flatten_util.ravel_pytree(
            obs["imitation_target"]
        )[0].shape[-1]  # TODO: use getter functions
        info["proprioceptive_obs_size"] = jax.flatten_util.ravel_pytree(
            obs["proprioception"]
        )[0].shape[-1]  # TODO: use getter functions

        obs = jax.flatten_util.ravel_pytree(obs)[0]  # TODO: Use wrapper instead.

        rewards, dists = self._get_rewards(data, info)
        reward = jp.sum(jax.flatten_util.ravel_pytree(rewards)[0])
        costs, magnitudes = self._get_costs(data, info)
        cost = jp.sum(jax.flatten_util.ravel_pytree(costs)[0])
        total_reward = reward - cost

        termination_conditions = self._get_termination_conditions(data, info)
        done = jp.any(jax.flatten_util.ravel_pytree(termination_conditions)[0])

        metrics = {
            **rewards,
            **costs,
            **{n: jp.astype(t, float) for n, t in termination_conditions.items()},
            **dists,
            **magnitudes,
            "current_frame": jp.astype(self._get_cur_frame(data, info), float),
        }
        return mjx_env.State(
            data, obs, total_reward, jp.astype(done, float), metrics, info
        )

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        """Step the environment forward by one timestep.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            Next environment state.
        """
        n_steps = self._n_steps
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        last_valid_frame = self._clip_length() - self._config.reference_length - 1
        info["truncated"] = jp.astype(
            self._get_cur_frame(data, info) > last_valid_frame, float
        )
        info["prev_action"] = state.info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)
        obs = jax.flatten_util.ravel_pytree(obs)[0]

        termination_conditions = self._get_termination_conditions(data, info)
        terminated = jp.any(jax.flatten_util.ravel_pytree(termination_conditions)[0])
        done = jp.logical_or(terminated, info["truncated"])

        rewards, dists = self._get_rewards(data, info)
        costs, magnitudes = self._get_costs(data, info)

        total_reward = jp.sum(jax.flatten_util.ravel_pytree(rewards)[0]) - jp.sum(
            jax.flatten_util.ravel_pytree(costs)[0]
        )
        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=total_reward,
            done=jp.astype(done, float),
        )

        metrics = {
            **rewards,
            **costs,
            **{n: jp.astype(t, float) for n, t in termination_conditions.items()},
            **dists,
            **magnitudes,
            "current_frame": jp.astype(self._get_cur_frame(data, info), float),
        }
        state.metrics.update(metrics)

        return state

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        """Get observations from the environment.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Dictionary containing proprioception and imitation target data.
        """
        return collections.OrderedDict(
            proprioception=self._get_proprioception(data, flatten=False),
            imitation_target=self._get_imitation_target(data, info),
        )

    def _get_rewards(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        """Compute all reward terms for the current state.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Tuple of (reward_dict, distance_dict) where reward_dict contains
            computed rewards and distance_dict contains the corresponding distances.
        """
        rewards, dists = dict(), dict()
        for name, kwargs in self._config.reward_terms.items():
            r, d = _REWARD_FCN_REGISTRY[name](self, data, info, **kwargs)
            rewards[f"{name}_reward"] = r
            dists[f"{name}_dist"] = d
        return rewards, dists

    def _get_termination_conditions(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, bool]:
        """Check all termination conditions for the current state.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Dictionary mapping termination condition names to boolean values.
        """
        termination_reasons = dict()
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            termination_reasons[name] = termination_fcn(self, data, info, **kwargs)
        return termination_reasons

    def _get_costs(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        """Compute all cost terms for the current state.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Tuple of (cost_dict, magnitude_dict) where cost_dict contains
            computed costs and magnitude_dict contains the underlying magnitudes.
        """
        costs, magnitudes = dict(), dict()
        for name, kwargs in self._config.cost_terms.items():
            c, m = _COST_FCN_REGISTRY[name](self, data, info, **kwargs)
            costs[f"{name}_cost"] = c
            magnitudes[name] = m
        return costs, magnitudes

    def _reset_data(self, clip_idx: int, start_frame: int) -> mjx.Data:
        """Reset simulation data to a specific clip and frame.

        Args:
            clip_idx: Index of the clip to reset to.
            start_frame: Frame within the clip to start from.

        Returns:
            Initialized MuJoCo simulation data.
        """
        data = mjx.make_data(self.mj_model)  # ,
        #     impl=self._config.mujoco_impl,
        #     nconmax=self._config.nconmax,
        #     njmax=self._config.nconmax,
        # )
        reference = self.reference_clips.at(clip=clip_idx, frame=start_frame)
        data = data.replace(qpos=reference.qpos)
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "noise":
            raise NotImplementedError("qvel_init='noise' is not yet implemented.")
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel=reference.qvel)
        data = mjx.forward(self.mjx_model, data)
        return data

    def null_action(self) -> jp.ndarray:
        """Get a null (zero) action.

        Returns:
            Zero action array.
        """
        return jp.zeros(self.action_size)

    def _num_clips(self) -> int:
        """Get the number of available clips.

        Returns:
            Number of reference clips.
        """
        return self.reference_clips.n_clips

    def _clip_length(self) -> int:
        """Get the length of each clip.

        Returns:
            Number of frames per clip.
        """
        return self.reference_clips.frames_per_clip

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        """Get the current frame index in the reference clip.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Current frame index.
        """
        return jp.floor(data.time * self._config.mocap_hz + info["start_frame"]).astype(
            int
        )

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get reference data for the current timestep.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Reference clips data for the current frame.
        """
        return self.reference_clips.at(
            clip=info["reference_clip"], frame=self._get_cur_frame(data, info)
        )

    def _get_reference_clip(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the full reference clip.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Complete reference clip data.
        """
        return self.reference_clips.slice(
            clip=info["reference_clip"], start_frame=0, length=self._clip_length()
        )

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get future reference frames for imitation target.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Reference clips data for future frames used as imitation target.
        """
        return self.reference_clips.slice(
            clip=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.reference_length,
        )  # TODO: Use reference_clips.slice instead

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> Mapping[str, jp.ndarray]:
        """Get imitation targets transformed to agent's egocentric frame.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.

        Returns:
            Dictionary containing target positions, orientations, and joint angles
            in the agent's egocentric coordinate frame.
        """
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
    def _named_reward(name: str):
        """Decorator to register reward functions.

        Args:
            name: Name to register the reward function under.

        Returns:
            Decorator function that registers the reward function.
        """

        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("root_pos")
    def _root_pos_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        """Reward for matching root position to reference.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Reward weight multiplier.
            exp_scale: Exponential scaling factor for distance.

        Returns:
            Tuple of (reward_value, distance_to_target).
        """
        target = self._get_current_target(data, info)
        root_pos = self._get_root_pos(data)
        distance = jp.linalg.norm(target.root_position - root_pos)
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        return reward, distance

    @_named_reward("root_quat")
    def _root_quat_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        """Reward for matching root orientation to reference.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Reward weight multiplier.
            exp_scale: Exponential scaling factor for angular distance.

        Returns:
            Tuple of (reward_value, angular_distance_in_degrees).
        """
        target = self._get_current_target(data, info)
        root_quat = self._get_root_quat(data)
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        ang_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        ang_dist = jp.rad2deg(ang_dist)
        reward = weight * jp.exp(-((ang_dist / exp_scale) ** 2) / 2)
        return reward, ang_dist

    @_named_reward("joints")
    def _joints_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        """Reward for matching joint angles to reference.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Reward weight multiplier.
            exp_scale: Exponential scaling factor for joint space distance.

        Returns:
            Tuple of (reward_value, joint_space_l2_distance).
        """
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        distance = jp.linalg.norm(target.joints - joints)
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        return reward, distance

    @_named_reward("joints_vel")
    def _joint_vels_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        """Reward for matching joint velocities to reference.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Reward weight multiplier.
            exp_scale: Exponential scaling factor for velocity space distance.

        Returns:
            Tuple of (reward_value, joint_velocity_l2_distance).
        """
        target = self._get_current_target(data, info)
        joint_vels = self._get_joint_ang_vels(data)
        distance = jp.linalg.norm(target.joints_velocity - joint_vels)
        reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
        return reward, distance

    def _get_bodies_dist(
        self, data: mjx.Data, info: Mapping[str, Any], bodies: List[str] = consts.BODIES
    ) -> float:
        """Calculate distance between current and target body positions.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            bodies: List of body names to include in distance calculation.

        Returns:
            Total distance between current and target body positions.
        """
        target = self._get_current_target(data, info)
        body_pos = self._get_bodies_pos(data, flatten=False)
        total_dist_sqr = 0.0
        for body_name in bodies:
            dist_sqr = jp.sum((body_pos[body_name] - target.body_xpos(body_name)) ** 2)
            total_dist_sqr += dist_sqr
        return jp.sqrt(total_dist_sqr)

    @_named_reward("bodies_pos")
    def _body_pos_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        """Reward for matching body positions to reference.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Reward weight multiplier.
            exp_scale: Exponential scaling factor for distance.

        Returns:
            Tuple of (reward_value, total_body_distance).
        """
        total_dist = self._get_bodies_dist(data, info, bodies=self._config.bodies)
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        return reward, total_dist

    @_named_reward("end_eff")
    def _end_eff_reward(self, data, info, weight, exp_scale) -> Tuple[float, float]:
        """Reward for matching end effector positions to reference.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Reward weight multiplier.
            exp_scale: Exponential scaling factor for distance.

        Returns:
            Tuple of (reward_value, end_effector_distance).
        """
        total_dist = self._get_bodies_dist(
            data, info, bodies=self._config.end_effectors
        )
        reward = weight * jp.exp(-((total_dist / exp_scale) ** 2) / 2)
        return reward, total_dist

    @_named_reward("upright")
    def _upright_reward(
        self, data, info, weight, healthy_z_range
    ) -> Tuple[float, float]:
        """Reward for staying upright within a healthy height range.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Reward weight multiplier.
            healthy_z_range: Tuple of (min_height, max_height) for healthy range.

        Returns:
            Tuple of (reward_value, current_height).
        """
        torso_z = self._get_body_height(data)  # root_body(data).xpos[2]
        min_z, max_z = healthy_z_range
        in_range = jp.logical_and(torso_z >= min_z, torso_z <= max_z)
        return weight * in_range.astype(float), torso_z

    # Costs
    def _named_cost(name: str):
        """Decorator to register cost functions.

        Args:
            name: Name to register the cost function under.

        Returns:
            Decorator function that registers the cost function.
        """

        def decorator(cost_fcn: Callable):
            _COST_FCN_REGISTRY[name] = cost_fcn
            return cost_fcn

        return decorator

    @_named_cost("control")
    def _control_cost(self, data, info, weight) -> Tuple[float, float]:
        """Cost for control effort (action magnitude).

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Cost weight multiplier.

        Returns:
            Tuple of (cost_value, control_magnitude).
        """
        ctrl_magnitude = jp.sum(jp.square(info["action"]))
        return weight * ctrl_magnitude, ctrl_magnitude

    @_named_cost("control_diff")
    def _control_diff_cost(self, data, info, weight) -> Tuple[float, float]:
        """Cost for control smoothness (action differences).

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Cost weight multiplier.

        Returns:
            Tuple of (cost_value, control_difference).
        """
        ctrl_diff = jp.sum(jp.square(info["action"] - info["prev_action"]))
        return weight * ctrl_diff, ctrl_diff

    @_named_cost("energy")
    def _energy_cost(self, data, info, weight, max_value) -> Tuple[float, float]:
        """Cost for energy consumption.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Cost weight multiplier.
            max_value: Maximum energy value to clip at.

        Returns:
            Tuple of (cost_value, energy_consumption).
        """
        energy = jp.minimum(
            jp.sum(jp.abs(data.qvel) * jp.abs(data.qfrc_actuator)), max_value
        )
        return weight * energy, energy

    @_named_cost("jerk")
    def _jerk_cost(self, data, info, weight, window_len) -> Tuple[float, float]:
        """Cost for jerk (third derivative of position).

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            weight: Cost weight multiplier.
            window_len: Window length for jerk calculation.

        Returns:
            Tuple of (cost_value, jerk_magnitude).

        Raises:
            NotImplementedError: This cost function is not yet implemented.
        """
        raise NotImplementedError("jerk_cost is not implemented")

    # Termination
    def _named_termination_criterion(name: str):
        """Decorator to register termination criteria functions.

        Args:
            name: Name to register the termination criterion under.

        Returns:
            Decorator function that registers the termination function.
        """

        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("fall")
    def _fall(self, data, info, healthy_z_range) -> float:
        """Termination criterion for falling outside healthy height range.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            healthy_z_range: Tuple of (min_height, max_height) for healthy range.

        Returns:
            Boolean indicating if the agent has fallen.
        """
        torso_z = self._get_body_height(data)  # root_body(data).xpos[2]
        min_z, max_z = healthy_z_range
        return jp.logical_or(torso_z < min_z, torso_z > max_z)

    @_named_termination_criterion("root_too_far")
    def _root_too_far(self, data, info, max_distance) -> bool:
        """Termination criterion for root position deviation.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            max_distance: Maximum allowed distance from reference position.

        Returns:
            Boolean indicating if root is too far from reference.
        """
        target = self._get_current_target(data, info)
        root_pos = self._get_root_pos(data)
        distance = jp.linalg.norm(target.root_position - root_pos)
        return distance > max_distance

    @_named_termination_criterion("root_too_rotated")
    def _root_too_rotated(self, data, info, max_degrees) -> bool:
        """Termination criterion for root orientation deviation.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            max_degrees: Maximum allowed angular deviation in degrees.

        Returns:
            Boolean indicating if root is too rotated from reference.
        """
        target = self._get_current_target(data, info)
        root_quat = self._get_root_quat(data)
        quat_dist = 2.0 * jp.dot(root_quat, target.root_quaternion) ** 2 - 1.0
        ang_dist = 0.5 * jp.arccos(jp.minimum(1.0, quat_dist))
        return ang_dist > jp.deg2rad(max_degrees)

    @_named_termination_criterion("pose_error")
    def _bad_pose(self, data, info, max_l2_error) -> bool:
        """Termination criterion for joint pose error.

        Args:
            data: MuJoCo simulation data.
            info: Environment info dictionary.
            max_l2_error: Maximum allowed L2 error in joint space.

        Returns:
            Boolean indicating if pose error is too large.
        """
        target = self._get_current_target(data, info)
        joints = self._get_joint_angles(data)
        pose_error = jp.linalg.norm(target.joints - joints)
        return pose_error > max_l2_error

    # Properties for cleaner access
    @property
    def num_clips(self) -> int:
        """Get the number of available reference clips.

        Returns:
            Number of reference clips.
        """
        return self._num_clips()

    @property
    def clip_length(self) -> int:
        """Get the length of each clip.

        Returns:
            Number of frames per clip.
        """
        return self._clip_length()

    @property
    def mocap_dt(self) -> float:
        """Get the motion capture time step.

        Returns:
            Time step for motion capture data.
        """
        return self._mocap_dt

    @property
    def steps_for_cur_frame(self) -> int:
        """Get the number of steps per motion capture frame.

        Returns:
            Number of steps per motion capture frame.
        """
        return self._steps_for_cur_frame

    @property
    def n_steps(self) -> int:
        """Get the number of physics steps per control step .

        Returns:
            Number of physics steps per control step.
        """
        return self._n_steps

    @property
    def reference_length(self) -> int:
        """Get the length of reference windows for imitation.

        Returns:
            Number of reference frames used for imitation target.
        """
        return self._config.reference_length

    @property
    def start_frame_range(self) -> Tuple[int, int]:
        """Get the range of valid start frames.

        Returns:
            Tuple of (min_start_frame, max_start_frame).
        """
        return tuple(self._config.start_frame_range)

    @property
    def reward_terms(self) -> Dict[str, Any]:
        """Get the configured reward terms.

        Returns:
            Dictionary of reward term configurations.
        """
        return self._config.reward_terms

    @property
    def cost_terms(self) -> Dict[str, Any]:
        """Get the configured cost terms.

        Returns:
            Dictionary of cost term configurations.
        """
        return self._config.cost_terms

    @property
    def termination_criteria(self) -> Dict[str, Any]:
        """Get the configured termination criteria.

        Returns:
            Dictionary of termination criteria configurations.
        """
        return self._config.termination_criteria
    
    @property
    def reference_clips(self) -> ReferenceClips:
        """Get the reference clips.

        Returns:
            Reference clips.
        """
        return self._reference_clips
        

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        add_labels: bool = False,
        termination_extra_frames: int = 0,
    ) -> Sequence[np.ndarray]:
        """
        Renders a sequence of states (trajectory). The video includes the imitation
        target as a white transparent "ghost".

        Args:
            trajectory (List[mjx_env.State]): Sequence of environment states to render.
            height (int, optional): Height of the rendered frames in pixels. Defaults to 240.
            width (int, optional): Width of the rendered frames in pixels. Defaults to 320.
            camera (str, optional): Camera name or index to use for rendering.
            scene_option (mujoco.MjvOption, optional): Additional scene rendering options.
            modify_scene_fns (Sequence[Callable[[mujoco.MjvScene], None]], optional):
                Sequence of functions to modify the scene before rendering each frame.
                Defaults to None.
            add_labels (bool, optional): Whether to overlay clip and termination cause
                labels on frames. Defaults to False.
            termination_extra_frames (int, optional): If larger than 0, then repeat the
                frame triggering the termination this number of times. This gives
                a freeze-on-done effect that may help debug termination criteria.
                Additionally, a simple fade-out effect is applied during those frames
                to smooth the tranisition between clips. If this is larger than 0, the
                number of returned frames might be larger than `len(trajectory)`.
        Returns:
            Sequence[np.ndarray]: List of rendered frames as numpy arrays.
        """
        # Create a new spec with a ghost, without modifying the existing one
        spec, mj_model_with_ghost = self.add_ghost(
            rescale_factor=self._config.rescale_factor,
            dim=self._config.dim,
            pos=(0, 0, 0.05),
            ghost_rgba=(1.0, 1.0, 1.0, 0.2),
            suffix="-ghost",
            inplace=False,
        )

        mj_model_with_ghost.vis.global_.offwidth = width
        mj_model_with_ghost.vis.global_.offheight = height
        mj_data_with_ghost = mujoco.MjData(mj_model_with_ghost)

        renderer = mujoco.Renderer(mj_model_with_ghost, height=height, width=width)
        if camera is None:
            camera = -1

        rendered_frames = []
        for i, state in enumerate(trajectory):
            time_in_frames = state.data.time * self._config.mocap_hz
            frame = jp.floor(time_in_frames + state.info["start_frame"]).astype(int)
            clip = state.info["reference_clip"]
            ref = self.reference_clips.at(clip=clip, frame=frame)

            mj_data_with_ghost.qpos = jp.concatenate((state.data.qpos, ref.qpos))
            mj_data_with_ghost.qvel = jp.concatenate((state.data.qvel, ref.qvel))
            mujoco.mj_forward(mj_model_with_ghost, mj_data_with_ghost)
            try:
                renderer.update_scene(
                    mj_data_with_ghost,
                    camera=f"{camera}{self._suffix}",
                    scene_option=scene_option,
                )
            except ValueError as e:
                print(
                    f"Error updating scene for camera {f'{camera}{self._suffix}'}: {e}"
                )
                print(f"Available cameras: {[c.name for c in spec.cameras]}")
                raise e
            if modify_scene_fns is not None:
                modify_scene_fns[i](renderer.scene)
            rendered_frame = renderer.render()
            if add_labels:
                label = f"Clip {clip}"
                cv2.putText(
                    rendered_frame,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            rendered_frames.append(rendered_frame)
            if state.done:
                if add_labels:
                    reason = "<Unknown>"
                    if state.info["truncated"]:
                        reason = "truncated"
                    for name in self._config.termination_criteria.keys():
                        if state.metrics[name] > 0:
                            reason = name
                    cv2.putText(
                        rendered_frame,
                        reason,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                for t in range(termination_extra_frames):
                    rel_t = t / termination_extra_frames
                    fade_factor = 1 / (
                        1 + np.exp(10 * (rel_t - 0.5))
                    )  # Logistic fade-out
                    faded_frame = (rendered_frame * fade_factor).astype(np.uint8)
                    rendered_frames.append(faded_frame)
        return rendered_frames

    def verify_reference_data(self, atol: float = 5e-3) -> bool:
        """A set of non-exhaustive sanity checks that the reference data found in
        `config.REFERENCE_DATA_PATH` matches the environment's model. Most
        importantly, it verifies that the global coordinates of the
        body parts (xpos) match those the model produces when initialized to
        the corresponding qpos from the file. This can catch issues with nested
        free joints, mismatched joint orders, and incorrect scaling of the model
        but it's not exhaustive). This current implementation tests all
        frames of all clips in the reference data, and so is rather slow and
        does not have to be run every time.

        Args:
            atol (float): Absolute floating-point tolerance for the checks.
                          Defaults to 5e-3, because this seems to be the precision
                          of the reference data (TODO: why are there several mm
                          of error?).
        Returns:
            bool: True if all checks passed, False if any check failed.
        """

        def test_frame(clip_idx: int, frame: int) -> dict[str, bool]:
            data = self._reset_data(clip_idx, frame)
            reference = self.reference_clips.at(clip=clip_idx, frame=frame)
            checks = collections.OrderedDict()
            checks["root_pos"] = jp.allclose(
                self.root_body(data).xpos[..., : self._config.dim],
                reference.root_position[..., : self._config.dim],
                atol=atol,
            )
            checks["root_quat"] = jp.allclose(
                self.root_body(data).xquat, reference.root_quaternion, atol=atol
            )
            checks["joints"] = jp.allclose(
                self._get_joint_angles(data), reference.joints, atol=atol
            )
            body_pos = self._get_bodies_pos(data, flatten=False)
            for body_name, body_pos in body_pos.items():
                checks[f"body_xpos/{body_name}"] = jp.allclose(
                    body_pos[..., : self._config.dim],
                    reference.body_xpos(body_name)[..., : self._config.dim],
                    atol=atol,
                )
            if self._config.qvel_init == "reference":
                checks["joints_ang_vel"] = jp.allclose(
                    self._get_joint_ang_vels(data), reference.joints, atol=atol
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
            data = self._reset_data(clip, 0)
            reference = self.reference_clips.at(clip=clip, frame=0)
            test_result = test_clip(clip)

            for name, result in test_result.items():
                n_failed = jp.sum(np.logical_not(result))
                if n_failed > 0:
                    first_failed_frame = jp.argmax(np.logical_not(result))
                    warnings.warn(
                        f"Reference data verification failed for {n_failed} frames"
                        f" for check '{name}' for clip {clip}."
                        f" First failure at frame {first_failed_frame}."
                    )
                    if name == "root_pos":
                        warnings.warn(
                            f"Root position: {self.root_body(data).xpos} != {reference.root_position}"
                        )
                        warnings.warn(
                            f"diff: {jp.linalg.norm(self.root_body(data).xpos[..., : self._config.dim] - reference.root_position[..., : self._config.dim])}"
                        )
                    elif name == "root_quat":
                        warnings.warn(
                            f"Root quaternion: {self.root_body(data).xquat} != {reference.root_quaternion}"
                        )
                        warnings.warn(
                            f"diff: {jp.linalg.norm(self.root_body(data).xquat - reference.root_quaternion)}"
                        )
                    elif name == "joints":
                        warnings.warn(
                            f"Joints: {self._get_joint_angles(data)} != {reference.joints}"
                        )
                        warnings.warn(
                            f"diff: {jp.linalg.norm(self._get_joint_angles(data) - reference.joints)}"
                        )
                    elif name == "joints_ang_vel":
                        warnings.warn(
                            f"Joints ang vel: {self._get_joint_ang_vels(data)} != {reference.joints_velocity}"
                        )
                        warnings.warn(
                            f"diff: {jp.linalg.norm(self._get_joint_ang_vels(data) - reference.joints_velocity)}"
                        )
                    elif "body_xpos" in name:
                        body_name = name.split("/")[-1]
                        warnings.warn(
                            f"Bodies pos: {self._get_bodies_pos(data, flatten=False)[body_name][..., : self._config.dim]} != {reference.body_xpos(body_name)[..., : self._config.dim]}"
                        )
                        warnings.warn(
                            f"diff: {jp.linalg.norm(self._get_bodies_pos(data, flatten=False)[body_name][..., : self._config.dim] - reference.body_xpos(body_name)[..., : self._config.dim])}"
                        )
                    any_failed = True
        return not any_failed


def _assert_all_are_prefix(
    a: List[str], b: List[str], a_name: str = "a", b_name: str = "b"
) -> None:
    """Assert that all elements in list a are prefixes of corresponding elements in list b.

    Args:
        a: List of strings that should be prefixes.
        b: List of strings to check against.
        a_name: Name for list a in error messages.
        b_name: Name for list b in error messages.

    Raises:
        AssertionError: If lists have different lengths or elements don't match.
    """
    if isinstance(a, map):
        a = list(a)
    if isinstance(b, map):
        b = list(b)
    if len(a) != len(b):
        raise AssertionError(
            f"{a_name} has length {len(a)}, but {b_name} has length {len(b)}."
        )
    for a_el, b_el in zip(a, b):
        if not b_el.startswith(a_el):
            raise AssertionError(
                f"Comparing {a_name} and {b_name}. Expected {a_el} to match {b_el}."
            )
