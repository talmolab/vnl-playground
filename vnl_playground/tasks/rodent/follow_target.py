"""Follow-target task for virtual rodent.

The rodent must track a moving target that follows a smoothed random walk
trajectory. A new trajectory is generated each episode on reset.

Termination occurs if:
- Torso becomes too tilted (fallen)
- Torso goes below ground level
- NaN detected in simulation data
"""

import collections
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

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


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the FollowTarget environment."""
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="cg",
        mujoco_impl="jax",
        naconmax=16 * 8192,
        njmax=512,
        iterations=10,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        # Moving target trajectory config
        amp_x=0.1,
        amp_y=0.1,
        amp_z=0.2,
        z_min_offset=0.0,
        sigma_smooth=30.0,
        distance_margin=0.5,
        target_mode="generated",  # "generated" or "dataset"
        episode_length=500,
        action_repeat=1,
        reward_terms={
            "distance_to_target": {"weight": 1.0},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.02, "max_torso_angle": 80},
            "nan_termination": {},
        },
    )


def _smoothed_random_walk_jax(rng, T, amplitudes, sigma_smooth, z_min_offset):
    """Generate a smoothed random walk trajectory.

    Args:
        rng: JAX random key.
        T: Number of timesteps.
        amplitudes: (3,) array of amplitude scales for x, y, z.
        sigma_smooth: Gaussian smoothing sigma.
        z_min_offset: Minimum z offset (trajectory clamped to this).

    Returns:
        (T, 3) trajectory array starting at origin.
    """
    step_scale = amplitudes / jp.sqrt(T)
    raw_steps = jax.random.normal(rng, shape=(T, 3)) * step_scale
    cumsum = jp.cumsum(raw_steps, axis=0)

    radius = int(np.ceil(3 * sigma_smooth))
    x = jp.arange(-radius, radius + 1, dtype=jp.float32)
    kernel = jp.exp(-0.5 * (x / sigma_smooth) ** 2)
    kernel = kernel / kernel.sum()
    pad_width = len(kernel) // 2

    traj = jp.zeros((T, 3))
    for dim in range(3):  # unrolls at trace time
        padded = jp.pad(cumsum[:, dim], pad_width, mode="edge")
        smoothed = jp.convolve(padded, kernel, mode="valid")
        traj = traj.at[:, dim].set(smoothed[:T])

    # Center at origin and clamp
    traj = traj - traj[0]
    traj = jp.clip(traj, -amplitudes, amplitudes)
    traj = traj.at[:, 2].set(jp.maximum(traj[:, 2], z_min_offset))
    return traj


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


def _named_reward(name: str):
    """Decorator to register a reward function."""

    def decorator(reward_fcn: Callable):
        _REWARD_FCN_REGISTRY[name] = reward_fcn
        return reward_fcn

    return decorator


def _named_termination_criterion(name: str):
    """Decorator to register a termination criterion."""

    def decorator(termination_fcn: Callable):
        _TERMINATION_FCN_REGISTRY[name] = termination_fcn
        return termination_fcn

    return decorator


class FollowTarget(rodent_base.RodentEnv):
    """Follow-target environment.

    The rodent must track a moving target that follows a smoothed random walk
    trajectory. A new trajectory is generated each episode on reset.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        clips: Optional[Any] = None,
    ) -> None:
        del clips  # Follow-target task does not use reference clips
        super().__init__(config, config_overrides)

        # Initialize rodent at origin, standing pose
        init_x, init_y, init_z = 0.0, 0.0, 0.0
        init_quat = (1, 0, 0, 0)

        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
        )

        # Add translucent sphere visual at target position
        self._spec.worldbody.add_geom(
            name="target_sphere",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[0, 0, 0],
            size=[0.015],
            rgba=[1.0, 0.0, 0.0, 0.4],
            contype=0,
            conaffinity=0,
        )

        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

        # Dataset mode attributes (set via set_trajectory_dataset)
        self._trajectory_dataset = None
        self._initial_qpos_dataset = None
        self._num_trajectories = 0
        self._qpos_dataset = None

    def set_trajectory_dataset(self, trajectories, initial_qpos, qpos_dataset=None):
        """Set pre-extracted skull trajectories and initial poses for dataset mode.

        Args:
            trajectories: JAX array (N, T, 3) — absolute skull world positions.
            initial_qpos: JAX array (N, qpos_dim) — initial qpos for each clip.
            qpos_dataset: Optional JAX array (N, T, qpos_dim) — full qpos trajectories
                for ghost rendering. Only needed when render_imitation=True.
        """
        self._trajectory_dataset = trajectories
        self._initial_qpos_dataset = initial_qpos
        self._num_trajectories = trajectories.shape[0]
        self._qpos_dataset = qpos_dataset

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, traj_rng = jax.random.split(rng)

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "step_count": jp.array(0, dtype=jp.int32),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        if self._config.target_mode == "dataset":
            # Sample a random clip from the dataset
            idx = jax.random.randint(traj_rng, (), 0, self._num_trajectories)
            info["dataset_clip_idx"] = idx
            # Initialize rodent at the clip's starting pose
            data = data.replace(
                qpos=self._initial_qpos_dataset[idx],
                qvel=jp.zeros_like(data.qvel),
            )
            data = mjx.forward(self.mjx_model, data)
            # Use absolute skull trajectory directly (no offset needed)
            info["target_trajectory"] = self._trajectory_dataset[idx]
        else:  # "generated"
            # Compute forward kinematics to get body positions at reset
            data = mjx.forward(self.mjx_model, data)

            # Get skull position to anchor trajectory
            skull_pos = self._get_skull_pos(data)

            # Generate smoothed random walk trajectory
            episode_length = self._config.episode_length
            amplitudes = jp.array(
                [
                    self._config.amp_x,
                    self._config.amp_y,
                    self._config.amp_z,
                ]
            )
            traj = _smoothed_random_walk_jax(
                traj_rng,
                episode_length,
                amplitudes,
                self._config.sigma_smooth,
                self._config.z_min_offset,
            )
            info["target_trajectory"] = traj + skull_pos

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action

        # Increment step counter (clamp to episode_length - 1)
        episode_length = self._config.episode_length
        info["step_count"] = jp.minimum(info["step_count"] + 1, episode_length - 1)

        obs = self._get_obs(data, info)
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
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        # Compute egocentric target position from trajectory
        egocentric_target = self._get_egocentric_target(data, info)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
                egocentric_target,
            ]
        )

        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _get_skull_pos(self, data: mjx.Data) -> jax.Array:
        """Get skull world position (3D)."""
        skull_body = data.bind(self.mjx_model, self._spec.body(f"skull{self._suffix}"))
        return skull_body.xpos

    def _get_egocentric_target(self, data: mjx.Data, info: dict) -> jax.Array:
        """Get target position relative to the torso frame (3 values)."""
        step = info["step_count"]
        target_world = info["target_trajectory"][step]
        torso_body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        torso_pos = torso_body.xpos
        torso_frame = torso_body.xmat
        return jp.dot(target_world - torso_pos, torso_frame)

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics) -> bool:
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            terminated = termination_fcn(self, data, info, **kwargs)
            any_terminated = jp.logical_or(any_terminated, terminated)
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["proprioception"])[0])

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

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        render_ghost: bool = False,
    ) -> Sequence[np.ndarray]:
        """Renders a trajectory with the target sphere moving per frame.

        Args:
            trajectory: Sequence of environment states to render.
            height: Height of the rendered frames in pixels.
            width: Width of the rendered frames in pixels.
            camera: Camera name or index to use for rendering.
            scene_option: Additional scene rendering options.
            modify_scene_fns: Sequence of functions to modify the scene before
                rendering each frame.
            render_ghost: Whether to render a translucent ghost rodent showing
                the imitation qpos trajectory from the dataset. Requires
                set_trajectory_dataset to have been called with qpos_dataset.

        Returns:
            List of rendered frames as numpy arrays.
        """
        # Build ghost model if requested and qpos_dataset is available
        if render_ghost and self._qpos_dataset is not None:
            from vnl_playground.tasks.utils import dm_scale_spec, _recolour_tree

            spec = self._spec.copy()
            ghost_rodent = mujoco.MjSpec.from_file(self._walker_xml_path)
            rescale = self._config.rescale_factor
            if rescale != 1.0:
                ghost_rodent = dm_scale_spec(ghost_rodent, rescale)
            for body in ghost_rodent.worldbody.bodies:
                _recolour_tree(body, rgba=[1.0, 1.0, 1.0, 0.2])
            spawn_site = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
            spawn_body = spawn_site.attach_body(
                ghost_rodent.body("walker"), "", suffix="-ghost"
            )
            spawn_body.add_freejoint()
            mj_model = spec.compile()
        else:
            render_ghost = False  # disable if no qpos_dataset
            mj_model = self.mj_model

        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        mj_data = mujoco.MjData(mj_model)

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        if camera is None:
            camera = -1

        target_geom_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_sphere"
        )

        rendered_frames = []
        for i, state in enumerate(trajectory):
            step = int(state.info["step_count"])

            if render_ghost:
                clip_idx = int(state.info["dataset_clip_idx"])
                ghost_qpos = np.array(self._qpos_dataset[clip_idx, step])
                mj_data.qpos = np.concatenate([np.array(state.data.qpos), ghost_qpos])
                mj_data.qvel = np.concatenate(
                    [
                        np.array(state.data.qvel),
                        np.zeros_like(np.array(state.data.qvel)),
                    ]
                )
            else:
                mj_data.qpos = np.array(state.data.qpos)
                mj_data.qvel = np.array(state.data.qvel)

            mujoco.mj_forward(mj_model, mj_data)

            # Move target sphere to current trajectory position
            target_pos = np.array(state.info["target_trajectory"][step])
            mj_data.geom_xpos[target_geom_id] = target_pos

            renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
            if modify_scene_fns is not None:
                modify_scene_fns[i](renderer.scene)
            rendered_frames.append(renderer.render())

        return rendered_frames


# --- Reward Functions ---


@_named_reward("distance_to_target")
def _distance_to_target_reward(env, data, info, metrics, weight) -> float:
    """Dense reward: decreases as skull approaches the moving target."""
    skull_pos = env._get_skull_pos(data)
    step = info["step_count"]
    target = info["target_trajectory"][step]
    distance = jp.sqrt(jp.sum((skull_pos - target) ** 2))

    margin = env._config.distance_margin
    reward_value = reward_fns.tolerance(
        distance,
        bounds=(0, 0),
        margin=margin,
        sigmoid="linear",
        value_at_margin=0.0,
    )

    weighted_reward = reward_value * weight
    metrics["rewards/distance_to_target"] = weighted_reward
    metrics["skull_target_distance"] = distance
    return weighted_reward


# --- Termination Functions ---


@_named_termination_criterion("fallen")
def _fallen_termination(
    env, data: mjx.Data, info, min_torso_z: float, max_torso_angle: float
) -> bool:
    """Check if rodent has fallen."""
    del info
    torso_body = data.bind(env.mjx_model, env._spec.body(f"torso{env._suffix}"))
    torso_z = torso_body.xpos[2]
    below_ground = torso_z < min_torso_z

    upright_z = torso_body.xmat.reshape(3, 3)[2, 2]
    max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
    too_tilted = upright_z < max_cos_angle

    return jp.logical_or(below_ground, too_tilted)


@_named_termination_criterion("nan_termination")
def _nan_termination(env, data, info) -> bool:
    """Check for NaN values in simulation data."""
    del info
    flattened_vals, _ = flatten_util.ravel_pytree(data)
    num_nans = jp.sum(jp.isnan(flattened_vals))
    return num_nans > 0
