"""Walk-and-rear task for virtual rodent.

The rodent must walk to a target (x, y) location and rear up to a target z
height, then hold that position for a configurable duration. On success the
agent receives a bonus reward and the episode terminates.

Termination occurs if:
- Torso becomes too tilted (fallen)
- Torso goes below ground level
- NaN detected in simulation data
- Walk-rear goal is achieved (hold completed)
"""

import collections
from typing import Any, Callable, Dict, Mapping, Optional, Union

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
    """Returns the default configuration for the WalkRear environment."""
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
        # Walk-rear specific config
        target_x=0.5,
        target_y=0.0,
        target_z=0.10,
        xy_tolerance=0.05,
        z_tolerance=0.02,
        hold_duration=1.0,  # Seconds to hold at target for success
        distance_margin=0.5,  # Margin for distance reward shaping
        episode_length=2000,
        action_repeat=1,
        reward_terms={
            "distance_to_target": {"weight": 1.0},
            "walk_rear_success": {"weight": 500.0},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.02, "max_torso_angle": 80},
            "nan_termination": {},
            "walk_rear_complete": {},
        },
    )


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


class WalkRear(rodent_base.RodentEnv):
    """Walk-and-rear environment.

    The rodent must walk to a target (x, y) location and rear up to a target
    z height, then hold that position for a configurable duration.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        clips: Optional[Any] = None,
    ) -> None:
        del clips  # Walk-rear task does not use reference clips
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

        # Add translucent box visual at target position
        target_x = self._config.target_x
        target_y = self._config.target_y
        target_z = self._config.target_z
        xy_tol = self._config.xy_tolerance
        z_tol = self._config.z_tolerance

        self._spec.worldbody.add_geom(
            name="target_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[target_x, target_y, target_z],
            size=[xy_tol, xy_tol, z_tol],
            rgba=[1.0, 0.0, 0.0, 0.3],
            contype=0,
            conaffinity=0,
        )

        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "hold_steps": jp.array(0),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        # Compute forward kinematics to get body positions at reset
        data = mjx.forward(self.mjx_model, data)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        obs = self._get_obs(data, info)

        info["prev_action"] = info["action"]
        info["action"] = action

        # Check if skull is within tolerance of target
        skull_pos = self._get_skull_pos(data)
        target_x = self._config.target_x
        target_y = self._config.target_y
        target_z = self._config.target_z

        xy_dist = jp.sqrt(
            (skull_pos[0] - target_x) ** 2 + (skull_pos[1] - target_y) ** 2
        )
        z_in_range = jp.abs(skull_pos[2] - target_z) <= self._config.z_tolerance
        xy_in_range = xy_dist <= self._config.xy_tolerance
        at_target = jp.logical_and(xy_in_range, z_in_range)

        # Update hold counter: increment if at target, reset otherwise
        info["hold_steps"] = jp.where(
            at_target,
            info["hold_steps"] + 1,
            jp.array(0),
        )

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

        # Compute egocentric target position
        egocentric_target = self._get_egocentric_target(data)

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

    def _get_egocentric_target(self, data: mjx.Data) -> jax.Array:
        """Get target position relative to the torso frame (3 values)."""
        target_world = jp.array(
            [self._config.target_x, self._config.target_y, self._config.target_z]
        )
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


# --- Reward Functions ---


@_named_reward("distance_to_target")
def _distance_to_target_reward(env, data, info, metrics, weight) -> float:
    """Dense reward: decreases as skull approaches target position."""
    skull_pos = env._get_skull_pos(data)
    target = jp.array(
        [env._config.target_x, env._config.target_y, env._config.target_z]
    )
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


@_named_reward("walk_rear_success")
def _walk_rear_success_reward(env, data, info, metrics, weight) -> float:
    """Large reward when target pose is held for the required duration."""
    required_steps = int(env._config.hold_duration / env._config.ctrl_dt)
    hold_steps = info["hold_steps"]

    # Give reward only on the exact step when threshold is reached
    success = hold_steps == required_steps
    weighted_reward = jp.astype(success, float) * weight
    metrics["rewards/walk_rear_success"] = weighted_reward
    metrics["hold_steps"] = hold_steps
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


@_named_termination_criterion("walk_rear_complete")
def _walk_rear_complete_termination(env, data, info) -> bool:
    """Terminate episode when target pose has been held for required duration."""
    del data
    required_steps = int(env._config.hold_duration / env._config.ctrl_dt)
    return info["hold_steps"] >= required_steps
