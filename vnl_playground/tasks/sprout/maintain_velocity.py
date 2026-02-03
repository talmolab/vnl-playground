"""Maintain velocity task for Fauna Robotics Sprout humanoid.

The Sprout is initialized in standing pose facing forward (+x axis)
and must maintain a target forward velocity while walking with a
realistic bipedal gait.

Rewards encourage:
- Forward velocity tracking at the target speed
- Upright torso at standing height (discourages knee-walking)
- Phase-synchronized alternating foot contacts (walking gait)
- Smooth, energy-efficient actions

Termination occurs if:
- Torso becomes too tilted (fallen)
- Torso goes below a minimum height
- NaN detected in simulation data
"""

import collections
from typing import Any, Callable, Dict, Mapping, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_fns

from vnl_playground.tasks.sprout import base as sprout_base
from vnl_playground.tasks.sprout import consts


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the MaintainVelocity environment.

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        walker_xml_path=consts.SPROUT_XML_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="newton",
        mujoco_impl="jax",
        naconmax=16 * 512,
        njmax=400,
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        ccd_iterations=75,
        torque_actuators=True,
        target_speed=0.8,
        episode_length=2000,
        action_repeat=1,
        # Gait parameters
        gait_freq_range=[1.0, 1.5],
        max_foot_height=0.06,
        base_height_target=0.55,
        contact_threshold=0.015,
        reward_terms={
            "forward_velocity": {"weight": 1.5},
            "lateral_velocity": {"weight": 0.3},
            "angular_velocity_z": {"weight": 0.2},
            "upright": {"weight": 1.0},
            "torso_height": {"weight": 1.0},
            "energy_penalty": {"weight": 0.0},
            "action_rate": {"weight": 0.005},
            "feet_phase": {"weight": 1.0},
            "feet_air_time": {"weight": 0.5},
            "alive": {"weight": 0.2},
        },
        termination_criteria={
            "fallen": {"min_torso_z": 0.45, "max_torso_angle": 45},
            "nan_termination": {},
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


class MaintainVelocity(sprout_base.SproutEnv):
    """Maintain velocity environment with realistic walking gait for Sprout.

    The robot must maintain a target forward velocity in the +x direction
    using a proper bipedal walking gait with alternating foot contacts.
    Phase-based gait tracking encourages realistic foot swing/stance patterns,
    while torso height and upright rewards discourage knee-walking.
    """

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize the MaintainVelocity environment.

        Args:
            rng: Random number generator key.
            config: Configuration dictionary.
            config_overrides: Optional configuration overrides.
        """
        super().__init__(config, config_overrides)
        self._rng = rng

        # Initialize Sprout at spawn height facing forward (+x direction)
        init_x, init_y = 0.0, 0.0
        init_z = consts.SPAWN_HEIGHT
        init_quat = (1, 0, 0, 0)

        self.add_sprout(
            torque_actuators=self._config.torque_actuators,
            pos=[init_x, init_y, init_z],
            quat=init_quat,
        )
        self._spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        self.compile()

        # Set initial standing pose via keyframe
        self._init_qpos = self._compute_init_qpos()
        self._post_init()

    def _post_init(self) -> None:
        """Cache body IDs and default pose for efficient gait tracking."""
        # Sole reference bodies for foot height tracking.
        # These bodies sit at the bottom of each foot (z=-0.093 relative to
        # foot_link), so their world z is ~0 when the foot is on the ground.
        self._sole_body_ids = np.array([
            self._mj_model.body(f"ref_left_sole_link{self._suffix}").id,
            self._mj_model.body(f"ref_right_sole_link{self._suffix}").id,
        ])

        # Default joint positions for pose regularization
        self._default_joint_qpos = self._init_qpos[7:]

    def _compute_init_qpos(self) -> jp.ndarray:
        """Compute initial qpos with standing pose.

        Returns qpos array with freejoint (7) + joint positions (26).
        """
        qpos = np.zeros(self._mj_model.nq)
        # Freejoint: [x, y, z, qw, qx, qy, qz]
        qpos[2] = consts.SPAWN_HEIGHT  # z position
        qpos[3] = 1.0  # qw = 1 (identity quaternion)

        # Set standing pose joint angles
        for i, joint in enumerate(self._spec.joints):
            if joint.name == "root":
                continue
            joint_name = joint.name
            # Strip suffix if present
            base_name = joint_name.replace(self._suffix, "")
            if base_name in consts.STANDING_POSE:
                # Joint index in qpos: 7 (freejoint) + joint_index
                qpos_idx = self._mj_model.jnt_qposadr[
                    self._mj_model.joint(joint_name).id
                ]
                qpos[qpos_idx] = consts.STANDING_POSE[base_name]

        return jp.array(qpos)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment state with gait phase initialization.

        Args:
            rng: Random number generator state.

        Returns:
            mjx_env.State: The initial environment state after reset.
        """
        # Sample gait frequency for this episode
        rng, freq_key = jax.random.split(rng)
        gait_freq = jax.random.uniform(
            freq_key,
            (1,),
            minval=self._config.gait_freq_range[0],
            maxval=self._config.gait_freq_range[1],
        )
        # Phase increment per control step: 2*pi*dt*freq
        phase_dt = 2 * jp.pi * self._config.ctrl_dt * gait_freq
        # Anti-phase for bipedal gait: left leg at 0, right leg at pi
        phase = jp.array([0.0, jp.pi])

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            # Gait phase tracking
            "phase": phase,
            "phase_dt": phase_dt,
            # Foot contact tracking
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "first_contact": jp.zeros(2, dtype=bool),
            "contact": jp.zeros(2, dtype=bool),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        # Apply standing pose and run forward kinematics so that
        # derived quantities (xpos, xmat, sensordata, etc.) are populated
        # before computing observations, rewards, and termination.
        data = data.replace(qpos=self._init_qpos)
        data = mjx.forward(self.mjx_model, data)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward with gait tracking.

        Args:
            state: Current environment state.
            action: Action to apply.

        Returns:
            mjx_env.State: The new environment state after stepping.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info

        # --- Foot contact detection via sole reference body z-position ---
        sole_z = data.xpos[self._sole_body_ids, 2]
        contact = sole_z < self._config.contact_threshold
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt

        # Track air time and swing peak (before clearing)
        info["feet_air_time"] += self._config.ctrl_dt
        info["swing_peak"] = jp.maximum(info["swing_peak"], sole_z)

        # Store contact state for reward computation
        info["contact"] = contact
        info["first_contact"] = first_contact

        # Update actions
        info["prev_action"] = info["action"]
        info["action"] = action

        # Compute obs, termination, and reward
        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        # --- Advance gait phase ---
        phase_tp1 = info["phase"] + info["phase_dt"]
        info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

        # --- Clear air time and swing peak for feet now in contact ---
        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> collections.OrderedDict:
        """Get the current observation including gait phase.

        Args:
            data: The simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with task_obs and proprioception keys.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        origin = self._get_origin(data)

        # Gait phase observation: cos/sin for each leg (4 values total)
        phase_obs = jp.concatenate([
            jp.cos(info["phase"]),
            jp.sin(info["phase"]),
        ])

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                origin,
                phase_obs,
            ]
        )

        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _is_done(
        self, data: mjx.Data, info: Mapping[str, Any], metrics
    ) -> bool:
        """Check if episode should terminate."""
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
        """Compute total reward."""
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    # ---- Reward functions ----

    def _named_reward(name: str):
        """Decorator to register reward functions."""

        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("forward_velocity")
    def _forward_velocity_reward(self, data, info, metrics, weight) -> float:
        """Reward for maintaining target forward velocity in +x direction.

        Uses the framelinvel sensor (world-frame linear velocity) instead of
        subtree_linvel, which is only computed by MJX when a subtreelinvel
        sensor is defined in the MJCF.
        """
        del info

        # World-frame linear velocity from framelinvel sensor
        linvel = data.bind(
            self.mjx_model,
            self._spec.sensor(f"torso_link_site_linvel{self._suffix}"),
        ).sensordata
        forward_vel = linvel[0]  # x-component = forward

        target_speed = self._config.target_speed

        reward_value = reward_fns.tolerance(
            forward_vel,
            bounds=(target_speed, target_speed),
            margin=target_speed,
            sigmoid="linear",
            value_at_margin=0.0,
        )

        weighted_reward = reward_value * weight
        metrics["rewards/forward_velocity"] = weighted_reward
        return weighted_reward

    @_named_reward("lateral_velocity")
    def _lateral_velocity_cost(self, data, info, metrics, weight) -> float:
        """Penalty for lateral (y-direction) velocity."""
        del info
        linvel = data.bind(
            self.mjx_model,
            self._spec.sensor(f"torso_link_site_linvel{self._suffix}"),
        ).sensordata
        lateral_vel = linvel[1]  # y-component = lateral
        cost = -weight * jp.square(lateral_vel)
        metrics["rewards/lateral_velocity"] = cost
        return cost

    @_named_reward("angular_velocity_z")
    def _angular_velocity_z_cost(self, data, info, metrics, weight) -> float:
        """Penalty for yaw rate (z-axis angular velocity) to prevent turning."""
        del info
        angvel = data.bind(
            self.mjx_model,
            self._spec.sensor(f"torso_link_site_angvel{self._suffix}"),
        ).sensordata
        yaw_rate = angvel[2]
        cost = -weight * jp.square(yaw_rate)
        metrics["rewards/angular_velocity_z"] = cost
        return cost

    @_named_reward("upright")
    def _upright_reward(self, data, info, metrics, weight) -> float:
        """Reward for keeping torso upright (z-axis aligned with world up).

        Tighter bounds than before to strongly discourage forward lean.
        """
        del info
        torso_body = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        # xmat[2,2] = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        reward_value = reward_fns.tolerance(
            upright_z,
            bounds=(0.95, 1.0),
            margin=0.3,
            sigmoid="linear",
            value_at_margin=0.0,
        )
        weighted_reward = reward_value * weight
        metrics["rewards/upright"] = weighted_reward
        return weighted_reward

    @_named_reward("torso_height")
    def _torso_height_reward(self, data, info, metrics, weight) -> float:
        """Reward for maintaining target torso height.

        This is the key reward for discouraging knee-walking. The torso must
        stay near standing height (~0.55m for Sprout). Uses a tolerance band
        to allow natural vertical bobbing during walking while penalizing
        crouching or kneeling postures.
        """
        del info
        torso_body = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        torso_z = torso_body.xpos[2]
        target = self._config.base_height_target

        reward_value = reward_fns.tolerance(
            torso_z,
            bounds=(target - 0.05, target + 0.10),
            margin=0.15,
            sigmoid="linear",
            value_at_margin=0.0,
        )

        weighted_reward = reward_value * weight
        metrics["rewards/torso_height"] = weighted_reward
        return weighted_reward

    @_named_reward("energy_penalty")
    def _energy_penalty(self, data, info, metrics, weight) -> float:
        """Penalty for high actuator forces (energy efficiency)."""
        del info
        ctrl = data.ctrl
        cost = -weight * jp.sum(jp.square(ctrl))
        metrics["rewards/energy_penalty"] = cost
        return cost

    @_named_reward("action_rate")
    def _action_rate_cost(self, data, info, metrics, weight) -> float:
        """Penalty for jerky actions (encourages smooth motor commands).

        Penalizes the squared difference between consecutive actions.
        """
        del data
        action_diff = info["action"] - info["prev_action"]
        cost = -weight * jp.sum(jp.square(action_diff))
        metrics["rewards/action_rate"] = cost
        return cost

    @_named_reward("feet_phase")
    def _feet_phase_reward(self, data, info, metrics, weight) -> float:
        """Reward for foot height tracking the desired gait phase.

        Uses a phase-based reference trajectory from gait.get_rz() to define
        target foot heights. Each foot alternates between stance (foot on
        ground, phase < 0) and swing (foot lifted, phase > 0) in anti-phase.

        The reference trajectory uses cubic Bezier interpolation to create
        smooth swing/stance transitions at the configured max_foot_height.

        Adapted from the Berkeley Humanoid joystick task.
        """
        # Get sole z-positions (world frame, ~0 when on ground)
        sole_z = data.xpos[self._sole_body_ids, 2]

        # Desired foot heights from phase-based gait reference
        rz = gait.get_rz(
            info["phase"],
            swing_height=self._config.max_foot_height,
        )

        error = jp.sum(jp.square(sole_z - rz))
        reward_value = jp.exp(-error / 0.01)

        weighted_reward = reward_value * weight
        metrics["rewards/feet_phase"] = weighted_reward
        return weighted_reward

    @_named_reward("feet_air_time")
    def _feet_air_time_reward(self, data, info, metrics, weight) -> float:
        """Reward for proper alternating foot contact timing.

        Encourages each foot to spend a minimum time in the air before
        landing. Only gives reward at the moment of first contact after
        a swing phase, and clips at a maximum air time to prevent the
        robot from just holding feet in the air.

        Adapted from the Berkeley Humanoid joystick task.
        """
        del data
        first_contact = info["first_contact"]
        air_time = info["feet_air_time"]

        threshold_min = 0.15  # Minimum air time for reward (seconds)
        threshold_max = 0.4   # Maximum rewarded air time (seconds)

        clipped_air_time = (air_time - threshold_min) * first_contact
        clipped_air_time = jp.clip(clipped_air_time, max=threshold_max - threshold_min)
        reward_value = jp.sum(clipped_air_time)

        weighted_reward = reward_value * weight
        metrics["rewards/feet_air_time"] = weighted_reward
        return weighted_reward

    @_named_reward("alive")
    def _alive_reward(self, data, info, metrics, weight) -> float:
        """Small constant reward for staying alive (not terminated)."""
        del data, info
        reward_value = weight
        metrics["rewards/alive"] = reward_value
        return reward_value

    # ---- Termination criteria ----

    def _named_termination_criterion(name: str):
        """Decorator to register termination functions."""

        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("fallen")
    def _fallen_termination(
        self,
        data: mjx.Data,
        info,
        min_torso_z: float = 0.45,
        max_torso_angle: float = 45,
    ) -> bool:
        """Check if robot has fallen.

        Args:
            data: Simulation data.
            info: State info (unused).
            min_torso_z: Minimum z height threshold for torso.
            max_torso_angle: Maximum angle from vertical in degrees.

        Returns:
            Boolean indicating if fallen.
        """
        del info

        torso_body = data.bind(
            self.mjx_model, self._spec.body(f"torso_link{self._suffix}")
        )
        torso_z = torso_body.xpos[2]

        below_threshold = torso_z < min_torso_z

        # xmat[-1, -1] is element (2,2) = cos(angle from vertical)
        upright_z = torso_body.xmat[-1, -1]
        max_cos_angle = np.cos(np.deg2rad(max_torso_angle))
        too_tilted = upright_z < max_cos_angle

        return jp.logical_or(below_threshold, too_tilted)

    @_named_termination_criterion("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        """Check for NaN values in simulation data."""
        del info
        flattened_vals, _ = flatten_util.ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        return num_nans > 0

    def null_action(self) -> jp.ndarray:
        """Return zero action."""
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
        return jax.tree_util.tree_map(
            lambda x: jp.prod(jp.array(x.shape)), obs
        )
