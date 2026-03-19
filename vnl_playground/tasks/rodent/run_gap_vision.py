"""Vision-enabled run through corridor with gaps task.

Extends RunGap with egocentric camera observations rendered via the
JAX-callable mujoco_warp GPU ray-tracer.  Supports both monocular
(single camera) and binocular (stereo left/right eye) modes via the
``config.binocular`` flag.

The observation dict from _get_obs() returns::

    {
        "state": OrderedDict(
            task_obs=...,                       # body-state signals (flat)
            proprioception=OrderedDict(...),     # nested dict
            vision=jp.zeros(H, W, C),           # placeholder zeros
        ),
        "privileged_state": OrderedDict(...),   # same as state
    }

In monocular mode the vision placeholder has shape (H, W, C) where C is 1
(grayscale) or 3 (RGB).  In binocular mode the shape is (H, W, 2*C) — left
and right eye images concatenated along the channel dimension.

``task_obs`` contains body-state signals (prev_action, kinematic sensors,
touch sensors, origin) rather than hand-crafted gap features — gap
information should be inferred from vision.

Vision rendering happens in ``VisionRenderWrapper`` (monocular) or
``BinocularVisionRenderWrapper`` (binocular), which wrap the vmapped/batched
env and render on all worlds at once using the JAX-callable warp renderer.
The wrapper replaces the zero placeholders with real rendered images.

Compatible with ``HighLevelWrapper`` for transfer learning pipelines
and direct ff_ppo training.  Downstream ``observation_utils.flatten_obs_dict()``
maps ``task_obs`` to ``imitation_target`` internally.

Usage::

    env = RunGapVision(config=cfg)
    brax_env = wrap_for_brax_training(env, ...)
    vision_env = VisionRenderWrapper(brax_env, env.mj_model, nworld=num_envs,
                                      **vision_config)
"""

import collections
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from jax import flatten_util
from scipy.spatial.transform import Rotation as ScipyRotation
from ml_collections import config_dict

from vnl_playground.tasks.rodent import run_gap


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the RunGapVision environment."""
    cfg = run_gap.default_config()
    cfg.mujoco_impl = "warp"  # Required for mujoco_warp rendering
    # Vision parameters
    cfg.vision = True
    cfg.vision_width = 32
    cfg.vision_height = 32
    cfg.grayscale = True
    cfg.vision_camera_name = "egocentric-rodent"
    cfg.render_depth = False
    cfg.use_textures = False
    cfg.use_shadows = False
    # Binocular parameters (disabled by default = monocular mode)
    cfg.binocular = False
    cfg.left_camera_name = "eye_left-rodent"
    cfg.right_camera_name = "eye_right-rodent"
    # Stochastic eye dropout (for binocular training fairness)
    cfg.eye_dropout_rate = 0.0  # probability of masking one eye per step (0=disabled)
    cfg.eval_eye_mode = "binocular"  # eval mode: "binocular", "left_only", "right_only"
    # Eye camera yaw offset from center (radians). Controls binocular overlap:
    #   overlap_deg ≈ fovy - 2 * degrees(eye_angle_offset)  [square aspect ratio]
    # Default 0.2 rad ≈ 11.5° offset → ~57° overlap (matches original XML).
    cfg.eye_angle_offset = 0.2
    # Actuable eye parameters (requires binocular=True)
    cfg.actuable_eyes = False
    cfg.eye_yaw_range = 0.698  # ±40° in radians
    cfg.eye_force_range = 0.01
    cfg.eye_damping = 0.001
    cfg.eye_stiffness = 0.0
    return cfg


class RunGapVision(run_gap.RunGap):
    """RunGap with egocentric vision observations (monocular or binocular).

    Observations are returned with state/privileged_state wrapping containing
    task_obs, proprioception, and vision keys. The vision key contains
    placeholder zeros -- real rendering is handled by ``VisionRenderWrapper``
    (monocular) or ``BinocularVisionRenderWrapper`` (binocular) which wraps
    the batched env.

    When ``config.binocular`` is False (default), vision shape is (H, W, C).
    When ``config.binocular`` is True, vision shape is (H, W, 2*C) for stereo
    left/right eye images concatenated along the channel dimension.

    ``task_obs`` provides body-state signals (prev_action, kinematic sensors,
    touch sensors, origin). Gap information is not included -- the agent should
    infer it from vision.

    Compatible with both track-mjx's ff_ppo observation_utils and
    ``HighLevelWrapper`` for transfer learning pipelines.
    """

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(rng=rng, config=config, config_overrides=config_overrides)

        if self._config.mujoco_impl != "warp":
            raise ValueError("RunGapVision requires mujoco_impl='warp' for rendering")

        self._vision_enabled = self._config.vision
        self._vision_width = self._config.vision_width
        self._vision_height = self._config.vision_height
        self._grayscale = self._config.get("grayscale", False)

        self._n_eye_actuators = 0

        # Apply configurable eye camera angles when in binocular mode.
        # This modifies the spec and triggers a second compile — the first
        # compile happens in RunGap.__init__() before we get control here.
        # The double compilation is a one-time startup cost.
        if self._config.get("binocular", False):
            if self._config.get("actuable_eyes", False):
                # Record pre-eye model dimensions for proprioception masking
                self._pre_eye_nu = self._mj_model.nu
                self._pre_eye_nq = self._mj_model.nq
                self._pre_eye_nv = self._mj_model.nv
                self._add_eye_actuators()
                # Update camera names to point at actuated cameras
                suffix = self._suffix
                self._config.left_camera_name = f"eye_left_actuated{suffix}"
                self._config.right_camera_name = f"eye_right_actuated{suffix}"
            else:
                offset = self._config.get("eye_angle_offset", 0.2)
                self._configure_eye_cameras(offset)

    def _configure_eye_cameras(self, eye_angle_offset: float) -> None:
        """Modify eye camera yaw angles on the spec and recompile.

        Yaws each eye camera outward by ±eye_angle_offset radians around the
        skull's dorsal (Z) axis.  Uses quaternion composition to avoid the
        gimbal-lock / roll-vs-yaw confusion that arises when tweaking a single
        euler component.

        The resulting binocular overlap (for square images) is approximately:
            overlap_deg ≈ fovy - 2 × degrees(eye_angle_offset)

        Args:
            eye_angle_offset: Yaw offset in radians for each eye from center.
                0.0 = both eyes look straight ahead (max overlap = fovy).
                0.2 ≈ 11.5° offset → ~57° overlap with fovy=80°.
                0.698 ≈ 40° offset → 0° overlap.

        Raises:
            ValueError: If eye_angle_offset is outside [0, π/2].
        """
        if eye_angle_offset < 0 or eye_angle_offset > np.pi / 2:
            raise ValueError(
                f"eye_angle_offset must be in [0, π/2], got {eye_angle_offset}"
            )

        import mujoco as mj

        # Base egocentric orientation: forward-looking camera in skull frame.
        # euler [0, -π/2, -π/2] => look along skull +X, up along skull -Z.
        R_ego = ScipyRotation.from_euler("XYZ", [0, -np.pi / 2, -np.pi / 2])

        suffix = self._suffix
        for cam in self._spec.cameras:
            if cam.name == f"eye_left{suffix}":
                yaw = +eye_angle_offset
            elif cam.name == f"eye_right{suffix}":
                yaw = -eye_angle_offset
            else:
                continue

            # Compose: yaw around skull Z, then base camera orientation
            R_yaw = ScipyRotation.from_rotvec([0, 0, yaw])
            R_combined = R_yaw * R_ego
            quat_xyzw = R_combined.as_quat()
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]  # MuJoCo uses (w, x, y, z)

            cam.quat = quat_wxyz
            cam.alt.type = mj.mjtOrientation.mjORIENTATION_QUAT

        # Recompile with updated camera orientations
        self.compile(forced=True)

    @property
    def n_eye_actuators(self) -> int:
        """Number of eye actuators (2 if actuable_eyes, 0 otherwise)."""
        return self._n_eye_actuators

    def _add_eye_actuators(self) -> None:
        """Add eye mount bodies with hinge joints, cameras, and actuators.

        For each eye (left, right):
        - Adds a child body on the skull at the camera position
        - Adds a hinge joint for yaw rotation around the skull Z axis
        - Adds a camera with egocentric orientation
        - Adds a torque actuator to drive the joint

        The convention is that positive control drives both eyes outward
        (divergence): left axis=[0,0,1], right axis=[0,0,-1].

        After adding all elements, recompiles the model and computes
        index arrays for proprioception masking.
        """
        import mujoco as mj

        suffix = self._suffix
        skull = self._spec.body(f"skull{suffix}")
        cfg = self._config

        eye_specs = [
            ("left", [0.015, 0.004, 0.01], [0, 0, 1]),
            ("right", [0.015, -0.004, 0.01], [0, 0, -1]),
        ]

        for side, pos, axis in eye_specs:
            # Mount body (needs explicit inertial since it has no geoms)
            mount = skull.add_body(
                name=f"eye_{side}_mount{suffix}",
                pos=pos,
            )
            mount.mass = 1e-6
            mount.inertia = [1e-9, 1e-9, 1e-9]
            mount.ipos = [0, 0, 0]
            mount.explicitinertial = True
            # Hinge joint
            mount.add_joint(
                name=f"eye_{side}_yaw{suffix}",
                type=mj.mjtJoint.mjJNT_HINGE,
                axis=axis,
                limited=1,
                range=[-cfg.eye_yaw_range, cfg.eye_yaw_range],
                damping=cfg.eye_damping,
                stiffness=cfg.eye_stiffness,
            )
            # Camera on the mount body
            mount.add_camera(
                name=f"eye_{side}_actuated{suffix}",
                euler=[0, -np.pi / 2, -np.pi / 2],
                fovy=80,
            )
            # Torque actuator
            gainprm = [0.0] * 10
            gainprm[0] = 1.0
            self._spec.add_actuator(
                name=f"eye_{side}_yaw{suffix}",
                trntype=mj.mjtTrn.mjTRN_JOINT,
                target=f"eye_{side}_yaw{suffix}",
                gaintype=mj.mjtGain.mjGAIN_FIXED,
                gainprm=gainprm,
                biastype=mj.mjtBias.mjBIAS_NONE,
                forcelimited=1,
                forcerange=[-cfg.eye_force_range, cfg.eye_force_range],
                ctrllimited=1,
                ctrlrange=[-1.0, 1.0],
            )

        self._n_eye_actuators = 2
        self.compile(forced=True)
        self._compute_eye_indices()

    def _compute_eye_indices(self) -> None:
        """Pre-compute numpy index arrays for proprioception masking.

        After eye joints/actuators are added, the model's qpos and qvel
        arrays grow. We need to identify which indices correspond to the
        eye joints vs. body joints so that proprioception can exclude
        eye joint data (preserving the frozen decoder input size).
        """
        import mujoco as mj

        m = self._mj_model
        suffix = self._suffix

        # Find eye joint IDs
        eye_joint_ids = []
        for side in ["left", "right"]:
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"eye_{side}_yaw{suffix}")
            assert jid >= 0, f"Eye joint eye_{side}_yaw{suffix} not found"
            eye_joint_ids.append(jid)

        # Eye joint qpos and qvel/dof addresses
        eye_qpos_addrs = set()
        eye_dof_addrs = set()
        for jid in eye_joint_ids:
            eye_qpos_addrs.add(int(m.jnt_qposadr[jid]))
            eye_dof_addrs.add(int(m.jnt_dofadr[jid]))

        # Rodent body joint indices (starting after freejoint)
        if self._config.randomize_gaps:
            qpos_start = self._rodent_qpos_start
            qvel_start = self._rodent_qvel_start
        else:
            qpos_start = 7
            qvel_start = 6

        # All qpos indices from rodent body joints onward
        all_body_qpos = np.arange(qpos_start, m.nq)
        # All qvel indices from rodent body DOFs onward
        all_body_qvel = np.arange(qvel_start, m.nv)

        # Filter out eye joint indices
        self._body_qpos_indices = np.array(
            [i for i in all_body_qpos if i not in eye_qpos_addrs]
        )
        self._body_qvel_indices = np.array(
            [i for i in all_body_qvel if i not in eye_dof_addrs]
        )

        # For qfrc_actuator masking (uses nv-based indexing)
        if self._config.randomize_gaps:
            nv_start = self._rodent_root_dof
        else:
            nv_start = 0
        all_body_nv = np.arange(nv_start, m.nv)
        self._body_nv_indices = np.array(
            [i for i in all_body_nv if i not in eye_dof_addrs]
        )

        # Eye qpos indices (for eye state observation)
        self._eye_qpos_indices = np.array(sorted(eye_qpos_addrs))

        # Validate shapes match pre-eye dimensions
        expected_nq_body = self._pre_eye_nq - qpos_start
        expected_nv_body = self._pre_eye_nv - qvel_start
        assert len(self._body_qpos_indices) == expected_nq_body, (
            f"Body qpos indices {len(self._body_qpos_indices)} != "
            f"expected {expected_nq_body}"
        )
        assert len(self._body_qvel_indices) == expected_nv_body, (
            f"Body qvel indices {len(self._body_qvel_indices)} != "
            f"expected {expected_nv_body}"
        )

    @property
    def vision_shape(self):
        """Shape of the vision observation: (H, W, C) or (H, W, 2*C) for binocular."""
        mono_channels = 1 if self._grayscale else 3
        channels = (
            2 * mono_channels if self._config.get("binocular", False) else mono_channels
        )
        return (self._vision_height, self._vision_width, channels)

    @property
    def vision_enabled(self):
        """Whether vision observations are enabled."""
        return self._vision_enabled

    def _get_proprioception(self, data, info, flatten=True):
        """Get proprioception, excluding eye joint data when actuable_eyes=True.

        When actuable eyes are enabled, uses pre-computed index arrays to
        exclude eye joint angles, velocities, and actuator forces from the
        proprioception dict. This preserves the frozen prior decoder's
        expected input size. Eye state is instead included in task_obs.

        Args:
            data: The simulation data (mjx.Data).
            info: State info dictionary.
            flatten: Whether to flatten the OrderedDict to a 1-D array.

        Returns:
            Proprioception as an OrderedDict (if flatten=False) or flat array.
        """
        if not self._config.get("actuable_eyes", False):
            return super()._get_proprioception(data, info, flatten=flatten)

        # Use masked indices to exclude eye joint data
        proprioception = collections.OrderedDict(
            joint_angles=data.qpos[self._body_qpos_indices],
            joint_ang_vels=data.qvel[self._body_qvel_indices],
            actuator_ctrl=data.qfrc_actuator[self._body_nv_indices],
            body_height=self._get_body_height(data).reshape(1),
            world_zaxis=self._get_world_zaxis(data),
            appendages_pos=self._get_appendages_pos(data, flatten=flatten),
            kinematic_sensors=self._get_kinematic_sensors(data, flatten=flatten),
            touch_sensors=self._get_touch_sensors(data),
            prev_action=info["prev_action"][: self._pre_eye_nu],
        )
        if flatten:
            proprioception, _ = jax.flatten_util.ravel_pytree(proprioception)
        return proprioception

    def _get_obs(self, data, info) -> collections.OrderedDict:
        """Get observations in state/privileged_state dict format.

        Returns an OrderedDict with state and privileged_state keys, each
        containing:

        - task_obs: Body-state signals (prev_action, kinematic sensors, touch
          sensors, origin). When actuable_eyes=True, also includes eye joint
          angles. Downstream ``observation_utils.flatten_obs_dict()``
          maps this key to ``imitation_target`` internally.
        - proprioception: Nested OrderedDict of body state sensors. Used by
          the decoder in transfer learning.
        - vision: Zeros placeholder with shape (H, W, C). Real pixels are
          injected by VisionRenderWrapper after batched rendering.

        Gap features are intentionally excluded — the agent should infer gap
        information from vision.

        This structure is compatible with both ``HighLevelWrapper`` (for
        transfer learning) and direct ff_ppo training.

        Args:
            data: The simulation data (mjx.Data).
            info: State info dictionary.

        Returns:
            OrderedDict with state and privileged_state keys.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        task_obs_parts = [
            info["prev_action"],
            kinematic_sensors,
            touch_sensors,
            origin,
        ]

        if self._config.get("actuable_eyes", False):
            task_obs_parts.append(data.qpos[self._eye_qpos_indices])

        task_obs = jp.concatenate(task_obs_parts)

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
            vision=jp.zeros(self.vision_shape),
        )
        return collections.OrderedDict(
            state=obs,
            privileged_state=obs,
        )

    @property
    def observation_size(self) -> int:
        """Total flat observation size for the MLP (excludes vision pixels)."""
        obs_size = self.non_flattened_observation_size
        total = 0
        for key in ("task_obs", "proprioception"):
            total += jp.sum(flatten_util.ravel_pytree(obs_size["state"][key])[0])
        return total

    @property
    def proprioceptive_obs_size(self) -> int:
        """Flat size of the proprioceptive observation component."""
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0])

    @property
    def vision_obs_size(self) -> int:
        """Total number of pixels in the vision observation (H * W * C)."""
        h, w, c = self.vision_shape
        return h * w * c
