"""Base class for modular rodent environments.

Corresponds to modular_rodent.jl in the Julia implementation.
"""

import collections
from typing import Any, Dict, Mapping, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import get_sensor_data

from . import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.MODULAR_RODENT_WALKER_XML_PATH,
        arena_xml_path=consts.MODULAR_RODENT_ARENA_XML_PATH,
        sim_dt=0.002,
        ctrl_dt=0.01,
        solver="cg",
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        mujoco_impl="warp",
    )


class ModularRodentEnv(mjx_env.MjxEnv):
    """Base class for modular rodent environments.

    Composes arena + walker at runtime (same pattern as rodent/ environments).
    Provides hierarchical proprioception organized by body module, matching
    the Julia ModularRodent implementation.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._spec = mujoco.MjSpec.from_file(str(self._config.arena_xml_path))
        walker_spec = mujoco.MjSpec.from_file(str(self._config.walker_xml_path))
        frame = self._spec.worldbody.add_frame()
        body = frame.attach_body(walker_spec.body("walker"), "", suffix="")
        body.add_freejoint(name="root")
        self._compiled = False

    def compile(self, forced: bool = False) -> None:
        """Compile the model spec into mj_model and mjx_model."""
        if not self._compiled or forced:
            self._spec.option.noslip_iterations = self._config.noslip_iterations
            # The XML has CPU-optimised njmax/nconmax (8000/4000). Warp requires
            # much smaller values; override them here.
            # self._spec.njmax = self._config.njmax
            # self._spec.nconmax = self._config.naconmax
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mj_model.opt.iterations = self._config.iterations
            self._mj_model.opt.ls_iterations = self._config.ls_iterations
            self._mj_model.opt.solver = {
                "cg": mujoco.mjtSolver.mjSOL_CG,
                "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            }[self._config.solver.lower()]
            self._mjx_model = mjx.put_model(
                self._mj_model, impl=self._config.mujoco_impl
            )
            self._compiled = True

    # -------------------------------------------------------------------------
    # Coordinate frame helpers (matching Julia's modular_rodent.jl)
    # -------------------------------------------------------------------------

    def torso_yawmat(self, data: mjx.Data) -> jp.ndarray:
        """Yaw-only rotation matrix of the torso (3×3).

        Extracts only the yaw angle from the full torso rotation, giving a
        horizontal reference frame aligned with the animal's heading direction.
        """
        xmat = data.xmat[self._mj_model.body("torso").id].reshape(3, 3)
        yaw = jp.arctan2(xmat[1, 0], xmat[0, 0])
        c, s = jp.cos(yaw), jp.sin(yaw)
        return jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def body_relpos(self, data: mjx.Data, body_name: str) -> jp.ndarray:
        """Position of a body in the yaw-aligned torso frame.

        Subtracts only (torso_x, torso_y, 0), so the z component equals the
        body's absolute height above ground.
        """
        body_pos = data.xpos[self._mj_model.body(body_name).id]
        torso_pos = data.xpos[self._mj_model.body("torso").id]
        torso_xy = jp.array([torso_pos[0], torso_pos[1], 0.0])
        return self.torso_yawmat(data) @ (body_pos - torso_xy)

    def site_z(self, data: mjx.Data, site_name: str) -> jp.ndarray:
        """Z-coordinate (absolute height) of a named site."""
        return data.site_xpos[self._mj_model.site(site_name).id, 2]

    def root_pos(self, data: mjx.Data) -> jp.ndarray:
        """Global position of the torso (used as the root)."""
        return data.xpos[self._mj_model.body("torso").id]

    def root_xquat(self, data: mjx.Data) -> jp.ndarray:
        """Global quaternion of the torso."""
        return data.xquat[self._mj_model.body("torso").id]

    # -------------------------------------------------------------------------
    # Modular proprioception (matching Julia's proprioception() function)
    # -------------------------------------------------------------------------

    def _get_modular_proprioception(
        self, data: mjx.Data
    ) -> dict[str, dict[str, jp.ndarray]]:
        """Hierarchical proprioception dict organized by body module.

        Matches Julia's ModularRodent.proprioception() named tuple structure,
        using Python dicts instead of named tuples.
        """
        m = self._mj_model
        yaw_mat = self.torso_yawmat(data)

        return {
            "hand_L": {
                "palm_contact": get_sensor_data(m, data, "hand_L/palm_contact"),
                "egocentric_hand_pos": self.body_relpos(data, "hand_L"),
                "wrist_angle": get_sensor_data(m, data, "hand_L/wrist_angle"),
                "finger_angle": get_sensor_data(m, data, "hand_L/finger_angle"),
                "wrist_angle_vel": get_sensor_data(m, data, "hand_L/wrist_angle_vel"),
                "finger_angle_vel": get_sensor_data(m, data, "hand_L/finger_angle_vel"),
                "finger_limit": get_sensor_data(m, data, "hand_L/finger_limit"),
                "wrist_limit": get_sensor_data(m, data, "hand_L/wrist_limit"),
                "xaxis": get_sensor_data(m, data, "hand_L/xaxis"),
                "finger_zz": get_sensor_data(m, data, "hand_L/finger_global_zaxis")[2:3],
                "finger_force": get_sensor_data(m, data, "hand_L/finger_force"),
                "wrist_force": get_sensor_data(m, data, "hand_L/wrist_force"),
            },
            "arm_L": {
                "egocentric_elbow_pos": get_sensor_data(m, data, "arm_L/egocentric_elbow_pos"),
                "elbow_angle": get_sensor_data(m, data, "arm_L/elbow_angle"),
                "shoulder_sup": get_sensor_data(m, data, "arm_L/shoulder_sup"),
                "shoulder": get_sensor_data(m, data, "arm_L/shoulder"),
                "scapula_extend": get_sensor_data(m, data, "arm_L/scapula_extend"),
                "scapula_abduct": get_sensor_data(m, data, "arm_L/scapula_abduct"),
                "scapula_supinate": get_sensor_data(m, data, "arm_L/scapula_supinate"),
                "elbow_angle_vel": get_sensor_data(m, data, "arm_L/elbow_angle_vel"),
                "shoulder_sup_vel": get_sensor_data(m, data, "arm_L/shoulder_sup_vel"),
                "shoulder_vel": get_sensor_data(m, data, "arm_L/shoulder_vel"),
                "scapula_extend_vel": get_sensor_data(m, data, "arm_L/scapula_extend_vel"),
                "scapula_abduct_vel": get_sensor_data(m, data, "arm_L/scapula_abduct_vel"),
                "scapula_supinate_vel": get_sensor_data(
                    m, data, "arm_L/scapula_supinate_vel"
                ),
                "elbow_limit": get_sensor_data(m, data, "arm_L/elbow_limit"),
                "elbow_force": get_sensor_data(m, data, "arm_L/elbow_force"),
                "shoulder_force": get_sensor_data(m, data, "arm_L/shoulder_force"),
                "elbow_height": jp.expand_dims(
                    self.site_z(data, "elbow_L"), axis=0
                ),
                "shoulder_height": jp.expand_dims(
                    self.site_z(data, "shoulder_L"), axis=0
                ),
                "hand_linvel": yaw_mat @ get_sensor_data(m, data, "arm_L/hand_linvel"),
            },
            "hand_R": {
                "palm_contact": get_sensor_data(m, data, "hand_R/palm_contact"),
                "egocentric_hand_pos": self.body_relpos(data, "hand_R"),
                "wrist_angle": get_sensor_data(m, data, "hand_R/wrist_angle"),
                "finger_angle": get_sensor_data(m, data, "hand_R/finger_angle"),
                "wrist_angle_vel": get_sensor_data(m, data, "hand_R/wrist_angle_vel"),
                "finger_angle_vel": get_sensor_data(m, data, "hand_R/finger_angle_vel"),
                "finger_limit": get_sensor_data(m, data, "hand_R/finger_limit"),
                "wrist_limit": get_sensor_data(m, data, "hand_R/wrist_limit"),
                "xaxis": get_sensor_data(m, data, "hand_R/xaxis"),
                "finger_zz": get_sensor_data(m, data, "hand_R/finger_global_zaxis")[2:3],
                "finger_force": get_sensor_data(m, data, "hand_R/finger_force"),
                "wrist_force": get_sensor_data(m, data, "hand_R/wrist_force"),
            },
            "arm_R": {
                "egocentric_elbow_pos": (
                    get_sensor_data(m, data, "arm_R/egocentric_elbow_pos")
                ),
                "elbow_angle": get_sensor_data(m, data, "arm_R/elbow_angle"),
                "shoulder_sup": get_sensor_data(m, data, "arm_R/shoulder_sup"),
                "shoulder": get_sensor_data(m, data, "arm_R/shoulder"),
                "scapula_extend": get_sensor_data(m, data, "arm_R/scapula_extend"),
                "scapula_abduct": get_sensor_data(m, data, "arm_R/scapula_abduct"),
                "scapula_supinate": get_sensor_data(m, data, "arm_R/scapula_supinate"),
                "elbow_angle_vel": get_sensor_data(m, data, "arm_R/elbow_angle_vel"),
                "shoulder_sup_vel": get_sensor_data(m, data, "arm_R/shoulder_sup_vel"),
                "shoulder_vel": get_sensor_data(m, data, "arm_R/shoulder_vel"),
                "scapula_extend_vel": get_sensor_data(m, data, "arm_R/scapula_extend_vel"),
                "scapula_abduct_vel": get_sensor_data(m, data, "arm_R/scapula_abduct_vel"),
                "scapula_supinate_vel": get_sensor_data(
                    m, data, "arm_R/scapula_supinate_vel"
                ),
                "elbow_limit": get_sensor_data(m, data, "arm_R/elbow_limit"),
                "elbow_force": get_sensor_data(m, data, "arm_R/elbow_force"),
                "shoulder_force": get_sensor_data(m, data, "arm_R/shoulder_force"),
                "elbow_height": jp.expand_dims(
                    self.site_z(data, "elbow_R"), axis=0
                ),
                "shoulder_height": jp.expand_dims(
                    self.site_z(data, "shoulder_R"), axis=0
                ),
                "hand_linvel": yaw_mat @ get_sensor_data(m, data, "arm_R/hand_linvel"),
            },
            "foot_L": {
                "sole_contact": get_sensor_data(m, data, "foot_L/sole_contact"),
                "heel_contact": get_sensor_data(m, data, "foot_L/heel_contact"),
                "egocentric_foot_pos": self.body_relpos(data, "foot_L"),
                "knee_angle": get_sensor_data(m, data, "foot_L/knee_angle"),
                "ankle_angle": get_sensor_data(m, data, "foot_L/ankle_angle"),
                "toe_angle": get_sensor_data(m, data, "foot_L/toe_angle"),
                "knee_angle_vel": get_sensor_data(m, data, "foot_L/knee_angle_vel"),
                "ankle_angle_vel": get_sensor_data(m, data, "foot_L/ankle_angle_vel"),
                "toe_angle_vel": get_sensor_data(m, data, "foot_L/toe_angle_vel"),
                "toe_limit": get_sensor_data(m, data, "foot_L/toe_limit"),
                "foot_limit": get_sensor_data(m, data, "foot_L/foot_limit"),
                "xaxis": get_sensor_data(m, data, "foot_L/xaxis"),
                "toe_zz": get_sensor_data(m, data, "foot_L/toe_global_zaxis")[2:3],
                "toe_force": get_sensor_data(m, data, "foot_L/toe_force"),
                "ankle_force": get_sensor_data(m, data, "foot_L/ankle_force"),
            },
            "leg_L": {
                "egocentric_knee_pos": (
                    get_sensor_data(m, data, "leg_L/egocentric_knee_pos")
                ),
                "hip_supinate": get_sensor_data(m, data, "leg_L/hip_supinate"),
                "hip_abduct": get_sensor_data(m, data, "leg_L/hip_abduct"),
                "hip_extend": get_sensor_data(m, data, "leg_L/hip_extend"),
                "knee_angle": get_sensor_data(m, data, "leg_L/knee_angle"),
                "hip_supinate_vel": get_sensor_data(m, data, "leg_L/hip_supinate_vel"),
                "hip_abduct_vel": get_sensor_data(m, data, "leg_L/hip_abduct_vel"),
                "hip_extend_vel": get_sensor_data(m, data, "leg_L/hip_extend_vel"),
                "knee_angle_vel": get_sensor_data(m, data, "leg_L/knee_angle_vel"),
                "knee_limit": get_sensor_data(m, data, "leg_L/knee_limit"),
                "knee_force": get_sensor_data(m, data, "leg_L/knee_force"),
                "hip_force": get_sensor_data(m, data, "leg_L/hip_force"),
                "pelvis_zaxis": yaw_mat @ get_sensor_data(m, data, "pelvis/zaxis"),
                "hip_height": jp.expand_dims(
                    self.site_z(data, "hip_L"), axis=0
                ),
                "hip_accelerometer": (
                    get_sensor_data(m, data, "leg_L/hip_accelerometer") * 0.1
                ),
                "hip_gyro": get_sensor_data(m, data, "leg_L/hip_gyro") * 0.1,
            },
            "foot_R": {
                "sole_contact": get_sensor_data(m, data, "foot_R/sole_contact"),
                "heel_contact": get_sensor_data(m, data, "foot_R/heel_contact"),
                "egocentric_foot_pos": self.body_relpos(data, "foot_R"),
                "knee_angle": get_sensor_data(m, data, "foot_R/knee_angle"),
                "ankle_angle": get_sensor_data(m, data, "foot_R/ankle_angle"),
                "toe_angle": get_sensor_data(m, data, "foot_R/toe_angle"),
                "knee_angle_vel": get_sensor_data(m, data, "foot_R/knee_angle_vel"),
                "ankle_angle_vel": get_sensor_data(m, data, "foot_R/ankle_angle_vel"),
                "toe_angle_vel": get_sensor_data(m, data, "foot_R/toe_angle_vel"),
                "toe_limit": get_sensor_data(m, data, "foot_R/toe_limit"),
                "foot_limit": get_sensor_data(m, data, "foot_R/foot_limit"),
                "xaxis": get_sensor_data(m, data, "foot_R/xaxis"),
                "toe_zz": get_sensor_data(m, data, "foot_R/toe_global_zaxis")[2:3],
                "toe_force": get_sensor_data(m, data, "foot_R/toe_force"),
                "ankle_force": get_sensor_data(m, data, "foot_R/ankle_force"),
            },
            "leg_R": {
                "egocentric_knee_pos": (
                    get_sensor_data(m, data, "leg_R/egocentric_knee_pos")
                ),
                "hip_supinate": get_sensor_data(m, data, "leg_R/hip_supinate"),
                "hip_abduct": get_sensor_data(m, data, "leg_R/hip_abduct"),
                "hip_extend": get_sensor_data(m, data, "leg_R/hip_extend"),
                "knee_angle": get_sensor_data(m, data, "leg_R/knee_angle"),
                "ankle_angle": get_sensor_data(m, data, "leg_R/ankle_angle"),
                "hip_supinate_vel": get_sensor_data(m, data, "leg_R/hip_supinate_vel"),
                "hip_abduct_vel": get_sensor_data(m, data, "leg_R/hip_abduct_vel"),
                "hip_extend_vel": get_sensor_data(m, data, "leg_R/hip_extend_vel"),
                "knee_angle_vel": get_sensor_data(m, data, "leg_R/knee_angle_vel"),
                "ankle_angle_vel": get_sensor_data(m, data, "leg_R/ankle_angle_vel"),
                "knee_limit": get_sensor_data(m, data, "leg_R/knee_limit"),
                "ankle_force": get_sensor_data(m, data, "leg_R/ankle_force"),
                "knee_force": get_sensor_data(m, data, "leg_R/knee_force"),
                "hip_force": get_sensor_data(m, data, "leg_R/hip_force"),
                "pelvis_zaxis": yaw_mat @ get_sensor_data(m, data, "pelvis/zaxis"),
                "hip_height": jp.expand_dims(
                    self.site_z(data, "hip_R"), axis=0
                ),
                "hip_accelerometer": get_sensor_data(m, data, "leg_R/hip_accelerometer"),
                "hip_gyro": get_sensor_data(m, data, "leg_R/hip_gyro"),
            },
            "torso": {
                "accelerometer": get_sensor_data(m, data, "torso/accelerometer"),
                "gyro": get_sensor_data(m, data, "torso/gyro"),
                "zaxis": yaw_mat @ get_sensor_data(m, data, "torso/zaxis"),
                "lumbar_extend": get_sensor_data(m, data, "torso/lumbar_extend"),
                "lumbar_bend": get_sensor_data(m, data, "torso/lumbar_bend"),
                "lumbar_twist": get_sensor_data(m, data, "torso/lumbar_twist"),
                "lumbar_extend_vel": get_sensor_data(m, data, "torso/lumbar_extend_vel"),
                "lumbar_bend_vel": get_sensor_data(m, data, "torso/lumbar_bend_vel"),
                "lumbar_twist_vel": get_sensor_data(m, data, "torso/lumbar_twist_vel"),
                "linvel": get_sensor_data(m, data, "torso/linvel"),
                "height_above_ground": jp.expand_dims(
                    self.body_relpos(data, "torso")[2], axis=0
                ),
                "pelvis_zaxis": yaw_mat @ get_sensor_data(m, data, "pelvis/zaxis"),
            },
            "head": {
                "accelerometer": get_sensor_data(m, data, "head/accelerometer"),
                "egocentric_pos": self.body_relpos(data, "skull"),
                "xaxis": get_sensor_data(m, data, "head/xaxis"),
                "zaxis": get_sensor_data(m, data, "head/zaxis"),
                "neck_force": get_sensor_data(m, data, "head/neck_force"),
                "atlas": get_sensor_data(m, data, "head/atlas"),
                "atlant_extend": get_sensor_data(m, data, "head/atlant_extend"),
                "mandible": get_sensor_data(m, data, "head/mandible"),
                "cervical_extend": get_sensor_data(m, data, "head/cervical_extend"),
                "cervical_bend": get_sensor_data(m, data, "head/cervical_bend"),
                "cervical_twist": get_sensor_data(m, data, "head/cervical_twist"),
                "atlas_vel": get_sensor_data(m, data, "head/atlas_vel"),
                "atlant_extend_vel": get_sensor_data(m, data, "head/atlant_extend_vel"),
                "mandible_vel": get_sensor_data(m, data, "head/mandible_vel"),
                "cervical_extend_vel": get_sensor_data(m, data, "head/cervical_extend_vel"),
                "cervical_bend_vel": get_sensor_data(m, data, "head/cervical_bend_vel"),
                "cervical_twist_vel": get_sensor_data(m, data, "head/cervical_twist_vel"),
                "linvel": get_sensor_data(m, data, "head/linvel"),
            },
        }

    def _get_target_from_pose(
        self, data: mjx.Data
    ) -> dict[str, dict[str, jp.ndarray]]:
        """Subset of proprioception used as imitation targets.

        Matches Julia's target_from_pose() function — only includes quantities
        that appear in compute_rewards() comparisons. Excludes velocity/force
        sensors that don't make sense as reference targets.
        """
        m = self._mj_model
        return {
            "hand_L": {
                "pos": self.body_relpos(data, "hand_L"),
                "wrist_angle": get_sensor_data(m, data, "hand_L/wrist_angle"),
                "finger_angle": get_sensor_data(m, data, "hand_L/finger_angle"),
                "xaxis": get_sensor_data(m, data, "hand_L/xaxis"),
            },
            "arm_L": {
                "elbow_angle": get_sensor_data(m, data, "arm_L/elbow_angle"),
                "elbow_height": jp.expand_dims(
                    self.site_z(data, "elbow_L"), axis=0
                ),
                "shoulder_height": jp.expand_dims(
                    self.site_z(data, "shoulder_L"), axis=0
                ),
            },
            "hand_R": {
                "pos": self.body_relpos(data, "hand_R"),
                "wrist_angle": get_sensor_data(m, data, "hand_R/wrist_angle"),
                "finger_angle": get_sensor_data(m, data, "hand_R/finger_angle"),
                "xaxis": get_sensor_data(m, data, "hand_R/xaxis"),
            },
            "arm_R": {
                "elbow_angle": get_sensor_data(m, data, "arm_R/elbow_angle"),
                "elbow_height": jp.expand_dims(
                    self.site_z(data, "elbow_R"), axis=0
                ),
                "shoulder_height": jp.expand_dims(
                    self.site_z(data, "shoulder_R"), axis=0
                ),
            },
            "foot_L": {
                "pos": self.body_relpos(data, "foot_L"),
                "toe_angle": get_sensor_data(m, data, "foot_L/toe_angle"),
                "ankle_angle": get_sensor_data(m, data, "foot_L/ankle_angle"),
                "xaxis": get_sensor_data(m, data, "foot_L/xaxis"),
            },
            "leg_L": {
                "knee_pos": get_sensor_data(m, data, "leg_L/egocentric_knee_pos"),
                "knee_angle": get_sensor_data(m, data, "leg_L/knee_angle"),
                "hip_height": jp.expand_dims(
                    self.site_z(data, "hip_L"), axis=0
                ),
            },
            "foot_R": {
                "pos": self.body_relpos(data, "foot_R"),
                "toe_angle": get_sensor_data(m, data, "foot_R/toe_angle"),
                "ankle_angle": get_sensor_data(m, data, "foot_R/ankle_angle"),
                "xaxis": get_sensor_data(m, data, "foot_R/xaxis"),
            },
            "leg_R": {
                "knee_pos": get_sensor_data(m, data, "leg_R/egocentric_knee_pos"),
                "knee_angle": get_sensor_data(m, data, "leg_R/knee_angle"),
                "hip_height": jp.expand_dims(
                    self.site_z(data, "hip_R"), axis=0
                ),
            },
            "torso": {
                "zaxis": self.torso_yawmat(data) @ get_sensor_data(m, data, "torso/zaxis"),
                "pelvis_zaxis": self.torso_yawmat(data) @ get_sensor_data(m, data, "pelvis/zaxis"),
                "lumbar_bend": get_sensor_data(m, data, "torso/lumbar_bend"),
                "lumbar_twist": get_sensor_data(m, data, "torso/lumbar_twist"),
                "lumbar_extend": get_sensor_data(m, data, "torso/lumbar_extend"),
                "height_above_ground": jp.expand_dims(
                    self.body_relpos(data, "torso")[2], axis=0
                ),
            },
            "head": {
                "xaxis": get_sensor_data(m, data, "head/xaxis"),
                "zaxis": get_sensor_data(m, data, "head/zaxis"),
                "egocentric_pos": self.body_relpos(data, "skull"),
                "mandible": get_sensor_data(m, data, "head/mandible"),
            },
            "root": {
                "pos": self.root_pos(data),
                "rot": self.root_xquat(data),
            },
        }

    # -------------------------------------------------------------------------
    # Action helpers
    # -------------------------------------------------------------------------

    def null_action(self) -> dict[str, jp.ndarray]:
        """Zero action dict matching Julia's null_action(rodent)."""
        return {
            "hand_L": jp.zeros(2),
            "arm_L": jp.zeros(6),
            "hand_R": jp.zeros(2),
            "arm_R": jp.zeros(6),
            "foot_L": jp.zeros(2),
            "leg_L": jp.zeros(4),
            "foot_R": jp.zeros(2),
            "leg_R": jp.zeros(4),
            "torso": jp.zeros(3),
            "head": jp.zeros(5),
        }

    def _action_to_ctrl(self, action: dict[str, jp.ndarray]) -> jp.ndarray:
        """Map action dict to the 38-element ctrl array.

        Actuator ordering in rodent_modular_walker.xml:
          [0:3]   torso:  lumbar_extend, lumbar_bend, lumbar_twist
          [3:6]   head:   cervical_extend, cervical_bend, cervical_twist
          [6:8]   tail:   caudal_extend, caudal_bend  (uncontrolled, set to 0)
          [8:12]  leg_L:  hip_L_supinate, hip_L_abduct, hip_L_extend, knee_L
          [12:14] foot_L: ankle_L, toe_L
          [14:18] leg_R:  hip_R_supinate, hip_R_abduct, hip_R_extend, knee_R
          [18:20] foot_R: ankle_R, toe_R
          [20:22] head:   atlas, mandible           (head[3:5])
          [22:28] arm_L:  scapula_L x3, shoulder_L, shoulder_sup_L, elbow_L
          [28:30] hand_L: wrist_L, finger_L
          [30:36] arm_R:  scapula_R x3, shoulder_R, shoulder_sup_R, elbow_R
          [36:38] hand_R: wrist_R, finger_R
        """
        return jp.concatenate([
            action["torso"],        # [0:3]
            action["head"][:3],     # [3:6] cervical
            jp.zeros(2),            # [6:8] tail (uncontrolled)
            action["leg_L"],        # [8:12]
            action["foot_L"],       # [12:14]
            action["leg_R"],        # [14:18]
            action["foot_R"],       # [18:20]
            action["head"][3:],     # [20:22] atlas + mandible
            action["arm_L"],        # [22:28]
            action["hand_L"],       # [28:30]
            action["arm_R"],        # [30:36]
            action["hand_R"],       # [36:38]
        ])

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return str(self._config.walker_xml_path)

    @property
    def action_size(self) -> int:
        return jax.tree.map(len, self.null_action())

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
