"""Base environment for PlanarWalker tasks.

Provides shared utilities (proprioception, body positions, joint access)
following the vnl_playground pattern while using the mujoco_playground
PlanarWalker XML model.
"""

import collections
from typing import Any, Dict, Mapping, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

from vnl_playground.tasks.walker import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        sim_dt=0.0025,
        ctrl_dt=0.025,
        episode_length=1000,
        mujoco_impl="jax",
        nconmax=50_000,
        njmax=100,
    )


class WalkerEnv(mjx_env.MjxEnv):
    """Base class for PlanarWalker environments.

    Loads the walker XML from mujoco_playground and provides helper methods
    for extracting observations, body positions, and joint states.

    The walker is a 2D planar biped with:
    - 3 root DOFs: rootz (slide z), rootx (slide x), rooty (hinge y)
    - 6 joint DOFs: right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle
    - 6 actuators (motors): one per joint
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        # Load walker model
        xml_path = consts.WALKER_XML_PATH
        model_assets = common.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(), model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.mujoco_impl)
        self._xml_path = xml_path.as_posix()

        # Cache body and joint indices
        self._torso_id = self._mj_model.body("torso").id
        self._body_ids = {
            name: self._mj_model.body(name).id for name in consts.BODIES
        }
        self._joint_lowers = self._mj_model.jnt_range[consts.N_ROOT_QPOS:, 0]
        self._joint_uppers = self._mj_model.jnt_range[consts.N_ROOT_QPOS:, 1]

    # -------------------------------------------------------------------------
    # Joint and body accessors
    # -------------------------------------------------------------------------

    def _get_joint_angles(self, data: mjx.Data) -> jp.ndarray:
        """Get the 6 leg joint angles (excluding root DOFs)."""
        return data.qpos[consts.N_ROOT_QPOS:]

    def _get_joint_ang_vels(self, data: mjx.Data) -> jp.ndarray:
        """Get the 6 leg joint velocities (excluding root DOFs)."""
        return data.qvel[consts.N_ROOT_QVEL:]

    def _get_body_height(self, data: mjx.Data) -> jp.ndarray:
        """Get the torso z-position (height)."""
        return data.xpos[self._torso_id, 2]

    def _get_torso_upright(self, data: mjx.Data) -> jp.ndarray:
        """Get the zz component of torso rotation matrix (1.0 = upright)."""
        return data.xmat[self._torso_id, 2, 2]

    def _get_horizontal_velocity(self, data: mjx.Data) -> jp.ndarray:
        """Get horizontal (x) velocity of the torso center of mass."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, "torso_subtreelinvel"
        )[0]

    def _get_bodies_pos(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[dict[str, jp.ndarray], jp.ndarray]:
        """Get global positions of all body parts."""
        bodies_pos = collections.OrderedDict()
        for name in consts.BODIES:
            bodies_pos[name] = data.xpos[self._body_ids[name]]
        if flatten:
            bodies_pos, _ = jax.flatten_util.ravel_pytree(bodies_pos)
        return bodies_pos

    def _get_bodies_quat(
        self, data: mjx.Data, flatten: bool = True
    ) -> Union[dict[str, jp.ndarray], jp.ndarray]:
        """Get global quaternions of all body parts."""
        bodies_quat = collections.OrderedDict()
        for name in consts.BODIES:
            bodies_quat[name] = data.xquat[self._body_ids[name]]
        if flatten:
            bodies_quat, _ = jax.flatten_util.ravel_pytree(bodies_quat)
        return bodies_quat

    def _get_orientations(self, data: mjx.Data) -> jp.ndarray:
        """Get planar orientations of all bodies (xx, xz components).

        Returns shape (n_bodies * 2,) = (14,) for 7 bodies.
        """
        return data.xmat[1:, [0, 0], [0, 2]].ravel()

    def _get_proprioception(
        self, data: mjx.Data, info: Mapping[str, Any], flatten: bool = True
    ) -> Union[jp.ndarray, Mapping[str, jp.ndarray]]:
        """Get proprioceptive observations."""
        proprioception = collections.OrderedDict(
            orientations=self._get_orientations(data),
            height=self._get_body_height(data).reshape(1),
            upright=self._get_torso_upright(data).reshape(1),
            velocity=data.qvel,
            joint_angles=self._get_joint_angles(data),
            prev_action=info.get("prev_action", jp.zeros(self.action_size)),
        )
        if flatten:
            proprioception, _ = jax.flatten_util.ravel_pytree(proprioception)
        return proprioception

    def get_joint_names(self):
        return consts.JOINT_NAMES

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
