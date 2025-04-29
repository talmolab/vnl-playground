from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

from vnl_mjx.tasks.rodent import base as rodent_base
from vnl_mjx.tasks.rodent import consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.02,
        sim_dt=0.004,
        Kp=35.0,
        Kd=0.5,
        episode_length=300,
        drop_from_height_prob=0.6,
        settle_time=0.5,
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        energy_termination_threshold=np.inf,
        reward_config=config_dict.create(
            scales=config_dict.create(
                target_speed=2.0,
                action_rate=-0.001,
                torques=-1e-5,
                dof_acc=-2.5e-7,
                dof_vel=-0.1,
            ),
        ),
    )


class FlatWalk(rodent_base.RodentEnv):
    """Flat walk environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self.add_rodent()
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Sample a random initial configuration with some probability.

        data = mjx_env.init(self.mjx_model)

        info = {}

        obs = self._get_obs(data)
        reward, done = jp.zeros(2)

        metrics = {}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        # Unpack the state.

        # Apply the action to the model.
        data = mjx_env.step(self.mjx_model, state.data, action)

        # Get the new observation.
        obs = self._get_obs(data)

        # Compute the reward.
        rewards = self._get_reward(data)
        reward = rewards["speed * upright"]
        done = self._get_termination(data)
        state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
        )

        return state

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        obs = jp.concatenate([data.qpos, data.qvel])
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
    ) -> Dict[str, jax.Array]:
        speed_reward = self._get_speed_reward(data)
        upright_reward = self._upright_reward(data, deviation_angle=10)
        return {
            "speed_reward": speed_reward,
            "upright_reward": upright_reward,
            "speed * upright": speed_reward * upright_reward,
        }

    def _get_speed_reward(
        self,
        data: mjx.Data,
    ) -> jp.ndarray:
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        vel = jp.linalg.norm(body.subtree_linvel)
        target_speed = self._config.reward_config.scales.target_speed
        reward_value = reward.tolerance(
            vel, bounds=(target_speed, target_speed), margin=target_speed
        )
        return reward_value

    def _upright_reward(self, data: mjx.Data, deviation_angle=0):
        """Returns a reward proportional to how upright the torso is.

        Args:
        physics: an instance of `Physics`.
        walker: the focal walker.
        deviation_angle: A float, in degrees. The reward is 0 when the torso is
            exactly upside-down and 1 when the torso's z-axis is less than
            `deviation_angle` away from the global z-axis.
        """
        deviation = np.cos(np.deg2rad(deviation_angle))
        # xmat is the 3x3 rotation matrix of the current frame
        upright_torso = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xmat[
            -1, -1
        ]
        upright = reward.tolerance(
            upright_torso,
            bounds=(deviation, np.inf),
            sigmoid="linear",
            margin=1 + deviation,
            value_at_margin=0,
        )
        return np.min(upright)

    def _get_termination(
        self,
        data: mjx.Data,
    ) -> jax.Array:
        """
        Returns 1 if the rodent falls under the ground, 0 otherwise.

        Args:
            data (mjx.Data): _description_

        Returns:
            jax.Array: _description_
        """
        z = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xpos[-1]
        fall_under_ground = jp.where(z < 0.0, 1.0, 0.0)
        return fall_under_ground
