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

rodent_xml_path = epath.Path(__file__).parent / "xmls" / "rodent_sphere_feet.xml"

ARENA_XML = """
<mujoco>
  <visual>
    <headlight diffuse=".5 .5 .5" specular="1 1 1"/>
    <global offwidth="2048" offheight="2048"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="10" height="10"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="400" height="400"/>
    <material name="groundplane" texture="groundplane" texrepeat="45 45" reflectance="0"/>
  </asset>

  <worldbody>
    <geom name="floor" size="10 10 0.005" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
    xml_path=rodent_xml_path,
    arena_xml=ARENA_XML,
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
            target_speed=5.0,
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
        reward = self._get_reward(data)

        done = state.done
        
        state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
        )

        return state
        
    def _get_obs(self, data: mjx.Data) -> jax.Array:
        # Get the position and velocity of the rodent.
        pos = data.qpos[3:6]
        vel = data.qvel[3:6]
        # Concatenate all observations.
        obs = jp.concatenate([pos, vel])
        return obs
    
    def _get_reward(
      self,
      data: mjx.Data,
    ) -> jp.ndarray:
        body = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        vel = jp.linalg.norm(body.subtree_linvel)
        reward_value = reward.tolerance(
            vel,
            bounds=(self._config.reward_config.scales.target_speed, self._config.reward_config.scales.target_speed),
            margin=self._config.reward_config.scales.target_speed
        )
        return reward_value