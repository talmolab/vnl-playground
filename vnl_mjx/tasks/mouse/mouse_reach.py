"""Class for mouse forelimb reaching task."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from vnl_mjx.tasks.mouse import consts

def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls" / "assets")
    return assets

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.MOUSE_XML_PATH,
    )

class MouseEnv(mjx_env.MjxEnv):
    """Base class for rodent environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize the RodentEnv class with only arena

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for the environment.
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Optional overrides for the configuration. Defaults to None.
            compile_spec (bool, optional): Whether to compile the model. Defaults to False.
        """
        super().__init__(config, config_overrides)
        self._mouse_xml_path = str(config.mouse_xml_path)
        self._spec = mujoco.MjSpec.from_string(config.mouse_xml_path.read_text())
        self._compiled = False

    def add_target(self, pos=(0, 0, 0.05)) -> None:
        """Adds the rodent model to the environment."""
        mouse = mujoco.MjSpec.from_string(
            epath.Path(self._walker_xml_path).read_text()
        )
        spawn_site = self._spec.worldbody.add_site(
            name="target",
            pos=list(pos),
            quat=[1, 0, 0, 0],
        )
        spawn_body = spawn_site.attach_body(mouse.worldbody, "", "-rodent")
        spawn_body.add_freejoint()

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
        pos = data.qpos
        vel = data.qvel

        # Implement to_target observable 

        # Concatenate all observations.
        obs = jp.concatenate([pos, vel]) # ADD TO_TARGET HERE
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
    ) -> jp.ndarray:
        body = data.bind(self.mjx_model, self._spec.body("wrist_body"))
        
        # Calculate distance between finger tip and target 

        reward_value = reward.tolerance(
            vel, bounds=(target_speed, target_speed), margin=.006
        )
        return reward_value
