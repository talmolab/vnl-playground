"""Class for mouse forelimb reaching task."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import time

from mujoco_playground._src import mjx_env, reward
from vnl_mjx.tasks.mouse import consts


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.MOUSE_PATH / "xmls" / "assets")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.MOUSE_XML_PATH,
        ctrl_dt=0.001,
        sim_dt=0.001,
        Kp=35.0,
        Kd=0.5,
        episode_length=300,
    )


class MouseEnv(mjx_env.MjxEnv):
    """Base class for mouse environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize the MouseEnv class with mouse model

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for the environment.
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Optional overrides for the configuration. Defaults to None.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._spec = mujoco.MjSpec.from_string(
            epath.Path(self._walker_xml_path).read_text()
        )
        self._target_size = 0.001  # Size of target for the reaching task
        self._compiled = False

    @staticmethod
    def get_target_positions():
        """Return predefined target positions for reaching task."""
        return jp.array(
            [
                [0.004, 0.012, -0.006],
                [0.0025355, 0.012, -0.0024645],
                [-0.001, 0.012, -0.001],
                [-0.0045355, 0.012, -0.0024645],
                [-0.006, 0.012, -0.006],
                [-0.0045355, 0.012, -0.0095355],
                [-0.001, 0.012, -0.011],
                [0.0025355, 0.012, -0.0095355],
            ]
        )

    def add_target(self, pos=None, random_target=False) -> None:
        """
        Adds the target to the environment.

        Args:
            pos: Optional position for the target. If None and random_target is True,
                 a random position from the predefined list will be used.
            random_target: If True, select a random target position.
        """
        # If random_target is True, select a random position
        if random_target or pos is None:
            target_positions = self.get_target_positions()
            key = jax.random.PRNGKey(int(time.time() * 1e3))
            idx = jax.random.randint(key, (), 0, target_positions.shape[0])
            pos = target_positions[idx]

        # Store the target position
        self._target_position = jp.array(pos) if not isinstance(pos, jax.Array) else pos

        # Check if a target site already exists
        try:
            # If target exists, update its position
            target_site = self._spec.site("target")
            target_site.pos = list(pos)
        except (KeyError, AttributeError):
            # If target doesn't exist, create a new one
            self._spec.worldbody.add_site(
                name="target",
                pos=list(pos),
                size=[0.001, 0.001, 0.001],  # 1mm radius in all dimensions
                rgba=[0, 1, 0, 0.5],  # Green, semi-transparent
            )

        self.compile()

    def compile(self) -> None:
        """Compiles the model from the mj_spec and put models to mjx"""
        if not self._compiled:
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            # Increase offscreen framebuffer size to render at higher resolutions.
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mjx_model = mjx.put_model(self._mj_model)
            self._compiled = True

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        info = {}
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)  # Match the style in flat_arena.py
        metrics = {}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        # Apply the action to the model
        data = mjx_env.step(self.mjx_model, state.data, action)

        # Get the new observation
        obs = self._get_obs(data)

        # Compute the reward
        reward = jp.asarray(self._get_reward(data), dtype=jp.float32)

        # Check termination condition
        done = self._get_termination(data)

        # Update state with explicit types
        state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
        )

        return state

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        # Get the position and velocity of the rodent
        pos = data.qpos
        vel = data.qvel

        # Calculate target relative position (to_target observable)
        wrist_body = data.bind(self.mjx_model, self._spec.body("wrist_body"))
        wrist_pos = wrist_body.xpos

        # Get target position
        target_site = data.bind(self.mjx_model, self._spec.site("target"))
        target_pos = target_site.xpos

        # Vector from wrist to target
        to_target = target_pos - wrist_pos

        # Concatenate all observations
        obs = jp.concatenate([pos, vel, to_target])
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
    ) -> jp.ndarray:
        # Get the wrist position
        wrist_body = data.bind(self.mjx_model, self._spec.body("wrist_body"))
        wrist_pos = wrist_body.xpos

        # Get target position directly from the site in the model
        try:
            # Get target position directly from the site
            target_site = data.bind(self.mjx_model, self._spec.site("target"))
            target_pos = target_site.xpos
        except (KeyError, AttributeError):
            # Fallback to stored position if site doesn't exist
            target_pos = getattr(
                self, "_target_position", jp.array([0.004, 0.012, -0.006])
            )

        # Calculate distance between wrist and target
        to_target_dist = jp.linalg.norm(wrist_pos - target_pos)

        # Calculate reward based on distance - ensure proper float32 type
        radii = self._target_size
        reward_value = reward.tolerance(
            to_target_dist, bounds=(0, radii), margin=0.006, sigmoid="hyperbolic"
        )
        return jp.asarray(reward_value, dtype=jp.float32)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        return jp.zeros((), dtype=jp.float32)  # 0 → continue, 1 → terminate

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def xml_path(self) -> str:
        return self._walker_xml_path

    @property
    def walker_xml_path(self) -> str:
        return self._walker_xml_path

    @property
    def arena_xml_path(self) -> str:
        return self._arena_xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
