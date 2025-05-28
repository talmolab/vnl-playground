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

        # Spawn a random target for this env instance
        # self.add_target(random_target=True)

    @staticmethod
    def get_target_positions():
        """Return predefined target positions for reaching task."""
        return jp.array(
            [
                [0.007, 0.010, -0.006],
                [0.0055355, 0.010, -0.0024645],
                [0.002, 0.010, -0.001],
                [-0.0015355, 0.010, -0.0024645],
                [-0.003, 0.010, -0.006],
                [-0.0015355, 0.010, -0.0095355],
                [0.002, 0.010, -0.011],
                [0.0055355, 0.010, -0.0095355],
            ]
        )

    def add_target(self, pos=None, random_target=False, rng=None) -> None:
        """
        Adds the target to the environment.

        Args:
            pos: Optional position for the target. If None and random_target is True,
                 a random position from the predefined list will be used.
            random_target: If True, select a random target position.
            rng: Optional JAX random key. If provided, uses this for randomization,
                 which enables proper vectorization in JAX.
        """
        # ---------------------------------------------------------------------
        # Choose a target position without Python‑level branching
        # ---------------------------------------------------------------------
        target_positions = self.get_target_positions()  # (N, 3)
        rng_fallback = jax.random.PRNGKey(int(time.time() * 1e3))
        rng = rng if rng is not None else rng_fallback

        # Sample index once; will be ignored when not used
        idx = jax.random.randint(rng, (), 0, target_positions.shape[0], jp.int32)
        sampled_pos = target_positions[idx]  # (3,)

        # Convert user‑provided pos (may be None) to a JAX array without Python branching
        pos_is_none = pos is None  # Python bool, treated as static
        provided_pos = jax.lax.cond(
            pos_is_none,
            lambda _: jp.zeros(3, dtype=jp.float32),
            lambda p: jp.asarray(p, dtype=jp.float32),
            operand=pos if pos is not None else jp.zeros(3),
        )

        # Use lax.cond to pick between sampled_pos and provided_pos
        pos = jax.lax.cond(
            random_target | (pos is None),
            lambda _: sampled_pos,
            lambda _: provided_pos,
            operand=None,
        )

        self._spec.worldbody.add_site(
            name="target",
            pos=pos,  # Use Python floats
            size=[0.001, 0.001, 0.001],
            rgba=[0, 1, 0, 0.5],
        )

        self.compile()
        self._compiled = True

    def compile(self) -> None:
        """Compiles the model from the mj_spec and put models to mjx"""
        if not self._compiled:
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mjx_model = mjx.put_model(self._mj_model)

            # Store the wrist body ID for faster access
            self._wrist_body_id = self._mj_model.body("wrist_body").id

            # Store the wrist_marker geom ID for reward calculation
            self._wrist_marker_geom_id = self._mj_model.geom("wrist_marker").id

            self._compiled = True

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """
        Reset the environment state for a new episode with a new random target.
        """
        # Get a new random target position
        target_positions = self.get_target_positions()
        idx = jax.random.randint(rng, (), 0, target_positions.shape[0], jp.int32)
        target_position = target_positions[idx]

        # --- Host-only: Update target site to match the reward target position ---
        if not isinstance(target_position, jax.core.Tracer):
            # Safe conversion to Python floats
            pos_list = [float(x) for x in target_position]

            # First check if target site exists
            target_site_id = -1
            if hasattr(self, "_mj_model"):
                try:
                    target_site_id = mujoco.mj_name2id(
                        self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "target"
                    )
                except Exception:
                    pass

            if target_site_id >= 0:
                # Target site exists in compiled model, update it
                # First update the spec
                for i, site in enumerate(self._spec.worldbody.site):
                    if site.name == "target":
                        self._spec.worldbody.site[i].pos = pos_list
                        break
                # Recompile to update visual
                self._mj_model = self._spec.compile()
                self._mjx_model = mjx.put_model(self._mj_model)
            else:
                # No target site, create one
                self._spec.worldbody.site = [
                    site for site in self._spec.worldbody.site if site.name != "target"
                ]
                self._spec.worldbody.add_site(
                    name="target",
                    pos=pos_list,
                    size=[0.001, 0.001, 0.001],
                    rgba=[0, 1, 0, 0.5],
                )
                # Compile to update model
                self.compile()

        # --- End host-only update ---

        # Initialize data with the model
        data = mjx_env.init(self.mjx_model)

        # Create observation and state - target flows through state.info
        obs = self._get_obs(data, target_position)
        reward, done = jp.zeros(2)
        metrics = {}
        info = {"target_position": target_position}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        # Apply the action to the model
        data = mjx_env.step(self.mjx_model, state.data, action)

        # Get the new observation and reward using target from info
        target_position = state.info["target_position"]
        obs = self._get_obs(data, target_position)
        reward = jp.asarray(self._get_reward(data, target_position), dtype=jp.float32)

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

    def _get_obs(self, data: mjx.Data, target_position: jp.ndarray) -> jax.Array:
        # Get joint positions and velocities
        pos = data.qpos
        vel = data.qvel

        # Use the stored body ID instead of looking it up via spec
        wrist_pos = data.xpos[self._wrist_body_id]

        # Target position passed in from reset/step
        to_target = target_position - wrist_pos

        # Concatenate all observation components
        obs = jp.concatenate([pos, vel, to_target])

        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        target_position: jp.ndarray,
    ) -> jp.ndarray:
        # Get the wrist_marker geom position using the stored ID
        wrist_marker_pos = data.geom_xpos[self._wrist_marker_geom_id]

        # Target position passed in from reset/step
        target_pos = target_position

        # Calculate distance between wrist_marker and target
        to_target_dist = jp.linalg.norm(wrist_marker_pos - target_pos)

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
