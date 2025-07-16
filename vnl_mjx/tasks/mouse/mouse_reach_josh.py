"""Class for mouse forelimb reaching task."""

from typing import Any, Dict, Optional, Union, Tuple

from absl import logging
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
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
        user_eval = False,
    ) -> None:
        """
        Initialize the MouseEnv class with mouse model

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for the environment.
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Optional overrides for the configuration. Defaults to None.
            target (jp array): Location of target. Defaults to randomization (if passed in None)
        """

        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config.walker_xml_path)
        self._spec = mujoco.MjSpec.from_string(
            epath.Path(self._walker_xml_path).read_text()
        )
        self._target_size = 0.001  # Size of target for the reaching task
        self._target = None
        self._compiled = False
        self._user_eval = user_eval



    def add_target(
            self,
            target: list,
            add_site: bool = False,
        ) -> None:
        """
        Adds the target to the environment.

        Args:
            target: Adds target to specified target
            add_site: Will add mujoco site to self._spec object and compile
        """
        # if self._target is not None:
        #     logging.info(f"Overwriting current target {self._target} \
        #                  with new target {target.tolist()}")

        # self._target = jp.array(target)
        self._target = target
        logging.info(f"Target added to pos {target}")
        
        if add_site:
            # add target to spec object
            self._spec.worldbody.add_site(
                name="target",
                pos = target,
                size=[self._target_size] * 3,
                rgba=[0, 1, 0, 0.5],
            )

        self.compile()


    def compile(self) -> None:
        """Compiles the model from the mj_spec and put models to mjx"""
        if not self._compiled:
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mjx_model = mjx.put_model(self._mj_model)

            # Store the wrist body ID for faster access
            # self._wrist_body_id = self._mj_model.body("wrist_body").id

            # Store hand mesh geom ID for faster access
            self._hand_mesh_geom_id = self._mj_model.geom("hand_mesh_geom").id
            self._compiled = True


    def reset(self, rng: jax.Array) -> mjx_env.State:
        """
        Reset the environment state for a new episode with a new random target.
        """
        
        # target = self._get_target_positions(rng)
        if not self._user_eval:
            target = jax.random.choice(rng, self._get_target_positions(rng))
            self.add_target(target)
        else:
            if self.target is None: raise AssertionError("target not defined")
            
        assert self.target.shape == (3,)
        print("------------------------------------------------------------------")
        print(f"Env reset with target position {self.target}")

        if not self._compiled: raise AssertionError("model not compiled; need to add target")

        # Initialize data with the model
        data = mjx_env.init(self.mjx_model)

        # Create observation and state - target flows through state.info
        info = {}
        info["target_position"] = self.target
        info["last_action"] = jp.zeros(self.mjx_model.nu)

        # add information for plotting activation and velocity distribution
        info['hand_pos'] = data.geom_xpos[self._hand_mesh_geom_id]
        # get the velocity of the body that is connected to the hand ('wrist body')
        hand_body_id = self._mj_model.geom_bodyid[self._hand_mesh_geom_id]
        info['hand_vel'] = data.cvel[hand_body_id][3:] # hand_body_id = 8
        info['qvel'] = data.qvel


        # code from scott that i'm trying to get work
        task_obs, proprioceptive_obs = self._get_obs(data, info)
        task_obs_size = task_obs.shape[0]
        proprioceptive_obs_size = proprioceptive_obs.shape[0]
        info["reference_obs_size"] = task_obs_size
        info["proprioceptive_obs_size"] = proprioceptive_obs_size
        obs = jp.concatenate([task_obs, proprioceptive_obs])
        reward, done = jp.zeros(2)
        metrics = {}

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

        # update last action
        state.info['last_action'] = action

        # update hand position in state
        hand_pos = data.geom_xpos[self._hand_mesh_geom_id]
        state.info['hand_pos'] = hand_pos

        # update velocity stuff
        hand_body_id = self._mj_model.geom_bodyid[self._hand_mesh_geom_id]
        state.info['hand_vel'] = data.cvel[hand_body_id][3:] # hand_body_id = 8
        state.info['qvel'] = data.qvel
        

        # get observations and reward
        task_obs, proprioceptive_obs = self._get_obs(data, state.info)
        obs = jp.concatenate([task_obs, proprioceptive_obs])
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
    

    def _get_target_positions(self, rng):
        targets = jp.array(
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
        return targets

        # defined these by visualizing possible arm reaches in arm env
        low  = jp.array([-0.010, 0.008, -0.012])
        high = jp.array([ 0.008, 0.015,  0.002])

        sample_point = jax.random.uniform(
            rng,
            shape=(3,),
            minval=low,
            maxval=high,
            dtype=jp.float32
        )
        return sample_point
    

    def _get_obs(
            self, data: mjx.Data, info: dict[str, Any]
        ) -> Tuple[jp.ndarray, jp.ndarray]:

        # Target position passed in from reset/step
        hand_pos = data.geom_xpos[self._hand_mesh_geom_id]
        distance_to_target = info['target_position'] - hand_pos

        task_obs = jp.concatenate(
            [
                info["last_action"],
                data.qpos,
                data.qvel,
                distance_to_target
            ]
        )

        # Concatenate all observation components
        proprioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel
            ]
        )

        return task_obs, proprioceptive_obs


    def _get_reward(
        self,
        data: mjx.Data,
        target_position: jp.ndarray,
    ) -> jp.ndarray:
        # Get the hand position using the stored ID
        hand_pos = data.geom_xpos[self._hand_mesh_geom_id]

        # Target position passed in from reset/step
        target_pos = target_position

        # Calculate distance between wrist and target
        to_target_dist = jp.linalg.norm(hand_pos - target_pos)

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
    def target(self) -> jp.ndarray:
        return self._target


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
