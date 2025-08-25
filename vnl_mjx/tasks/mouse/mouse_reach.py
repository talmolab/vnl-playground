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
from vnl_mjx.tasks.mouse.base import MouseBaseEnv, default_config  # <- import the base & config


class MouseEnv(MouseBaseEnv):
    """Mouse reaching env: targets + reward on wrist marker proximity."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize the MouseEnv with arena-only spec (via base), then you may
        call add_mouse(), add_target(), and compile().

        Args:
            config: Config dictionary.
            config_overrides: Optional overrides to config.
        """
        super().__init__(config, config_overrides)
        self._target_size = 0.001  # reaching radius
        self._wrist_body_id = None
        self._wrist_marker_geom_id = None

    @staticmethod
    def get_target_positions() -> jp.ndarray:
        """Return a fixed set of reachable target positions."""
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
            ],
            dtype=jp.float32,
        )

    def add_target(
        self,
        pos: Optional[Union[tuple[float, float, float], list[float]]] = None,
        random_target: bool = False,
        rng: Optional[jax.Array] = None,
    ) -> None:
        """
        Add a 'target' site for reaching.

        Args:
            pos: Optional (x, y, z). If None and `random_target=True`, a random
                position from `get_target_positions()` is used.
            random_target: Whether to sample a target when pos is None.
            rng: Optional JAX key for sampling.

        Returns:
            None
        """
        target_positions = self.get_target_positions()
        if rng is None:
            rng = jax.random.PRNGKey(int(time.time() * 1e3))

        idx = jax.random.randint(rng, (), 0, target_positions.shape[0], jp.int32)
        sampled = target_positions[idx]

        final_pos = sampled if (random_target or pos is None) else jp.asarray(pos)
        assert final_pos.shape == (3,)

        self._spec.worldbody.add_site(
            name="target",
            pos=[float(final_pos[0]), float(final_pos[1]), float(final_pos[2])],
            size=[0.001, 0.001, 0.001],
            rgba=[0, 1, 0, 0.5],
        )

    def compile(self) -> None:
        """
        Compile + cache IDs for wrist body and marker.

        Returns:
            None
        """
        super().compile()
        # Cache IDs once compiled
        self._wrist_body_id = self._mj_model.body("wrist_body").id
        self._wrist_marker_geom_id = self._mj_model.geom("wrist_marker").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """
        Reset the environment and sample a target for this episode.

        Args:
            rng: JAX PRNGKey.

        Returns:
            mjx_env.State: Initialized state with obs, reward=0, done=0, info.
        """
        # Sample a target
        target_positions = self.get_target_positions()
        idx = jax.random.randint(rng, (), 0, target_positions.shape[0], jp.int32)
        target_position = target_positions[idx]

        # Ensure a 'target' site exists and matches sampled target (host update).
        # Recompile if we have to add/update the site.
        try:
            _ = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "target")
            # Update spec site pos (by name) then recompile to reflect visuals.
            for i, site in enumerate(self._spec.worldbody.site):
                if site.name == "target":
                    self._spec.worldbody.site[i].pos = [float(x) for x in target_position]
                    break
            # Recompile visuals only; mjx model will be refreshed.
            self._mj_model = self._spec.compile()
            self._mjx_model = mjx.put_model(self._mj_model)
            # Refresh cached IDs.
            self._wrist_body_id = self._mj_model.body("wrist_body").id
            self._wrist_marker_geom_id = self._mj_model.geom("wrist_marker").id
        except Exception:
            # No target yet: create and compile.
            self._spec.worldbody.add_site(
                name="target",
                pos=[float(x) for x in target_position],
                size=[0.001, 0.001, 0.001],
                rgba=[0, 1, 0, 0.5],
            )
            self.compile()

        data = mjx_env.init(self.mjx_model)
        obs = self._get_obs(data, target_position)
        reward_val, done = jp.zeros(2, dtype=jp.float32)

        info = {"target_position": target_position}
        return mjx_env.State(data, obs, reward_val, done, {}, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """
        Take one physics step.

        Args:
            state: Current mjx_env.State.
            action: Control vector.

        Returns:
            mjx_env.State: Updated state.
        """
        data = mjx_env.step(self.mjx_model, state.data, action)
        target_position = state.info["target_position"]
        obs = self._get_obs(data, target_position)
        rew = jp.asarray(self._get_reward(data, target_position), dtype=jp.float32)
        done = self._get_termination(data)
        return state.replace(data=data, obs=obs, reward=rew, done=done)

    def _get_obs(self, data: mjx.Data, target_position: jp.ndarray) -> jax.Array:
        """
        Build observation vector [qpos, qvel, wrist_to_target].

        Args:
            data: mjx.Data.
            target_position: (3,) array.

        Returns:
            (D,) array observation.
        """
        pos = data.qpos
        vel = data.qvel
        wrist_pos = data.xpos[self._wrist_body_id]
        to_target = target_position - wrist_pos
        return jp.concatenate([pos, vel, to_target])

    def _get_reward(self, data: mjx.Data, target_position: jp.ndarray) -> jp.ndarray:
        """
        Distance-based tolerance reward from wrist marker to target.

        Args:
            data: mjx.Data.
            target_position: (3,) array.

        Returns:
            Scalar reward (jp.ndarray()).
        """
        wrist_marker_pos = data.geom_xpos[self._wrist_marker_geom_id]
        dist = jp.linalg.norm(wrist_marker_pos - target_position)
        return jp.asarray(
            reward.tolerance(dist, bounds=(0, self._target_size), margin=0.006, sigmoid="hyperbolic"),
            dtype=jp.float32,
        )

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """No early term by default."""
        return jp.zeros((), dtype=jp.float32)
