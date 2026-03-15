"""Arena forage task for virtual rodent.

The rodent starts at the center of a flat arena and must navigate to collect
treats (blue spheres) placed at random locations. Each treat emits a Gaussian
"scent" signal; the rodent observes the sum of all active scents at its skull
position as a single scalar. Collecting a treat (skull enters sphere region)
gives a sparse reward and removes it. The episode terminates when all treats
are collected or on NaN.
"""

import collections
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import flatten_util
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the ArenaForage environment."""
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="cg",
        mujoco_impl="jax",
        naconmax=16 * 8192,
        njmax=512,
        iterations=10,
        ls_iterations=5,
        noslip_iterations=0,
        torque_actuators=True,
        rescale_factor=0.9,
        # Arena forage specific config
        num_treats=5,
        treat_sphere_radius=0.03,
        treat_height=0.03,
        spawn_radius=1.0,
        min_spawn_radius=0.3,
        scent_sigma=0.5,
        reward_per_treat=1.0,
        discounting=0.99,
        episode_length=2000,
        action_repeat=1,
        reward_terms={
            "treat_collection": {"weight": 1.0},
        },
        termination_criteria={
            "all_collected": {},
            "nan_termination": {},
        },
    )


_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


def _named_reward(name: str):
    """Decorator to register a reward function."""

    def decorator(reward_fcn: Callable):
        _REWARD_FCN_REGISTRY[name] = reward_fcn
        return reward_fcn

    return decorator


def _named_termination_criterion(name: str):
    """Decorator to register a termination criterion."""

    def decorator(termination_fcn: Callable):
        _TERMINATION_FCN_REGISTRY[name] = termination_fcn
        return termination_fcn

    return decorator


class ArenaForage(rodent_base.RodentEnv):
    """Arena forage environment.

    The rodent must navigate a flat arena to collect treats (blue spheres).
    It observes a scalar scent signal (sum of Gaussians from active treats).
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        clips: Optional[Any] = None,
    ) -> None:
        del clips  # Arena forage task does not use reference clips
        super().__init__(config, config_overrides)

        self.add_rodent(
            torque_actuators=self._config.torque_actuators,
            rescale_factor=self._config.rescale_factor,
            pos=[0.0, 0.0, 0.0],
            quat=(1, 0, 0, 0),
        )

        # Add placeholder treat sphere geoms (hidden below floor initially)
        num_treats = self._config.num_treats
        treat_r = self._config.treat_sphere_radius
        for i in range(num_treats):
            self._spec.worldbody.add_geom(
                name=f"treat_{i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0, 0, -10],
                size=[treat_r],
                rgba=[0.2, 0.4, 1.0, 1.0],
                contype=0,
                conaffinity=0,
            )

        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, treat_rng = jax.random.split(rng)

        num_treats = self._config.num_treats
        spawn_radius = self._config.spawn_radius
        treat_height = self._config.treat_height

        # Jittered grid sampling: angular sectors + radial bands for even spread
        angle_rng, radius_rng, perm_rng = jax.random.split(treat_rng, 3)
        min_spawn_radius = self._config.min_spawn_radius

        # Angular: one treat per sector with random jitter
        sector_width = 2 * jp.pi / num_treats
        base_angles = jp.linspace(0, 2 * jp.pi, num_treats, endpoint=False)
        angle_jitter = jax.random.uniform(
            angle_rng, (num_treats,), minval=0, maxval=sector_width
        )
        angles = base_angles + angle_jitter

        # Radial: stratified annular bands, shuffled so radius/angle uncorrelated
        band_edges = jp.linspace(min_spawn_radius, spawn_radius, num_treats + 1)
        band_lo = band_edges[:-1]
        band_hi = band_edges[1:]
        band_order = jax.random.permutation(perm_rng, num_treats)
        band_lo = band_lo[band_order]
        band_hi = band_hi[band_order]
        u = jax.random.uniform(radius_rng, (num_treats,))
        radii = jp.sqrt(band_lo**2 + u * (band_hi**2 - band_lo**2))
        treat_positions = jp.stack(
            [
                radii * jp.cos(angles),
                radii * jp.sin(angles),
                jp.full((num_treats,), treat_height),
            ],
            axis=-1,
        )

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
            "treat_positions": treat_positions,
            "treat_collected": jp.zeros(num_treats, dtype=bool),
            "newly_collected": jp.zeros(num_treats, dtype=bool),
        }

        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Initialize prev_scent for potential-based shaping
        info["prev_scent"] = self._compute_scent(data, info)

        metrics = {}
        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action

        # Check treat collection
        skull_pos = self._get_skull_pos(data)
        treat_positions = info["treat_positions"]
        treat_collected = info["treat_collected"]
        treat_r = self._config.treat_sphere_radius

        dists = jp.linalg.norm(skull_pos - treat_positions, axis=-1)
        in_range = dists < treat_r
        newly_collected = jp.logical_and(in_range, jp.logical_not(treat_collected))
        treat_collected = jp.logical_or(treat_collected, in_range)
        info["treat_collected"] = treat_collected
        info["newly_collected"] = newly_collected

        obs = self._get_obs(data, info)
        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> collections.OrderedDict:
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        # Compute scent signal and gradient
        total_scent = self._compute_scent(data, info)
        scent_gradient = self._compute_scent_gradient(data, info)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
                total_scent.reshape(1),
                scent_gradient,
            ]
        )

        return collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )

    def _get_skull_pos(self, data: mjx.Data) -> jax.Array:
        """Get skull world position (3D)."""
        skull_body = data.bind(
            self.mjx_model, self._spec.body(f"skull{self._suffix}")
        )
        return skull_body.xpos

    def _compute_scent(self, data: mjx.Data, info: dict) -> jax.Array:
        """Compute total scent at skull position from all active treats."""
        skull_pos = self._get_skull_pos(data)
        treat_positions = info["treat_positions"]
        treat_collected = info["treat_collected"]
        sigma = self._config.scent_sigma

        dists = jp.linalg.norm(skull_pos - treat_positions, axis=-1)
        scent_per_treat = jp.exp(-dists**2 / (2 * sigma**2))
        active_mask = jp.logical_not(treat_collected).astype(float)
        total_scent = jp.sum(scent_per_treat * active_mask)
        return total_scent

    def _compute_scent_gradient(self, data: mjx.Data, info: dict) -> jax.Array:
        """Compute egocentric scent gradient at skull position (3D vector).

        The gradient of the sum-of-Gaussians scent field points toward
        uncollected treats, weighted by proximity. Returned in the torso
        body frame so the signal is orientation-invariant.
        """
        skull_pos = self._get_skull_pos(data)
        treat_positions = info["treat_positions"]
        treat_collected = info["treat_collected"]
        sigma = self._config.scent_sigma

        # diff[i] = treat_i - skull  (points toward treat i)
        diff = treat_positions - skull_pos  # (num_treats, 3)
        dists = jp.linalg.norm(diff, axis=-1)  # (num_treats,)
        scent_per_treat = jp.exp(-dists**2 / (2 * sigma**2))  # (num_treats,)
        active_mask = jp.logical_not(treat_collected).astype(float)

        # ∇S = Σ_i active_i * exp(-d_i²/2σ²) * (x_i - x) / σ²
        weights = scent_per_treat * active_mask / (sigma**2)  # (num_treats,)
        grad_world = jp.sum(weights[:, None] * diff, axis=0)  # (3,)

        # Transform to egocentric (torso) frame
        torso_body = data.bind(
            self.mjx_model, self._spec.body(f"torso{self._suffix}")
        )
        torso_frame = torso_body.xmat
        grad_ego = jp.dot(grad_world, torso_frame)
        return grad_ego

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics) -> bool:
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            terminated = termination_fcn(self, data, info, **kwargs)
            any_terminated = jp.logical_or(any_terminated, terminated)
            metrics["terminations/" + name] = jp.astype(terminated, float)
        metrics["terminations/any"] = jp.astype(any_terminated, float)
        return any_terminated

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict
    ) -> float:
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += _REWARD_FCN_REGISTRY[name](
                self, data, info, metrics, **kwargs
            )
        return net_reward

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["proprioception"])[0])

    @property
    def non_proprioceptive_obs_size(self) -> int:
        return self.observation_size - self.proprioceptive_obs_size

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        obs = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs)[0])

    @property
    def non_flattened_observation_size(self) -> mjx_env.ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return jax.tree_util.tree_map(lambda x: jp.prod(jp.array(x.shape)), obs)

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        render_scent: bool = True,
        scent_grid_extent: float = 5.0,
        scent_grid_resolution: int = 40,
        scent_threshold: float = 0,
    ) -> Sequence[np.ndarray]:
        """Renders a trajectory with treat spheres and optional scent tiles.

        Args:
            render_scent: If True, draw viridis-colored floor tiles showing
                the Gaussian scent field from active (uncollected) treats.
            scent_grid_extent: Half-size of the scent tile grid in meters.
            scent_grid_resolution: Number of tiles per axis.
            scent_threshold: Minimum scent intensity to draw a tile.
        """
        from matplotlib import cm as mpl_cm

        mj_model = self.mj_model
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        mj_data = mujoco.MjData(mj_model)

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        if camera is None:
            camera = -1

        num_treats = self._config.num_treats
        treat_geom_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"treat_{i}")
            for i in range(num_treats)
        ]

        # Pre-compute per-treat scent contributions for the tile grid.
        # Treat positions are fixed within one trajectory (set at reset).
        if render_scent:
            sigma = self._config.scent_sigma
            treat_pos_np = np.array(trajectory[0].info["treat_positions"])
            grid_1d = np.linspace(
                -scent_grid_extent, scent_grid_extent, scent_grid_resolution
            )
            gx, gy = np.meshgrid(grid_1d, grid_1d)
            # (num_treats, res, res)
            scent_per_treat = np.zeros(
                (num_treats, scent_grid_resolution, scent_grid_resolution)
            )
            for t in range(num_treats):
                dist2 = (gx - treat_pos_np[t, 0]) ** 2 + (
                    gy - treat_pos_np[t, 1]
                ) ** 2
                scent_per_treat[t] = np.exp(-dist2 / (2 * sigma**2))

            tile_half = scent_grid_extent / scent_grid_resolution
            tile_size = np.array([tile_half, tile_half, 0.0001], dtype=np.float64)
            tile_mat = np.eye(3, dtype=np.float64).flatten()

        rendered_frames = []
        for i, state in enumerate(trajectory):
            # Set treat geom positions in model before mj_forward so they
            # propagate into geom_xpos for worldbody geoms.
            treat_positions = np.array(state.info["treat_positions"])
            treat_collected = np.array(state.info["treat_collected"])

            for j in range(num_treats):
                if treat_collected[j]:
                    mj_model.geom_pos[treat_geom_ids[j]] = [0, 0, -10]
                else:
                    mj_model.geom_pos[treat_geom_ids[j]] = treat_positions[j]

            mj_data.qpos = np.array(state.data.qpos)
            mj_data.qvel = np.array(state.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)

            renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)

            # Inject scent floor tiles as visual-only scene geoms.
            if render_scent:
                active = ~treat_collected
                if active.any():
                    scent_grid = np.sum(scent_per_treat[active], axis=0)
                else:
                    scent_grid = np.zeros_like(gx)

                scene = renderer.scene
                for row in range(scent_grid_resolution):
                    for col in range(scent_grid_resolution):
                        intensity = scent_grid[row, col]
                        if intensity < scent_threshold:
                            continue
                        if scene.ngeom >= scene.maxgeom:
                            break
                        clamped = min(float(intensity), 1.0)
                        rgba = np.array(mpl_cm.viridis(clamped), dtype=np.float32)
                        rgba[3] = clamped * 0.6
                        pos = np.array(
                            [grid_1d[col], grid_1d[row], 0.0], dtype=np.float64
                        )
                        mujoco.mjv_initGeom(
                            scene.geoms[scene.ngeom],
                            mujoco.mjtGeom.mjGEOM_BOX,
                            tile_size,
                            pos,
                            tile_mat,
                            rgba,
                        )
                        scene.ngeom += 1

            if modify_scene_fns is not None:
                modify_scene_fns[i](renderer.scene)
            rendered_frames.append(renderer.render())

        return rendered_frames


# --- Reward Functions ---


@_named_reward("treat_collection")
def _treat_collection_reward(env, data, info, metrics, weight) -> float:
    """Sparse reward for collecting treats."""
    newly_collected = info.get("newly_collected", jp.zeros(env._config.num_treats, dtype=bool))
    reward_value = jp.sum(newly_collected.astype(float)) * env._config.reward_per_treat
    weighted_reward = reward_value * weight
    metrics["rewards/treat_collection"] = weighted_reward
    metrics["mean_collected_treats"] = jp.sum(newly_collected.astype(float))
    return weighted_reward


@_named_reward("scent_proximity")
def _scent_proximity_reward(env, data, info, metrics, weight) -> float:
    """Potential-based shaping reward for movement toward uncollected treats.

    Uses the change in scent between consecutive steps rather than the
    absolute scent value.  This rewards *approaching* treats without
    rewarding hovering, and avoids penalising collection (the one-time
    scent drop is dwarfed by the treat_collection reward).

    Formally: F(s, s') = γ · Φ(s') − Φ(s), with Φ = total_scent.
    """
    total_scent = env._compute_scent(data, info)
    prev_scent = info.get("prev_scent", total_scent)
    discounting = env._config.get("discounting", 0.99)
    shaping = discounting * total_scent - prev_scent
    info["prev_scent"] = total_scent
    weighted_reward = shaping * weight
    metrics["rewards/scent_proximity"] = weighted_reward
    return weighted_reward


# --- Termination Functions ---


@_named_termination_criterion("all_collected")
def _all_collected_termination(env, data, info) -> bool:
    """Terminate when all treats have been collected."""
    del data
    return jp.all(info["treat_collected"])


@_named_termination_criterion("nan_termination")
def _nan_termination(env, data, info) -> bool:
    """Check for NaN values in simulation data."""
    del info
    flattened_vals, _ = flatten_util.ravel_pytree(data)
    num_nans = jp.sum(jp.isnan(flattened_vals))
    return num_nans > 0
