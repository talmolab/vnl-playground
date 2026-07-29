# Bowl escape definition, reflecting the mujoco playgrounds.
import collections
from collections.abc import Callable
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jp
import matplotlib.colors as mcolors
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_fns
from scipy.spatial.transform import Rotation

from vnl_playground.tasks.reward_registry import RewardRegistry
from vnl_playground.tasks.rodent import base as rodent_base
from vnl_playground.tasks.rodent import consts

_registry = RewardRegistry()


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=512,
        render_width=64,
        render_height=64,
        enabled_geom_groups=[0, 1, 2],
        use_rasterizer=False,
        history=3,
    )


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the BowlEscape environment.

    Since the resulting noise will be normalized by the bowl_vsize, the amplitude
    and the sigma interacting with each other.

    The configuration dictionary contains the following keys:
        walker_xml_path (str): Path to the XML file for the rodent walker.
        arena_xml_path (str): Path to the XML file for the arena.
        ctrl_dt (float): Control timestep.
        sim_dt (float): Simulation timestep.
        solver (str): Solver type.
        iterations (int): Number of solver iterations.
        ls_iterations (int): Number of line search iterations.
        vision (bool): Whether to enable vision.
        episode_length (int): Length of an episode.
        action_repeat (int): Number of times to repeat an action.
        bowl_hsize (int): Horizontal size of the bowl.
        bowl_vsize (int): Vertical size (depth) of the bowl.
        bowl_sigma (float): Standard deviation of the Gaussian bump on bowl, usually is the bowl_hsize / 4.
        bowl_amplitude (float): Amplitude of the Gaussian bump on bowl,

    Returns:
        config_dict.ConfigDict: The default configuration dictionary.
    """
    return config_dict.create(
        walker_xml_path=consts.RODENT_BOX_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.01,
        sim_dt=0.002,
        solver="newton",
        mujoco_impl="jax",
        naconmax=90 * 1024,
        njmax=1200,
        iterations=5,
        ls_iterations=5,
        noslip_iterations=0,
        vision=False,
        vision_config=default_vision_config(),
        torque_actuators=True,
        rescale_factor=0.9,
        target_speed=1.0,
        episode_length=2000,
        action_repeat=1,  # is this action repeat based on sim dit or control dt?
        bowl_hsize=2,
        bowl_vsize=0.2,
        bowl_sigma=1.25,
        bowl_amplitude=-10,
        reward_terms={
            "escape_x_upright": {"weight": 1.0},
            "speed": {"weight": 0.0},
        },
        termination_criteria={
            "fallen": {},
            "nan_termination": {},
        },
    )


class BowlEscape(rodent_base.RodentEnv):
    """Bowl escape environment."""

    _registry = _registry

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict[str, str | int | list[Any]] | None = None,
    ) -> None:
        """
        Initialize the BowlEscape class and set up the environment.

        Args:
            rng (jax.Array, optional): Random number generator key for reproducible randomness.
            config (config_dict.ConfigDict, optional): configs for the bowl escape. Defaults to default_config().
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): overrides for the configuration. Defaults to None.

        Raises:
            NotImplementedError: Raised if vision is enabled.
        """
        # super has already init a spec with the provided arena xml path
        super().__init__(config, config_overrides)
        self._rng = rng
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )
        self._vision = self._config.vision
        self._initialize_noisy_bowl(self._rng)
        # Compute rodent initial pose on bowl
        init_x, init_y = 0.0, 0.0
        init_z = self._interpolate_bowl_height(init_x, init_y) + 0.01
        # init_quat = self._surface_quaternion(init_x, init_y)
        print(f"Initial position: {init_x}, {init_y}, {init_z}")
        # print(f"Initial quaternion: {init_quat}")
        self.add_rodent(
            self._config.torque_actuators,
            self._config.rescale_factor,
            [init_x, init_y, init_z],
            # init_quat,
        )
        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

        if self._vision:
            raise NotImplementedError("Vision is not implemented for BowlEscape.")

    def _initialize_noisy_bowl(self, rng: jax.Array) -> None:
        """Initialize the noisy bowl heightfield and store it in the environment."""
        self._rng, bowl_rng = jax.random.split(rng)
        self._spec, bowl_noise = add_bowl_hfield(
            bowl_rng,
            self._spec,
            hsize=self._config.bowl_hsize,
            vsize=self._config.bowl_vsize,
            sigma=self._config.bowl_sigma,
            amplitude=self._config.bowl_amplitude,
        )
        # Store as JAX array for fast indexing in termination check
        self._bowl_noise = jp.array(bowl_noise)
        self._bowl_noise_np = bowl_noise

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment state, with the already constructed
        mj_model, which persists the same bowl shape.

        Args:
            rng (jax.Array): Random number generator state.

        Returns:
            mjx_env.State: The initial environment state after reset.
        """
        if self._vision:
            raise NotImplementedError("Vision is not implemented for BowlEscape.")

        info = {
            "prev_action": self.null_action(),
            "action": self.null_action(),
        }
        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        metrics = {}

        obs = self._get_obs(data, info)
        reward = self._get_reward(data, info, metrics)
        done = self._is_done(data, info, metrics)

        return mjx_env.State(data, obs, reward, jp.astype(done, float), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward by one timestep.

        Args:
            state (mjx_env.State): Current environment state.
            action (jax.Array): Action to apply.

        Returns:
            mjx_env.State: The new environment state after stepping.
        """
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        info["prev_action"] = info["action"]
        info["action"] = action

        obs = self._get_obs(data, info)

        done = self._is_done(data, info, state.metrics)
        reward = self._get_reward(data, info, state.metrics)
        # Handle nans during sim with this in addition to termination on nans in data
        reward = jp.nan_to_num(reward)

        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> collections.OrderedDict:
        """Get the current observation from the simulation data.

        Args:
            data (mjx.Data): The simulation data.
            info: State info dictionary.

        Returns:
            OrderedDict with state and privileged_state keys.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)
        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
            ]
        )

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=self._get_proprioception(data, info, flatten=False),
        )
        return collections.OrderedDict(state=obs)

    # Rewards
    @_registry.reward("escape_x_upright")
    def _escape_x_upright_reward(self, data, info, metrics, weight) -> float:
        """Calculate escape reward based on torso position relative to terrain size.

        Args:
            data (mjx.Data): The simulation data.

        Returns:
            float: Escape reward value.
        """
        del info
        terrain_size = float(self._config.bowl_hsize)
        torso_xpos = data.bind(
            self.mjx_model, self._spec.body(f"torso{self._suffix}")
        ).xpos
        escape_reward = reward_fns.tolerance(
            jp.linalg.norm(torso_xpos),
            bounds=(terrain_size, float("inf")),
            margin=terrain_size,
            value_at_margin=0,
            sigmoid="linear",
        )

        deviation_angle = 30
        deviation = np.cos(np.deg2rad(deviation_angle))
        # xmat is the 3x3 rotation matrix of the current frame
        upright_torso = data.bind(
            self.mjx_model, self._spec.body(f"torso{self._suffix}")
        ).xmat[-1, -1]
        upright_head = data.bind(
            self.mjx_model, self._spec.body(f"skull{self._suffix}")
        ).xmat[-1, -1]
        upright = reward_fns.tolerance(
            jp.stack([upright_torso, upright_head]),
            bounds=(deviation, np.inf),
            sigmoid="linear",
            margin=1 + deviation,
            value_at_margin=0,
        )
        upright_reward = np.min(upright)
        metrics["rewards/upright"] = upright_reward
        metrics["rewards/escape"] = escape_reward

        reward = escape_reward * upright_reward * weight
        metrics["rewards/escape_x_upright"] = reward
        return reward

    @_registry.reward("speed")
    def _speed_reward(self, data, info, metrics, weight) -> float:
        del info
        body = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        vel = jp.linalg.norm(body.subtree_linvel)
        target_speed = self._config.target_speed
        reward = (
            reward_fns.tolerance(
                vel,
                bounds=(target_speed, target_speed),
                margin=target_speed,
                sigmoid="linear",
                value_at_margin=0.0,
            )
            * weight
        )
        metrics["rewards/speed"] = reward
        return reward

    # Termination
    @_registry.termination("fallen")
    def _torso_too_low(self, data: mjx.Data, info) -> bool:
        """Check if the episode should terminate based on torso position relative to bowl surface.

        Args:
            data (mjx.Data): The simulation data.

        Returns:
            jp.ndarray: 1.0 if torso is below the bowl surface height, else 0.0.
        """
        del info
        # Torso (root) position
        torso_pos = data.bind(
            self.mjx_model, self._spec.body(f"torso{self._suffix}")
        ).xpos
        x, y, z = torso_pos

        # fetch bowl surface height at torso (x, y)
        height_z = self._interpolate_bowl_height(x, y)
        # make stricter by adding a small threshold
        return z <= height_z + 0.03

    @_registry.termination("nan_termination")
    def _nan_termination(self, data, info) -> bool:
        return jp.any(jp.isnan(data.qpos))

    def null_action(self) -> jp.ndarray:
        return jp.zeros(self.action_size)

    def _interpolate_bowl_height(self, x: float, y: float) -> float:
        """Interpolate the bowl surface height at world coordinates (x, y).

        Args:
            x (float): x-coordinate in world frame.
            y (float): y-coordinate in world frame.

        Returns:
            float: z-height of the bowl surface at (x, y) based on the noisy height field.
        """
        hsize = float(self._config.bowl_hsize)
        vsize = float(self._config.bowl_vsize)
        noise = self._bowl_noise_np
        size = noise.shape[0]
        # normalized texture coords in [0,1)
        u = (x + hsize) / (2 * hsize)
        v = (y + hsize) / (2 * hsize)
        # Use JAX floor and astype for col, row
        col = jp.floor(u * (size - 1)).astype(jp.int32)
        row = jp.floor(v * (size - 1)).astype(jp.int32)
        height_norm = noise[row, col]
        return height_norm * vsize

    def _compute_surface_normal(self, x: float, y: float) -> np.ndarray:
        """Compute the surface normal of the bowl heightfield at (x, y).

        Args:
            x (float): x-coordinate in world frame.
            y (float): y-coordinate in world frame.

        Returns:
            np.ndarray: The normalized surface normal vector at (x, y).
        """
        noise = self._bowl_noise_np
        hsize = float(self._config.bowl_hsize)
        size = noise.shape[0]
        # world-to-grid spacing
        dx = (2 * hsize) / (size - 1)
        # texture coords
        u = (x + hsize) / (2 * hsize)
        v = (y + hsize) / (2 * hsize)
        i = int(np.clip(v * (size - 1), 0, size - 1))
        j = int(np.clip(u * (size - 1), 0, size - 1))

        # finite differences
        def get(i_, j_):
            return noise[np.clip(i_, 0, size - 1), np.clip(j_, 0, size - 1)]

        dzdx = (get(i, j + 1) - get(i, j - 1)) / (2 * dx)
        dzdy = (get(i + 1, j) - get(i - 1, j)) / (2 * dx)
        # tangents and normal
        tx = np.array([1.0, 0.0, dzdx])
        ty = np.array([0.0, 1.0, dzdy])
        normal = np.cross(tx, ty)
        return normal / np.linalg.norm(normal)

    def _surface_quaternion(self, x: float, y: float) -> list[float]:
        """Return a quaternion [w, x, y, z] aligning +z to the surface normal.

        Args:
            x (float): x-coordinate in world frame.
            y (float): y-coordinate in world frame.

        Returns:
            list[float]: Quaternion representing rotation aligning +z axis to surface normal.
        """
        normal = self._compute_surface_normal(x, y)
        up = np.array([0.0, 0.0, 1.0])
        # axis-angle rotation from up to normal
        axis = np.cross(up, normal)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            # aligned or opposite
            return [1.0, 0.0, 0.0, 0.0] if normal[2] > 0 else [0.0, 1.0, 0.0, 0.0]
        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(up, normal), -1.0, 1.0))
        rot = Rotation.from_rotvec(axis * angle)
        q = rot.as_quat()  # [x, y, z, w]
        # convert to [w, x, y, z]
        return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


class BowlEscapeRender(BowlEscape):
    """Bowl escape environment with rendering capabilities."""

    def __init__(
        self,
        num_rodents: int = 1,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: dict[str, str | int | list[Any]] | None = None,
        line_coords: list[tuple[float, float, float]] | None = None,
        line_radius: float = 0.002,
    ) -> None:
        """Initialize the BowlEscapeRender class with rendering capabilities.

        Args:
            num_rodents (int, optional): Number of rodents. Defaults to 1.
            rng (jax.Array, optional): Random number generator key for reproducible randomness.
            config (config_dict.ConfigDict, optional): Configuration for the environment. Defaults to default_config().
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Overrides for the configuration. Defaults to None.
            line_coords (Optional[list[tuple[float, float, float]]], optional): List of 3D coordinates for line plotting. Defaults to None.
            line_radius (float, optional): Radius for plotted line spheres. Defaults to 0.02.
        """
        # super has already init a spec with the provided arena xml path
        rodent_base.RodentEnv.__init__(self, config, config_overrides)
        # Save init parameters for later reinitialization
        self._init_num_rodents = num_rodents
        self._init_rng = rng
        self._init_config = config
        self._init_config_overrides = config_overrides
        self._init_line_coords = line_coords
        self._init_line_radius = line_radius
        self._rng = rng
        self.line_coords = line_coords
        self.line_radius = line_radius
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )
        self._vision = self._config.vision
        self._initialize_noisy_bowl(self._rng)
        init_x, init_y = 0.0, 0.0
        for i in range(num_rodents):
            init_z = self._interpolate_bowl_height(init_x, init_y) + 0.01
            print(f"Initial position: {init_x}, {init_y}, {init_z}")
            self.add_rodent(
                self._config.torque_actuators,
                self._config.rescale_factor,
                [init_x, init_y, init_z],
                # init_quat,
                suffix=f"-rodent-{i}",
            )
            init_x += 0.1  # offset each rodent slightly in x direction
            init_y += 0.1  # offset each rodent slightly in y direction

        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()
        # record baseline geom count before adding line geoms
        self._base_geom_count = len(self._spec.worldbody.geoms)

    def add_line_geoms(
        self, line_coords: list[tuple[float, float, float]] | None = None
    ) -> None:
        """Add sphere geoms for each coordinate in self.line_coords."""
        if line_coords is None:
            line_coords = self.line_coords
        for i, (x, y, z) in enumerate(line_coords):
            self._spec.worldbody.add_geom(
                name=f"line_sphere_{i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.line_radius, self.line_radius, self.line_radius],
                pos=[x, y, z],
                rgba=[0.0, 0.0, 1.0, 0.5],
                contype=0,
                conaffinity=0,
            )
        # recompile to register new geoms
        self.compile(forced=True)

    def remove_line_geoms(self) -> None:
        """Remove all geoms added after initialization by reinitializing with saved parameters."""
        # Reinitialize environment to reset spec and remove line geoms
        self.__init__(
            num_rodents=self._init_num_rodents,
            rng=self._init_rng,
            config=self._init_config,
            config_overrides=self._init_config_overrides,
            line_coords=self._init_line_coords,
            line_radius=self._init_line_radius,
        )

    def update_line_geoms(self, new_coords: list[tuple[float, float, float]]) -> None:
        """Remove old geoms and add spheres at new coordinates."""
        self.remove_line_geoms()
        self.line_coords = new_coords
        self.add_line_geoms()


### Perlin noise generator, for height field generation
# adapted from https://github.com/pvigier/perlin-numpy


def interpolant(t: jp.ndarray) -> jp.ndarray:
    """Interpolation function used in Perlin noise generation.

    Args:
        t (jp.ndarray): Input array.

    Returns:
        jp.ndarray: Interpolated output.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(
    rng: jax.Array,
    shape: tuple[int, int],
    res: tuple[int, int],
    tileable: tuple[bool, bool] = (False, False),
    interpolant: Callable[[jp.ndarray], jp.ndarray] = interpolant,
) -> np.ndarray:
    """Generate a 2D numpy array of Perlin noise.

    Args:
        rng (jax.Array): JAX random number generator key.
        shape (Tuple[int, int]): The shape of the generated array. Must be a multiple of res.
        res (Tuple[int, int]): Number of periods of noise along each axis.
        tileable (Tuple[bool, bool], optional): Whether noise should be tileable along each axis. Defaults to (False, False).
        interpolant (Callable[[jp.ndarray], jp.ndarray], optional): Interpolation function. Defaults to the default interpolant.

    Returns:
        np.ndarray: Generated 2D Perlin noise array.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = jp.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.array(jax.random.uniform(rng, (res[0] + 1, res[1] + 1)))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def gaussian_bowl(
    shape: tuple[int, int], sigma: float = 0.5, amplitude: float = -5.0
) -> np.ndarray:
    """Generate a Gaussian bowl shape.

    Args:
        shape (Tuple[int, int]): Shape of the generated array.
        sigma (float, optional): Standard deviation of the Gaussian. Defaults to 0.5.
        amplitude (float, optional): Amplitude of the Gaussian. Defaults to -5.0.

    Returns:
        np.ndarray: Gaussian bowl array of given shape.
    """
    y = np.linspace(-1, 1, shape[0])
    x = np.linspace(-1, 1, shape[1])
    xv, yv = np.meshgrid(x, y)
    return amplitude * np.exp(-(xv**2 + yv**2) / (2 * sigma**2))


def add_bowl_hfield(
    rng: jax.Array,
    spec: mujoco.MjSpec | None = None,
    hsize: float = 10,
    vsize: float = 4,
    sigma: float = 0.5,
    amplitude: float = -5.0,
) -> tuple[mujoco.MjSpec, np.ndarray]:
    """Add a noisy bowl height field to the Mujoco spec.

    Args:
        rng (jax.Array): JAX random number generator key.
        spec (Optional[mujoco.MjSpec], optional): Mujoco specification to modify. If None, a new spec is created. Defaults to None.
        hsize (float, optional): Horizontal size of the bowl. Defaults to 10.
        vsize (float, optional): Vertical depth of the bowl. Defaults to 4.
        sigma (float, optional): Standard deviation of the Gaussian bump. Defaults to 0.5.
        amplitude (float, optional): Amplitude of the Gaussian bump. Defaults to -5.0.

    Returns:
        Tuple[mujoco.MjSpec, np.ndarray]: The modified spec and the height-field noise array.
    """

    # Initialize spec
    if spec is None:
        spec = mujoco.MjSpec()

    # Generate Perlin noise
    size = 128
    rng, perlin_rng = jax.random.split(rng)
    noise = perlin(perlin_rng, (size, size), (8, 8)) * 2

    # Remap noise to 0 to 1
    noise = (noise + 1) / 2

    bowl = gaussian_bowl(noise.shape, sigma=sigma, amplitude=amplitude)

    # Add the Gaussian bowl to the Perlin noise height field.
    noise = noise + bowl

    # Smoothly blend central region to avoid bumps
    inner_radius = 0.05 * size  # fraction of grid for fully smooth bowl
    outer_radius = 0.25 * size  # fraction of grid where noise resumes
    center = size // 2
    y, x = np.ogrid[:size, :size]
    # distance from center in grid units
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    # blend weight: 0 inside inner, 1 outside outer, smoothstep between
    w = np.clip((dist - inner_radius) / (outer_radius - inner_radius), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    # combine pure Gaussian bowl and noisy bowl heights
    noise = bowl * (1.0 - w) + noise * w

    noise -= np.min(noise)
    noise /= np.max(noise)

    # Create height field
    spec.add_hfield(
        name="hfield",
        size=[hsize, hsize, vsize, vsize],
        nrow=noise.shape[0],
        ncol=noise.shape[1],
        userdata=noise.flatten(),
    )

    # Add texture
    texture = spec.add_texture(
        name="contours", type=mujoco.mjtTexture.mjTEXTURE_2D, width=128, height=128
    )

    # Create texture map, assign to texture
    h = noise
    s = 0.7 * np.ones(h.shape)
    v = 0.7 * np.ones(h.shape)
    hsv = np.stack([h, s, v], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb = np.flipud((rgb * 255).astype(np.uint8))
    texture.data = rgb.tobytes()

    # Assign texture to material
    grid = spec.add_material(name="contours")
    grid.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "contours"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="hfield")

    return spec, noise
