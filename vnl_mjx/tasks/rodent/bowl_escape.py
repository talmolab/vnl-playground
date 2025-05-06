# Bowl escape definition, reflecting the mujoco playgrounds.

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import dm_control

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

from vnl_mjx.tasks.rodent import base as rodent_base
from vnl_mjx.tasks.rodent import consts

import matplotlib.colors as mcolors


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=consts.RODENT_SPHERE_FEET_PATH,
        arena_xml_path=consts.ARENA_XML_PATH,
        ctrl_dt=0.02,
        sim_dt=0.01,
        mj_model_timestep=0.01,
        solver="cg",
        iterations=100,
        ls_iterations=50,
        vision=False,
        episode_length=1000,
        action_repeat=1,
        bowl_hsize=4,
        bowl_vsize=2,
        bowl_sigma=1,
        bowl_amplitude=-10.0,
    )


class BowlEscape(rodent_base.RodentEnv):
    """Bowl escape environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        """
        Initialize the BowlEscape class and set up the environment.

        Args:
            config (config_dict.ConfigDict, optional): configs for the bowl escape. Defaults to default_config().
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): overrides for the configuration. Defaults to None.

        Raises:
            NotImplementedError: Raised if vision is enabled.
        """
        # super has already init a spec with the provided arena xml path
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )
        self._initialize_noisy_bowl()
        self.add_rodent([0, 0, 0.2])
        self._spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        self.compile()

    def _initialize_noisy_bowl(self):
        """Initialize the noisy bowl with specified parameters."""
        self._spec = add_bowl_hfield(
            self._spec,
            hsize=self._config.bowl_hsize,
            vsize=self._config.bowl_vsize,
            sigma=self._config.bowl_sigma,
            amplitude=self._config.bowl_amplitude,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        metrics = {}
        info = {}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        obs = jp.concatenate([data.qpos, data.qvel])
        return obs

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

    def _escape_reward(self, data: mjx.Data):
        terrain_size = float(self._config.bowl_hsize)
        torso_xpos = data.bind(self.mjx_model, self._spec.body("torso-rodent")).xpos
        escape_reward = reward.tolerance(
            jp.linalg.norm(torso_xpos),
            bounds=(terrain_size, float("inf")),
            margin=terrain_size,
            value_at_margin=0,
            sigmoid="linear",
        )
        return escape_reward

    def _get_reward(self, data: mjx.Data) -> Dict[str, jax.Array]:
        escape_reward = self._escape_reward(data)
        upright_reward = self._upright_reward(data, deviation_angle=10)
        return {
            "escape_reward": escape_reward,
            "upright_reward": upright_reward,
            "escape * upright": escape_reward * upright_reward,
        }

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Apply the action to the model.
        data = mjx_env.step(self.mjx_model, state.data, action)
        # Get the new observation.
        obs = self._get_obs(data)
        # Compute the reward.
        rewards = self._get_reward(data)
        reward = rewards["escape * upright"]
        state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
        )
        return state


### Perlin noise generator, for height field generation
# adapted from https://github.com/pvigier/perlin-numpy


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(shape, res, tileable=(False, False), interpolant=interpolant):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = jp.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
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


def gaussian_bowl(shape, sigma=0.5, amplitude=-5.0):
    """
    Generate a Gaussian bowl shape.

    Args:
        shape: Tuple of two ints representing the shape of the generated array.
        sigma: Standard deviation of the Gaussian.
        amplitude: Amplitude of the Gaussian.

    Returns:
        A numpy array of shape `shape` with the Gaussian bowl.
    """
    y = np.linspace(-1, 1, shape[0])
    x = np.linspace(-1, 1, shape[1])
    xv, yv = np.meshgrid(x, y)
    return amplitude * np.exp(-(xv**2 + yv**2) / (2 * sigma**2))


def add_bowl_hfield(spec=None, hsize=10, vsize=4, sigma=0.5, amplitude=-5.0):
    """
    add height field with noise

    Args:
        spec (_type_, optional): mj_spec that we attach the bowl to. Defaults to None.
        hsize (int, optional): horizontal size of the bowl. Defaults to 10.
        vsize (int, optional): vertical depth of the bowl. Defaults to 4.
        sigma (float, optional): standard deviation of the Gaussian of the noise. Defaults to 0.5.
        amplitude (float, optional): amplitude of the Gaussian. Defaults to -5.0.

    Returns:
        mj_spec: spec with the height bowl
    """

    # Initialize spec
    if spec is None:
        spec = mujoco.MjSpec()

    # Generate Perlin noise
    size = 128
    noise = perlin((size, size), (8, 8)) * 2

    # Remap noise to 0 to 1
    noise = (noise + 1) / 2

    bowl = gaussian_bowl(noise.shape, sigma=sigma, amplitude=amplitude)

    # Add the Gaussian bowl to the Perlin noise height field.
    noise = noise + bowl

    noise -= np.min(noise)
    noise /= np.max(noise)

    # Create height field
    hfield = spec.add_hfield(
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

    return spec
