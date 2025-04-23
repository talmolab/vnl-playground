# Bowl escape definition, reflecting the mujoco playgrounds.

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

import matplotlib.colors as mcolors

rodent_xml_path = epath.Path(__file__) / "assets" / "rodent_only.xml"

arena_xml = """
<mujoco>
  <visual>
    <headlight diffuse=".5 .5 .5" specular="1 1 1"/>
    <global offwidth="2048" offheight="1536"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="10" height="10"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="400" height="400"/>
    <material name="groundplane" texture="groundplane" texrepeat="45 45" reflectance="0"/>
  </asset>

  <worldbody>
    <geom name="floor" size="150 150 0.1" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
     )
    
class BowlEscape(mjx_env.MjxEnv):
    """Bowl escape environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )
        # TODO: inject mj_spec and initialize the environment here
        self._xml_path = rodent_xml_path.as_posix()
        self._spec = mujoco.MjSpec.from_xml_string(arena_xml)
        
    def _initialize_noisy_bowl(self, )
        


### Perlin noise generator, for height field generation
# adapted from https://github.com/pvigier/perlin-numpy

def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def perlin(shape, res, tileable=(False, False), interpolant=interpolant):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
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

def add_hfield(spec=None, hsize=10, vsize=4, sigma=0.5, amplitude=-5.0):
    """Function that adds a heighfield with countours"""

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