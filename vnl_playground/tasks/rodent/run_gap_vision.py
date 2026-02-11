"""Vision-enabled run through corridor with gaps task.

Extends RunGap with egocentric camera observations rendered via
mujoco_warp's GPU batch ray-tracer.

The observation dict from _get_obs() returns:

    {
        "proprioception": OrderedDict(...),       # nested dict, flattened by observation_utils
        "vision": jp.zeros(H, W, C),             # placeholder; actual pixels injected by renderer
    }

Vision observations are added externally by the training loop which handles
the mujoco_warp rendering pipeline:

    1. env.step() (physics via MJX/Warp)
    2. renderer.sync_state(state.data)
    3. rgb, depth = renderer.render()
    4. obs["vision"] = rgb / 255.0

See VisionRenderer in vision.py for the rendering wrapper.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from jax import flatten_util
from ml_collections import config_dict

from vnl_playground.tasks.rodent import run_gap


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the RunGapVision environment."""
    cfg = run_gap.default_config()
    cfg.mujoco_impl = "warp"  # Required for mujoco_warp rendering
    # Vision parameters
    cfg.vision = True
    cfg.vision_width = 32
    cfg.vision_height = 32
    cfg.grayscale = True
    cfg.vision_camera_name = "egocentric-rodent"
    cfg.render_depth = False
    return cfg


class RunGapVision(run_gap.RunGap):
    """RunGap with egocentric vision observations.

    Adds vision configuration and metadata to RunGap. The actual
    rendering is performed by VisionRenderer in the training loop,
    not inside step().

    Observations are returned as a dict with keys: proprioception, vision.
    Compatible with track-mjx's ff_ppo observation_utils.

    IMPORTANT: This task requires:
    - mujoco_impl="warp" for mujoco_warp rendering compatibility
    - A custom training loop that handles rendering (see vision.py)
    - Standard brax/track-mjx PPO loops will NOT work without adaptation
    """

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(rng=rng, config=config, config_overrides=config_overrides)

        if self._config.mujoco_impl != "warp":
            raise ValueError("RunGapVision requires mujoco_impl='warp' for rendering")

        self._vision_enabled = self._config.vision
        self._vision_width = self._config.vision_width
        self._vision_height = self._config.vision_height
        self._grayscale = self._config.get("grayscale", False)

    @property
    def vision_shape(self):
        """Shape of the vision observation: (H, W, C) where C=1 if grayscale else 3."""
        channels = 1 if self._config.get("grayscale", False) else 3
        return (self._vision_height, self._vision_width, channels)

    @property
    def vision_enabled(self):
        """Whether vision observations are enabled."""
        return self._vision_enabled

    def _get_obs(self, data, info) -> dict:
        """Get observations in ff_ppo-compatible dict format.

        Returns a dict with keys:

        - proprioception: nested OrderedDict from _get_proprioception
          (joint_angles, joint_ang_vels, etc.). Flattened by
          observation_utils._flatten_nested_obs at normalization time.
        - vision: zeros placeholder with shape (H, W, C). Actual pixels
          are injected by the training loop after rendering.

        Args:
            data: The simulation data (mjx.Data).
            info: State info dictionary.

        Returns:
            Dict with proprioception and vision keys.
        """
        return {
            "proprioception": self._get_proprioception(data, info, flatten=False),
            "vision": jp.zeros(self.vision_shape),
        }

    @property
    def observation_size(self) -> int:
        """Total flat observation size for the MLP (excludes vision pixels).

        Vision pixels are handled separately by the CNN and are NOT
        included here.

        Returns:
            int: Number of scalar observations fed to the MLP.
        """
        return self.proprioceptive_obs_size

    @property
    def proprioceptive_obs_size(self) -> int:
        """Flat size of the proprioceptive observation component.

        Computes the total number of scalars when the nested OrderedDict
        from _get_proprioception(flatten=False) is flattened.

        Returns:
            int: Number of proprioception scalars.
        """
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["proprioception"])[0])

    @property
    def vision_obs_size(self) -> int:
        """Total number of pixels in the vision observation (H * W * C).

        Returns:
            int: Product of vision shape dimensions.
        """
        h, w, c = self.vision_shape
        return h * w * c
