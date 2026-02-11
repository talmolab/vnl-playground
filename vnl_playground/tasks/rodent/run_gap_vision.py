"""Vision-enabled run through corridor with gaps task.

Extends RunGap with egocentric camera observations rendered via
mujoco_warp's GPU batch ray-tracer.

The observation dict from _get_obs() returns the same proprioceptive
observations as RunGap. Vision observations are added externally by
the training loop which handles the mujoco_warp rendering pipeline:

    1. env.step() (physics via MJX/Warp)
    2. renderer.sync_state(state.data)
    3. rgb, depth = renderer.render()
    4. obs["vision"] = rgb / 255.0

See VisionRenderer in vision.py for the rendering wrapper.
"""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict

from vnl_playground.tasks.rodent import run_gap


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the RunGapVision environment."""
    cfg = run_gap.default_config()
    cfg.mujoco_impl = "warp"  # Required for mujoco_warp rendering
    # Vision parameters
    cfg.vision = True
    cfg.vision_width = 64
    cfg.vision_height = 64
    cfg.vision_camera_name = "egocentric-rodent"
    cfg.render_depth = False
    return cfg


class RunGapVision(run_gap.RunGap):
    """RunGap with egocentric vision observations.

    Adds vision configuration and metadata to RunGap. The actual
    rendering is performed by VisionRenderer in the training loop,
    not inside step().

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

    @property
    def vision_shape(self):
        """Shape of the vision observation: (H, W, 3)."""
        return (self._vision_height, self._vision_width, 3)

    @property
    def vision_enabled(self):
        """Whether vision observations are enabled."""
        return self._vision_enabled
