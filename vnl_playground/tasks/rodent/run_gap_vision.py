"""Vision-enabled run through corridor with gaps task.

Extends RunGap with egocentric camera observations rendered via the
JAX-callable mujoco_warp GPU ray-tracer.  Supports both monocular
(single camera) and binocular (stereo left/right eye) modes via the
``config.binocular`` flag.

The observation dict from _get_obs() returns::

    {
        "state": OrderedDict(
            task_obs=...,                       # body-state signals (flat)
            proprioception=OrderedDict(...),     # nested dict
            vision=jp.zeros(H, W, C),           # placeholder zeros
        ),
        "privileged_state": OrderedDict(...),   # same as state
    }

In monocular mode the vision placeholder has shape (H, W, C) where C is 1
(grayscale) or 3 (RGB).  In binocular mode the shape is (H, W, 2*C) — left
and right eye images concatenated along the channel dimension.

``task_obs`` contains body-state signals (prev_action, kinematic sensors,
touch sensors, origin) rather than hand-crafted gap features — gap
information should be inferred from vision.

Vision rendering happens in ``VisionRenderWrapper`` (monocular) or
``BinocularVisionRenderWrapper`` (binocular), which wrap the vmapped/batched
env and render on all worlds at once using the JAX-callable warp renderer.
The wrapper replaces the zero placeholders with real rendered images.

Compatible with ``HighLevelWrapper`` for transfer learning pipelines
and direct ff_ppo training.  Downstream ``observation_utils.flatten_obs_dict()``
maps ``task_obs`` to ``imitation_target`` internally.

Usage::

    env = RunGapVision(config=cfg)
    brax_env = wrap_for_brax_training(env, ...)
    vision_env = VisionRenderWrapper(brax_env, env.mj_model, nworld=num_envs,
                                      **vision_config)
"""

import collections
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
    cfg.use_textures = False
    cfg.use_shadows = False
    # Binocular parameters (disabled by default = monocular mode)
    cfg.binocular = False
    cfg.left_camera_name = "eye_left-rodent"
    cfg.right_camera_name = "eye_right-rodent"
    return cfg


class RunGapVision(run_gap.RunGap):
    """RunGap with egocentric vision observations (monocular or binocular).

    Observations are returned with state/privileged_state wrapping containing
    task_obs, proprioception, and vision keys. The vision key contains
    placeholder zeros -- real rendering is handled by ``VisionRenderWrapper``
    (monocular) or ``BinocularVisionRenderWrapper`` (binocular) which wraps
    the batched env.

    When ``config.binocular`` is False (default), vision shape is (H, W, C).
    When ``config.binocular`` is True, vision shape is (H, W, 2*C) for stereo
    left/right eye images concatenated along the channel dimension.

    ``task_obs`` provides body-state signals (prev_action, kinematic sensors,
    touch sensors, origin). Gap information is not included -- the agent should
    infer it from vision.

    Compatible with both track-mjx's ff_ppo observation_utils and
    ``HighLevelWrapper`` for transfer learning pipelines.
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
        """Shape of the vision observation: (H, W, C) or (H, W, 2*C) for binocular."""
        mono_channels = 1 if self._grayscale else 3
        channels = 2 * mono_channels if self._config.get("binocular", False) else mono_channels
        return (self._vision_height, self._vision_width, channels)

    @property
    def vision_enabled(self):
        """Whether vision observations are enabled."""
        return self._vision_enabled

    def _get_obs(self, data, info) -> collections.OrderedDict:
        """Get observations in state/privileged_state dict format.

        Returns an OrderedDict with state and privileged_state keys, each
        containing:

        - task_obs: Body-state signals (prev_action, kinematic sensors, touch
          sensors, origin). Downstream ``observation_utils.flatten_obs_dict()``
          maps this key to ``imitation_target`` internally.
        - proprioception: Nested OrderedDict of body state sensors. Used by
          the decoder in transfer learning.
        - vision: Zeros placeholder with shape (H, W, C). Real pixels are
          injected by VisionRenderWrapper after batched rendering.

        Gap features are intentionally excluded — the agent should infer gap
        information from vision.

        This structure is compatible with both ``HighLevelWrapper`` (for
        transfer learning) and direct ff_ppo training.

        Args:
            data: The simulation data (mjx.Data).
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
            vision=jp.zeros(self.vision_shape),
        )
        return collections.OrderedDict(
            state=obs,
            privileged_state=obs,
        )

    @property
    def observation_size(self) -> int:
        """Total flat observation size for the MLP (excludes vision pixels)."""
        obs_size = self.non_flattened_observation_size
        total = 0
        for key in ("task_obs", "proprioception"):
            total += jp.sum(flatten_util.ravel_pytree(obs_size["state"][key])[0])
        return total

    @property
    def proprioceptive_obs_size(self) -> int:
        """Flat size of the proprioceptive observation component."""
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0])

    @property
    def vision_obs_size(self) -> int:
        """Total number of pixels in the vision observation (H * W * C)."""
        h, w, c = self.vision_shape
        return h * w * c
