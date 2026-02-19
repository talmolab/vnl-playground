"""Binocular vision-enabled run through corridor with gaps task.

Extends RunGap with stereo (left + right eye) camera observations.
Vision observations have shape (H, W, 2*C) — left and right eye images
concatenated along the channel dimension.

Rendering is handled by BinocularVisionRenderWrapper which creates two
JaxVisionRenderer instances (one per eye camera) and concatenates outputs.

The CNN architecture (shared Siamese vs independent dual-CNN) is configured
at the network level, not the task level. This task only defines the
observation shape.
"""

import collections
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from jax import flatten_util
from ml_collections import config_dict

from vnl_playground.tasks.rodent import run_gap


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the RunGapBinocularVision environment."""
    cfg = run_gap.default_config()
    cfg.mujoco_impl = "warp"
    cfg.vision = True
    cfg.vision_width = 32
    cfg.vision_height = 32
    cfg.grayscale = True
    cfg.binocular = True
    cfg.left_camera_name = "eye_left-rodent"
    cfg.right_camera_name = "eye_right-rodent"
    cfg.render_depth = False
    cfg.use_textures = False
    cfg.use_shadows = False
    return cfg


class RunGapBinocularVision(run_gap.RunGap):
    """RunGap with binocular (stereo) egocentric vision observations.

    Like RunGapVision, but uses two eye cameras (eye_left, eye_right)
    instead of a single egocentric camera. Vision observations have shape
    (H, W, 2*C) — left and right eye images concatenated along channels.

    Rendering is done by BinocularVisionRenderWrapper outside the vmapped
    env. This class provides zero-filled placeholders at the correct shape.
    """

    def __init__(
        self,
        rng=jax.random.PRNGKey(0),
        config=default_config(),
        config_overrides=None,
    ):
        super().__init__(rng=rng, config=config, config_overrides=config_overrides)

        if self._config.mujoco_impl != "warp":
            raise ValueError("RunGapBinocularVision requires mujoco_impl='warp'")

        self._vision_enabled = self._config.vision
        self._vision_width = self._config.vision_width
        self._vision_height = self._config.vision_height
        self._grayscale = self._config.get("grayscale", False)

    @property
    def vision_shape(self):
        """Shape of vision obs: (H, W, 2*C) for binocular."""
        mono_channels = 1 if self._grayscale else 3
        return (self._vision_height, self._vision_width, 2 * mono_channels)

    @property
    def vision_enabled(self):
        return self._vision_enabled

    def _get_obs(self, data, info):
        """Get observations with binocular vision placeholder.

        Same as RunGapVision._get_obs but vision placeholder has shape
        (H, W, 2*C) for the binocular channel-stacked format.
        """
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        task_obs = jp.concatenate([
            info["prev_action"],
            kinematic_sensors,
            touch_sensors,
            origin,
        ])

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
    def observation_size(self):
        """Total flat observation size for the MLP (excludes vision pixels)."""
        obs_size = self.non_flattened_observation_size
        total = 0
        for key in ("task_obs", "proprioception"):
            total += jp.sum(flatten_util.ravel_pytree(obs_size["state"][key])[0])
        return total

    @property
    def proprioceptive_obs_size(self):
        obs_size = self.non_flattened_observation_size
        return jp.sum(flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0])

    @property
    def vision_obs_size(self):
        h, w, c = self.vision_shape
        return h * w * c
