"""Vision-enabled go-to-target task.

Extends GoToTarget with egocentric camera observations rendered via the
JAX-callable mujoco_warp GPU ray-tracer.

Vision rendering happens in VisionRenderWrapper (vision_jax.py), which
replaces the zero placeholders with real rendered images.
"""

import collections
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from jax import flatten_util
from ml_collections import config_dict

from vnl_playground.tasks.rodent import go_to_target


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for GoToTargetVision."""
    cfg = go_to_target.default_config()
    cfg.mujoco_impl = "warp"
    cfg.vision = True
    cfg.vision_width = 64
    cfg.vision_height = 64
    cfg.grayscale = True
    cfg.vision_camera_name = "egocentric-rodent"
    cfg.render_depth = False
    cfg.use_textures = False
    cfg.use_shadows = False
    return cfg


class GoToTargetVision(go_to_target.GoToTarget):
    """GoToTarget with egocentric vision observations.

    task_obs = [prev_action, sensors, touch, origin, ego_target]
    vision = zeros(H, W, C) placeholder (filled by VisionRenderWrapper)
    """

    def __init__(
        self,
        rng: jax.Array = jax.random.PRNGKey(0),
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(rng=rng, config=config, config_overrides=config_overrides)

        if self._config.mujoco_impl != "warp":
            raise ValueError(
                "GoToTargetVision requires mujoco_impl='warp' for rendering"
            )

        self._vision_width = self._config.vision_width
        self._vision_height = self._config.vision_height
        self._grayscale = self._config.get("grayscale", False)

    @property
    def vision_shape(self):
        channels = 1 if self._grayscale else 3
        return (self._vision_height, self._vision_width, channels)

    @property
    def vision_enabled(self):
        return True

    @property
    def vision_obs_size(self) -> int:
        h, w, c = self.vision_shape
        return h * w * c

    def _get_obs(self, data, info) -> collections.OrderedDict:
        """Observations with vision placeholder."""
        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        torso = data.bind(self.mjx_model, self._spec.body("torso-rodent"))
        target_pos = info.get("target_position", jp.zeros(3))
        rel_target_world = target_pos - torso.xpos
        ego_target = jp.dot(rel_target_world, torso.xmat)

        task_obs = jp.concatenate(
            [
                info["prev_action"],
                kinematic_sensors,
                touch_sensors,
                origin,
                ego_target,
            ]
        )

        proprioception = self._get_proprioception(data, info, flatten=False)

        obs = collections.OrderedDict(
            task_obs=task_obs,
            proprioception=proprioception,
            vision=jp.zeros(self.vision_shape),
        )
        return collections.OrderedDict(
            state=obs,
            privileged_state=obs,
        )

    @property
    def observation_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        total = 0
        for key in ("task_obs", "proprioception"):
            total += jp.sum(
                flatten_util.ravel_pytree(obs_size["state"][key])[0]
            )
        return total

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0]
        )
