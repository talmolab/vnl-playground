"""Vision-enabled run through corridor with gaps task.

Extends RunGap with egocentric camera observations rendered via the
JAX-callable mujoco_warp GPU ray-tracer.

The observation dict from _get_obs() returns::

    {
        "state": OrderedDict(
            imitation_target=...,               # gap features (16-dim)
            proprioception=OrderedDict(...),     # nested dict
            vision=jp.zeros(H, W, C),           # placeholder zeros
        ),
        "privileged_state": OrderedDict(...),   # same as state
    }

Vision rendering happens in ``VisionRenderWrapper`` (vision_jax.py), which
wraps the vmapped/batched env and renders on all worlds at once using the
JAX-callable warp renderer. The wrapper replaces the zero placeholders with
real rendered images.

Compatible with ``HighLevelWrapper`` for transfer learning pipelines
(the wrapper extracts imitation_target for the high-level policy and
proprioception for the frozen decoder; vision is preserved for optional
CNN processing).

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
    return cfg


class RunGapVision(run_gap.RunGap):
    """RunGap with egocentric vision observations.

    Observations are returned with state/privileged_state wrapping containing
    imitation_target, proprioception, and vision keys. The vision key contains
    placeholder zeros — real rendering is handled by ``VisionRenderWrapper``
    which wraps the batched env.

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
        """Shape of the vision observation: (H, W, C) where C=1 if grayscale else 3."""
        channels = 1 if self._grayscale else 3
        return (self._vision_height, self._vision_width, channels)

    @property
    def vision_enabled(self):
        """Whether vision observations are enabled."""
        return self._vision_enabled

    def _get_obs(self, data, info) -> collections.OrderedDict:
        """Get observations in state/privileged_state dict format.

        Returns an OrderedDict with state and privileged_state keys, each
        containing:

        - imitation_target: Gap features from parent RunGap (16-dim). Used by
          the high-level policy in transfer learning, or by the intention
          encoder in direct training.
        - proprioception: Nested OrderedDict of body state sensors. Used by
          the decoder in transfer learning.
        - vision: Zeros placeholder with shape (H, W, C). Real pixels are
          injected by VisionRenderWrapper after batched rendering.

        This structure is compatible with both ``HighLevelWrapper`` (for
        transfer learning) and direct ff_ppo training.

        Args:
            data: The simulation data (mjx.Data).
            info: State info dictionary.

        Returns:
            OrderedDict with state and privileged_state keys.
        """
        obs = collections.OrderedDict(
            imitation_target=self._get_gap_features(data),
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
        for key in ("imitation_target", "proprioception"):
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
