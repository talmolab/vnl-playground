"""Vision-enabled discrete-trial gap-jumping task.

Extends GapJumpTrial with egocentric camera observations rendered via the
JAX-callable mujoco_warp GPU ray-tracer.

Key addition over base GapJumpTrial: phase-based vision masking.
During the HOLD phase the rodent's vision is occluded (all zeros),
matching the Liska et al. paradigm where a barrier blocks the view.
Vision becomes active during DECISION and JUMP phases.

The observation dict from _get_obs() returns::

    {
        "state": OrderedDict(
            task_obs=phase_indicator,           # [3] one-hot phase
            proprioception=OrderedDict(...),    # nested dict
            vision=masked_vision_placeholder,   # [H, W, C] zeros placeholder
            vision_mask=scalar,                 # 0 during HOLD, 1 otherwise
        ),
        "privileged_state": OrderedDict(
            task_obs=phase_indicator,
            proprioception=OrderedDict(...),
            gap_distance=...,
            vision=masked_vision_placeholder,
            vision_mask=scalar,
        ),
    }

Vision rendering happens in ``VisionRenderWrapper`` (vision_jax.py), which
wraps the vmapped/batched env and renders on all worlds at once using the
JAX-callable warp renderer. The wrapper replaces the zero placeholders with
real rendered images. The vision_mask field allows downstream policies to
mask out vision during the HOLD phase even after real pixels are injected.

Usage::

    env = GapJumpTrialVision(config=cfg)
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

from vnl_playground.tasks.rodent import gap_jump_trial
from vnl_playground.tasks.rodent.gap_jump_trial import (
    PHASE_HOLD,
    PHASE_DECISION,
    PHASE_JUMP,
)


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the GapJumpTrialVision environment."""
    cfg = gap_jump_trial.default_config()
    cfg.mujoco_impl = "warp"  # Required for mujoco_warp rendering
    # Vision parameters
    cfg.vision = True
    cfg.vision_width = 64
    cfg.vision_height = 64
    cfg.grayscale = True
    cfg.vision_camera_name = "egocentric-rodent"
    cfg.render_depth = False
    cfg.use_textures = False
    cfg.use_shadows = False
    return cfg


class GapJumpTrialVision(gap_jump_trial.GapJumpTrial):
    """GapJumpTrial with egocentric vision observations.

    Observations are returned with state/privileged_state wrapping containing
    task_obs, proprioception, vision, and vision_mask keys. The vision key
    contains placeholder zeros -- real rendering is handled by
    ``VisionRenderWrapper`` which wraps the batched env.

    Phase-based vision masking: during HOLD phase, the vision placeholder is
    zeroed out and vision_mask is 0.0. During DECISION and JUMP phases,
    vision_mask is 1.0 and the placeholder is left as-is (to be replaced
    by real rendered pixels via VisionRenderWrapper).
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
                "GapJumpTrialVision requires mujoco_impl='warp' for rendering"
            )

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

    @property
    def vision_obs_size(self) -> int:
        """Total number of pixels in the vision observation (H * W * C)."""
        h, w, c = self.vision_shape
        return h * w * c

    def _get_obs(
        self, data, info
    ) -> collections.OrderedDict:
        """Get observations with phase indicator, proprioception, and vision.

        Adds vision placeholder and phase-based vision masking on top of the
        base GapJumpTrial observations. During HOLD phase, vision is zeroed
        out (occluded barrier). During DECISION and JUMP phases, the vision
        placeholder is active (to be replaced by VisionRenderWrapper).

        Args:
            data: The simulation data (mjx.Data).
            info: State info dictionary.

        Returns:
            OrderedDict with state and privileged_state keys.
        """
        phase = info.get("trial_phase", jp.array(PHASE_HOLD, dtype=jp.int32))
        phase_indicator = jax.nn.one_hot(phase, 3)

        proprioception = self._get_proprioception(data, info, flatten=False)

        # Vision mask: 0 during HOLD, 1 during DECISION/JUMP
        vision_mask = (phase > PHASE_HOLD).astype(jp.float32)

        # Vision placeholder multiplied by mask -- zeros during HOLD phase.
        # VisionRenderWrapper replaces zeros with real pixels, but the policy
        # can use vision_mask to know when vision should be masked.
        masked_vision_placeholder = jp.zeros(self.vision_shape) * vision_mask

        obs = collections.OrderedDict(
            task_obs=phase_indicator,
            proprioception=proprioception,
            vision=masked_vision_placeholder,
            vision_mask=vision_mask,
        )

        privileged_obs = collections.OrderedDict(
            task_obs=phase_indicator,
            proprioception=proprioception,
            gap_distance=jp.array(info.get("gap_distance", 0.0)).reshape(1),
            vision=masked_vision_placeholder,
            vision_mask=vision_mask,
        )

        return collections.OrderedDict(
            state=obs,
            privileged_state=privileged_obs,
        )

    @property
    def observation_size(self) -> int:
        """Total flat observation size for the MLP (excludes vision pixels).

        Vision is handled separately by the CNN, so we only count task_obs
        and proprioception in the flat observation size.
        """
        obs_size = self.non_flattened_observation_size
        total = 0
        for key in ("task_obs", "proprioception"):
            total += jp.sum(flatten_util.ravel_pytree(obs_size["state"][key])[0])
        return total

    @property
    def proprioceptive_obs_size(self) -> int:
        """Flat size of the proprioceptive observation component."""
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0]
        )
