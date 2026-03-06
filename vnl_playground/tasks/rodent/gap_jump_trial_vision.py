"""Vision-enabled discrete-trial gap-jumping task.

Extends GapJumpTrial with egocentric camera observations rendered via the
JAX-callable mujoco_warp GPU ray-tracer.

The observation dict from _get_obs() returns::

    {
        "state": OrderedDict(
            task_obs=...,                       # body-state signals (flat)
            proprioception=OrderedDict(...),    # nested dict
            vision=jp.zeros(H, W, C),           # placeholder zeros
        ),
        "privileged_state": OrderedDict(...),   # same as state
    }

``task_obs`` contains body-state signals (prev_action, kinematic sensors,
touch sensors, origin) plus a 3-dim phase indicator one-hot vector.  Gap
information is not included -- the agent should infer it from vision.

Vision rendering happens in ``VisionRenderWrapper`` (vision_jax.py), which
wraps the vmapped/batched env and renders on all worlds at once using the
JAX-callable warp renderer. The wrapper replaces the zero placeholders with
real rendered images.

Compatible with ``HighLevelWrapper`` for transfer learning pipelines
and direct ff_ppo training.  Downstream ``observation_utils.flatten_obs_dict()``
maps ``task_obs`` to ``imitation_target`` internally.

Usage::

    env = GapJumpTrialVision(config=cfg)
    brax_env = wrap_for_brax_training(env, ...)
    vision_env = VisionRenderWrapper(brax_env, env.mj_model, nworld=num_envs,
                                      **vision_config)

Monocular vs Binocular Experiment Protocol::

    1. Train with default (binocular) vision using GapJumpTrialVision
    2. Evaluate with monocular masking using gap_jump_experiments.py:
       - BINOCULAR condition: no mask (baseline)
       - MONOCULAR_LEFT: left half of image zeroed
       - MONOCULAR_RIGHT: right half of image zeroed
    3. Compare success rates and decision times across conditions
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
    OUTCOME_ONGOING,
    OUTCOME_SUCCESS,
    OUTCOME_FAILURE,
    OUTCOME_ABORT,
    OUTCOME_TIMEOUT,
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
    cfg.vision_mode = "binocular"  # "binocular", "monocular_left", "monocular_right"
    return cfg


def dense_config() -> config_dict.ConfigDict:
    """Returns the legacy dense-reward vision configuration."""
    cfg = gap_jump_trial.dense_config()
    cfg.mujoco_impl = "warp"
    cfg.vision = True
    cfg.vision_width = 64
    cfg.vision_height = 64
    cfg.grayscale = True
    cfg.vision_camera_name = "egocentric-rodent"
    cfg.render_depth = False
    cfg.use_textures = False
    cfg.use_shadows = False
    cfg.vision_mode = "binocular"
    return cfg


class GapJumpTrialVision(gap_jump_trial.GapJumpTrial):
    """GapJumpTrial with egocentric vision observations.

    Observations are returned with state/privileged_state wrapping containing
    task_obs, proprioception, and vision keys. The vision key contains
    placeholder zeros -- real rendering is handled by ``VisionRenderWrapper``
    which wraps the batched env.

    ``task_obs`` provides body-state signals (prev_action, kinematic sensors,
    touch sensors, origin) plus a 3-dim phase indicator. Gap information is
    not included -- the agent should infer it from vision.

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
            raise ValueError(
                "GapJumpTrialVision requires mujoco_impl='warp' for rendering"
            )

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

    @property
    def vision_obs_size(self) -> int:
        """Total number of pixels in the vision observation (H * W * C)."""
        h, w, c = self.vision_shape
        return h * w * c

    def _get_obs(self, data, info) -> collections.OrderedDict:
        """Get observations with body-state signals, phase indicator, and vision.

        task_obs includes body-state signals (prev_action, kinematic sensors,
        touch sensors, origin) plus the phase indicator, matching the RunGapVision
        pattern. Vision placeholder is zeros -- real pixels are injected by
        VisionRenderWrapper after batched GPU rendering.
        """
        phase = info.get("trial_phase", jp.array(PHASE_HOLD, dtype=jp.int32))
        phase_indicator = jax.nn.one_hot(phase, 3)

        kinematic_sensors = self._get_kinematic_sensors(data)
        touch_sensors = self._get_touch_sensors(data)
        origin = self._get_origin(data)

        # Egocentric vector to target position
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
                phase_indicator,
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
        return jp.sum(flatten_util.ravel_pytree(obs_size["state"]["proprioception"])[0])
