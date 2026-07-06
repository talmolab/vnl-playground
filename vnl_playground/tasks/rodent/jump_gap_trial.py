"""JumpGapTrial: single-gap controlled trial for testing rodent gap-jumping.

Extends RunGapVision with the corridor constrained to exactly one gap
(start platform -> gap -> landing platform), so a jump attempt can be run
as a single controlled trial rather than a continuously running multi-gap
course.

This is kept as its own class -- rather than just passing ``n_platforms=1``
into RunGap/RunGapVision directly -- because trial-oriented extensions
(additional reward terms, a trial-status indicator, and a trial-sequence
history in ``task_obs``) belong here so they don't complicate the general
``run_gap`` task. None of that is implemented yet: this class is currently
a thin subclass that only fixes the corridor to a single gap and reuses
RunGapVision's observations, rewards, and terminations unchanged.

Usage::

    env = JumpGapTrial(config=cfg)
    brax_env = wrap_for_brax_training(env, ...)
    vision_env = VisionRenderWrapper(brax_env, env.mj_model, nworld=num_envs,
                                      **vision_config)
"""

import jax.numpy as jp
from ml_collections import config_dict

from vnl_playground.tasks.rodent import run_gap_vision
from vnl_playground.tasks.rodent.run_gap import _registry


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for the JumpGapTrial environment.

    Same as ``RunGapVision.default_config()``, constrained to a single gap:
    one starting platform, one gap, and one landing platform.
    """
    cfg = run_gap_vision.default_config()
    cfg.n_platforms = 1
    return cfg


class JumpGapTrial(run_gap_vision.RunGapVision):
    """Single-gap controlled variant of RunGapVision.

    Identical to RunGapVision except the corridor is constrained to exactly
    one gap (``n_platforms=1``) via ``default_config()``, making it suitable
    for controlled single-trial gap-jumping experiments instead of a
    continuously running multi-gap course.

    Reuses RunGap's reward/termination registry (``forward_velocity``,
    ``termination_penalty``, etc. -- see run_gap.py for the implementations)
    unchanged. New trial-specific reward terms, a trial-status indicator,
    and trial-sequence history in ``task_obs`` are expected to be added to
    this class going forward.
    """

    _registry = _registry

    @_registry.reward("gap_crossing_bonus_scaled")
    def _gap_crossing_bonus_scaled(self, data, info, metrics, weight) -> float:
        """Crossing bonus that scales with gap size -- larger gaps pay more.

        Awarded once, on the step the agent actually crosses the (single) gap
        far edge (``info['just_crossed_gap']``, set in RunGap.step()). The bonus
        is ``weight * (gap_size / max_gap)``, so clearing a wider gap yields a
        proportionally larger reward, pushing the policy toward its maximum
        jump distance. Because it only fires on a real crossing, it cannot be
        gamed by lunging out over the gap without landing.

        Assumes the single-gap trial (``n_platforms=1``, ``randomize_gaps``):
        gap size = landing-platform near edge - start-platform trailing edge,
        read from the current platform geometry.
        """
        plat_centers = data.xpos[self._platform_body_ids, 0]
        landing_near_edge = plat_centers[0] - self._platform_half_length
        gap_size = landing_near_edge - self._start_platform_half_length
        max_gap = self._config.gap_length_range[1]
        gap_frac = jp.clip(gap_size / max_gap, 0.0, 1.0)
        bonus = jp.where(
            info.get("just_crossed_gap", False), weight * gap_frac, 0.0
        )
        metrics["rewards/gap_crossing_bonus_scaled"] = bonus
        return bonus
