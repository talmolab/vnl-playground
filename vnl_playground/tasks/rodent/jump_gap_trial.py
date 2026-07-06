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
