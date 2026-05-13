"""Parity tests: stick imitation must expose the same surface as rodent."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import inspect


def test_default_config_has_reference_stride_and_rescale_one():
    from vnl_playground.tasks.stick import imitation
    cfg = imitation.default_config()
    assert "reference_stride" in cfg
    assert cfg.reference_stride == 1
    assert cfg.rescale_factor == 1.0


def test_imitation_class_has_rodent_parity_methods():
    from vnl_playground.tasks.stick.imitation import Imitation
    for name in ("_last_valid_frame", "_compile_with_ghost", "render_optimized",
                 "verify_reference_data"):
        assert hasattr(Imitation, name), f"missing {name}"


def test_last_valid_frame_uses_reference_stride_formula():
    """Same formula as rodent: clip_length - (reference_length - 1) * stride - 2."""
    from vnl_playground.tasks.stick.imitation import Imitation
    src = inspect.getsource(Imitation._last_valid_frame)
    assert "reference_length" in src
    assert "reference_stride" in src
    assert "- 2" in src


def test_torso_z_metric_renamed_from_body_z():
    """The torso_z_range reward must store the height under metrics['torso_z']."""
    from vnl_playground.tasks.stick.imitation import Imitation
    src = inspect.getsource(Imitation._torso_z_range_reward)
    assert '"torso_z"' in src
    assert '"body_z"' not in src
