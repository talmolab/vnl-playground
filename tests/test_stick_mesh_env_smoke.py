"""End-to-end smoke: load the mesh stick model (no reference data), reset, step.

Uses ``StickMaintainVelocity`` — the non-imitation env that builds the same
mesh walker (``consts.STICK_XML_PATH`` = ``stick_mesh_fast.xml``) via
``base.add_stick()`` + ``compile()``. This verifies the mesh model loads,
compiles to MJX, and simulates without requiring any vendored reference-data
H5 file.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import pytest


@pytest.fixture(scope="module")
def stick_env():
    from vnl_playground import registry

    # Raw env (no obs wrapper) so we can read state.data / mj_model directly.
    return registry.load("StickMaintainVelocity", flatten_obs=False)


def test_mesh_model_compiles_with_expected_shapes(stick_env):
    # Free joint (7) + 41 hinge joints = 48 qpos.
    assert stick_env.mj_model.nq == 48
    assert stick_env.mj_model.nu == 41
    # 6 floor<->claw pairs added by base.add_stick().
    assert stick_env.mj_model.npair == 6


def test_reset_and_step_returns_finite(stick_env):
    state = stick_env.reset(jax.random.PRNGKey(0))
    assert jp.all(jp.isfinite(state.data.qpos))
    next_state = stick_env.step(state, jp.zeros(stick_env.action_size))
    assert jp.all(jp.isfinite(next_state.data.qpos))
    assert jp.isfinite(next_state.reward)
