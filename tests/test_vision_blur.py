"""Tests for IIR temporal blur (motion blur) in BinocularVisionRenderWrapper.

`vision_blur_tau_ms` models the rat photoreceptor's finite integration time.
The warp batch renderer samples instantaneously (one ray/pixel from the current
`Data`, no shutter -- verified in `mujoco_warp/_src/render.py`), so the eye
image is temporally aliased: sharper in time than a real retina. This knob
applies a first-order low-pass across control steps,

    blurred_t = alpha * blurred_{t-1} + (1 - alpha) * render_t,
    alpha     = exp(-ctrl_dt / tau),

which smears moving content in proportion to how fast it moves. Unlike additive
i.i.d. noise, blur DESTROYS information and cannot be pooled away -- and it is
reducible by an action (slow down), which is the point.

Structured like `test_vision_noise.py`: the blur method is bound to a mock so
the logic can be tested without instantiating MuJoCo models.
"""

import math

import jax
import jax.numpy as jnp
import pytest


def _make_wrapper(vision_blur_tau_ms=25.0, ctrl_dt=0.01):
    """Bind the blur method to a mock instance (no MuJoCo needed)."""
    from vnl_playground.tasks.rodent.vision_jax import (
        BinocularVisionRenderWrapper,
        blur_alpha_from_tau,
    )

    class MockWrapper:
        pass

    w = MockWrapper()
    w._vision_blur_tau_ms = vision_blur_tau_ms
    w._blur_alpha = blur_alpha_from_tau(vision_blur_tau_ms, ctrl_dt)
    w._apply_vision_blur = BinocularVisionRenderWrapper._apply_vision_blur.__get__(w)
    return w


class TestBlurAlphaFromTau:
    """alpha is derived from tau and ctrl_dt, never configured directly.

    tau is the physical quantity (photoreceptor integration time); alpha is
    meaningless without ctrl_dt, and this repo's arms DO vary it (run_gap's
    default is 0.02, the E3 sweep arms override to 0.01). Configuring alpha
    would silently mean different exposures across arms.
    """

    def test_tau_zero_gives_zero_alpha(self):
        from vnl_playground.tasks.rodent.vision_jax import blur_alpha_from_tau

        assert blur_alpha_from_tau(0.0, 0.01) == 0.0

    def test_alpha_is_exp_minus_dt_over_tau(self):
        from vnl_playground.tasks.rodent.vision_jax import blur_alpha_from_tau

        # ctrl_dt = 10 ms, tau = 25 ms -> exp(-0.4)
        assert blur_alpha_from_tau(25.0, 0.01) == pytest.approx(math.exp(-0.4))

    def test_alpha_tracks_ctrl_dt(self):
        """Same tau at half the control period must retain more history."""
        from vnl_playground.tasks.rodent.vision_jax import blur_alpha_from_tau

        fast = blur_alpha_from_tau(25.0, 0.005)
        slow = blur_alpha_from_tau(25.0, 0.02)
        assert fast > slow

    def test_longer_tau_gives_larger_alpha(self):
        from vnl_playground.tasks.rodent.vision_jax import blur_alpha_from_tau

        assert blur_alpha_from_tau(50.0, 0.01) > blur_alpha_from_tau(10.0, 0.01)


class TestApplyVisionBlur:
    def test_zero_tau_is_exact_identity(self):
        """tau=0 must be bit-identical, not merely close.

        Every existing arm runs at tau=0, so any drift here would silently
        turn all of them into different experiments.
        """
        w = _make_wrapper(vision_blur_tau_ms=0.0)
        vision = jax.random.uniform(jax.random.PRNGKey(0), (4, 8, 8, 2))
        prev = jax.random.uniform(jax.random.PRNGKey(1), (4, 8, 8, 2))
        done = jnp.zeros((4,))
        out = w._apply_vision_blur(vision, prev, done)
        assert jnp.array_equal(out, vision)

    def test_negative_tau_is_identity(self):
        """A negative tau is a config typo, not a request to amplify motion."""
        w = _make_wrapper(vision_blur_tau_ms=-5.0)
        vision = jax.random.uniform(jax.random.PRNGKey(0), (4, 8, 8, 2))
        prev = jax.random.uniform(jax.random.PRNGKey(1), (4, 8, 8, 2))
        out = w._apply_vision_blur(vision, prev, jnp.zeros((4,)))
        assert jnp.array_equal(out, vision)

    def test_shape_and_dtype_preserved(self):
        w = _make_wrapper(vision_blur_tau_ms=25.0)
        vision = jnp.full((3, 8, 8, 2), 0.5, dtype=jnp.float32)
        prev = jnp.zeros((3, 8, 8, 2), dtype=jnp.float32)
        out = w._apply_vision_blur(vision, prev, jnp.zeros((3,)))
        assert out.shape == vision.shape
        assert out.dtype == vision.dtype

    def test_single_step_blend_matches_alpha(self):
        w = _make_wrapper(vision_blur_tau_ms=25.0, ctrl_dt=0.01)
        alpha = math.exp(-0.4)
        prev = jnp.zeros((2, 4, 4, 2))
        vision = jnp.ones((2, 4, 4, 2))
        out = w._apply_vision_blur(vision, prev, jnp.zeros((2,)))
        assert float(out.mean()) == pytest.approx(1.0 - alpha, abs=1e-6)

    def test_static_scene_converges_to_the_render(self):
        """Blur must be a no-op on a scene that is not moving.

        This is the property that makes it MOTION blur rather than a global
        softening: only moving content is smeared.
        """
        w = _make_wrapper(vision_blur_tau_ms=25.0, ctrl_dt=0.01)
        vision = jnp.full((2, 4, 4, 2), 0.7)
        state = jnp.zeros((2, 4, 4, 2))
        done = jnp.zeros((2,))
        for _ in range(200):
            state = w._apply_vision_blur(vision, state, done)
        assert jnp.allclose(state, vision, atol=1e-5)

    def test_step_response_follows_exponential(self):
        """After n steps of a step input, residual is exactly alpha**n."""
        w = _make_wrapper(vision_blur_tau_ms=25.0, ctrl_dt=0.01)
        alpha = math.exp(-0.4)
        vision = jnp.ones((1, 4, 4, 2))
        state = jnp.zeros((1, 4, 4, 2))
        done = jnp.zeros((1,))
        for n in range(1, 6):
            state = w._apply_vision_blur(vision, state, done)
            assert float(state.mean()) == pytest.approx(1.0 - alpha**n, abs=1e-6)

    def test_longer_tau_smears_more(self):
        """A longer integration time must retain more of the previous frame."""
        prev = jnp.ones((1, 4, 4, 2))
        vision = jnp.zeros((1, 4, 4, 2))
        done = jnp.zeros((1,))
        short = _make_wrapper(10.0)._apply_vision_blur(vision, prev, done)
        long = _make_wrapper(50.0)._apply_vision_blur(vision, prev, done)
        assert float(long.mean()) > float(short.mean())

    def test_done_hard_resets_to_fresh_render(self):
        """On done the filter must drop history entirely.

        The vision wrapper sits OUTSIDE AutoResetWrapper, so on a done step
        `state.data` is ALREADY the fresh episode. Blending would ghost the
        previous episode's last frame into the new episode's first frames --
        the same class of bug as the gap_crossing_bonus info ratchet.
        """
        w = _make_wrapper(vision_blur_tau_ms=25.0)
        prev = jnp.ones((4, 8, 8, 2))
        vision = jnp.zeros((4, 8, 8, 2))
        done = jnp.array([1.0, 0.0, 1.0, 0.0])

        out = w._apply_vision_blur(vision, prev, done)

        # Done worlds: exactly the fresh render, no ghost.
        assert jnp.array_equal(out[0], vision[0])
        assert jnp.array_equal(out[2], vision[2])
        # Not-done worlds: still carrying history.
        assert float(out[1].mean()) > 0.0
        assert float(out[3].mean()) > 0.0

    def test_done_is_per_world_not_broadcast(self):
        """A scalar-ish done must not blank the whole batch."""
        w = _make_wrapper(vision_blur_tau_ms=25.0)
        prev = jnp.ones((3, 4, 4, 2))
        vision = jnp.zeros((3, 4, 4, 2))
        out = w._apply_vision_blur(vision, prev, jnp.array([1.0, 0.0, 0.0]))
        assert float(out[0].max()) == 0.0
        assert float(out[1].max()) > 0.0

    def test_output_stays_within_unit_range(self):
        """A convex combination of two [0,1] images stays in [0,1]."""
        w = _make_wrapper(vision_blur_tau_ms=25.0)
        prev = jax.random.uniform(jax.random.PRNGKey(2), (8, 8, 8, 2))
        vision = jax.random.uniform(jax.random.PRNGKey(3), (8, 8, 8, 2))
        out = w._apply_vision_blur(vision, prev, jnp.zeros((8,)))
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_jit_compatible(self):
        """Must survive jit: it runs inside the rollout scan."""
        w = _make_wrapper(vision_blur_tau_ms=25.0)
        vision = jnp.full((4, 8, 8, 2), 0.5)
        prev = jnp.zeros((4, 8, 8, 2))
        out = jax.jit(w._apply_vision_blur)(vision, prev, jnp.zeros((4,)))
        assert out.shape == vision.shape
