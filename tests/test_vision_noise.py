"""Tests for Gaussian sensor noise in BinocularVisionRenderWrapper.

E3 (`_implementation_log/DMPO/NEXT_EXPERIMENTS.md`) adds `vision_noise_std`:
per-step i.i.d. Gaussian noise on the rendered image, `v + sigma * N(0,1)`,
clipped to the renderer's [0, 1] range. The knob lives next to
`eye_dropout_rate` and is applied in BOTH training and eval, because
deliberation has to be *trained* under noise, not merely probed under it.

Structured like `test_eye_dropout.py`: the masking/noise methods are bound to a
mock so the logic can be tested without instantiating MuJoCo models.
"""

import jax
import jax.numpy as jnp
import pytest


def _make_wrapper(vision_noise_std=0.1, eye_dropout_rate=0.0):
    """Bind the noise/mask methods to a mock instance (no MuJoCo needed)."""
    from vnl_playground.tasks.rodent.vision_jax import BinocularVisionRenderWrapper

    class MockWrapper:
        pass

    w = MockWrapper()
    w._vision_noise_std = vision_noise_std
    w._eye_dropout_rate = eye_dropout_rate
    w._eval_eye_mode = "binocular"
    w._apply_vision_noise = BinocularVisionRenderWrapper._apply_vision_noise.__get__(w)
    w._apply_eye_mask = BinocularVisionRenderWrapper._apply_eye_mask.__get__(w)
    return w


class TestApplyVisionNoise:
    def test_zero_std_is_exact_identity(self):
        """sigma=0 must be bit-identical, not merely close.

        sigma=0 IS the w2 baseline arm of the E3 ladder, so any drift here
        would silently make the control a different experiment.
        """
        w = _make_wrapper(vision_noise_std=0.0)
        vision = jax.random.uniform(jax.random.PRNGKey(0), (4, 8, 8, 2))
        out = w._apply_vision_noise(vision, jax.random.PRNGKey(1))
        assert jnp.array_equal(out, vision)

    def test_negative_std_is_identity(self):
        """A negative sigma is treated as 'off' rather than flipping the sign."""
        w = _make_wrapper(vision_noise_std=-0.5)
        vision = jax.random.uniform(jax.random.PRNGKey(0), (4, 8, 8, 2))
        out = w._apply_vision_noise(vision, jax.random.PRNGKey(1))
        assert jnp.array_equal(out, vision)

    def test_shape_and_dtype_preserved(self):
        w = _make_wrapper(vision_noise_std=0.1)
        vision = jnp.full((3, 8, 8, 2), 0.5, dtype=jnp.float32)
        out = w._apply_vision_noise(vision, jax.random.PRNGKey(2))
        assert out.shape == vision.shape
        assert out.dtype == vision.dtype

    def test_output_always_within_unit_range(self):
        """Clip is what keeps noise inside the renderer's [0,1] convention.

        Uses a large sigma so the unclipped tails would blow way past both
        bounds; the clip must catch every one.
        """
        w = _make_wrapper(vision_noise_std=1.0)
        vision = jax.random.uniform(jax.random.PRNGKey(3), (16, 8, 8, 2))
        out = w._apply_vision_noise(vision, jax.random.PRNGKey(4))
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_saturated_inputs_stay_saturated_on_the_clipped_side(self):
        """An all-white image can only be darkened; an all-black one lightened."""
        w = _make_wrapper(vision_noise_std=0.2)
        white = jnp.ones((8, 8, 8, 2))
        black = jnp.zeros((8, 8, 8, 2))
        out_w = w._apply_vision_noise(white, jax.random.PRNGKey(5))
        out_b = w._apply_vision_noise(black, jax.random.PRNGKey(6))
        assert float(out_w.max()) <= 1.0
        assert float(out_b.min()) >= 0.0
        # And they must actually move -- otherwise the clip is masking a no-op.
        assert float(out_w.min()) < 1.0
        assert float(out_b.max()) > 0.0

    @pytest.mark.parametrize("sigma", [0.05, 0.1, 0.2])
    def test_noise_is_unbiased_with_correct_scale(self, sigma):
        """On mid-gray (far from both clip bounds) the perturbation is N(0, sigma).

        0.5 +- 4*sigma stays inside [0,1] for every sigma on the E3 ladder, so
        clipping does not bite and the moments are the raw Gaussian's.
        """
        w = _make_wrapper(vision_noise_std=sigma)
        vision = jnp.full((64, 32, 32, 2), 0.5)
        out = w._apply_vision_noise(vision, jax.random.PRNGKey(7))
        delta = out - vision
        assert abs(float(delta.mean())) < 0.01, "noise should be zero-mean"
        assert abs(float(delta.std()) - sigma) < 0.01 * max(1.0, sigma / 0.05)

    def test_noise_is_iid_across_pixels_worlds_and_eyes(self):
        """i.i.d., not one draw broadcast over the image.

        A broadcast bug would still pass the moment test above, so check the
        structure directly: per-world means differ, and the two eyes of the
        same world are not the same pattern.
        """
        w = _make_wrapper(vision_noise_std=0.1)
        vision = jnp.full((8, 16, 16, 2), 0.5)
        out = w._apply_vision_noise(vision, jax.random.PRNGKey(8))
        delta = out - vision

        # Different worlds get different noise.
        world_means = delta.mean(axis=(1, 2, 3))
        assert len(jnp.unique(world_means)) == delta.shape[0]

        # The two eyes are independent, not a duplicated channel.
        assert not jnp.allclose(delta[..., 0], delta[..., 1])

        # Neighbouring pixels differ (not a per-image constant).
        assert not jnp.allclose(delta[:, 0, 0, :], delta[:, 0, 1, :])

    def test_different_keys_give_different_noise(self):
        """Per-STEP noise: advancing the key must change the draw."""
        w = _make_wrapper(vision_noise_std=0.1)
        vision = jnp.full((4, 8, 8, 2), 0.5)
        a = w._apply_vision_noise(vision, jax.random.PRNGKey(10))
        b = w._apply_vision_noise(vision, jax.random.PRNGKey(11))
        assert not jnp.allclose(a, b)

    def test_same_key_is_deterministic(self):
        """Same key -> same draw, so runs stay reproducible."""
        w = _make_wrapper(vision_noise_std=0.1)
        vision = jnp.full((4, 8, 8, 2), 0.5)
        rng = jax.random.PRNGKey(12)
        assert jnp.array_equal(
            w._apply_vision_noise(vision, rng), w._apply_vision_noise(vision, rng)
        )

    def test_jit_compatible(self):
        """Must survive jit: it runs inside the rollout scan."""
        w = _make_wrapper(vision_noise_std=0.1)
        vision = jnp.full((4, 8, 8, 2), 0.5)
        out = jax.jit(w._apply_vision_noise)(vision, jax.random.PRNGKey(13))
        assert out.shape == vision.shape
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


class TestNoiseComposesWithEyeDropout:
    """Ordering matters: noise renders the sensor, dropout blinds the eye."""

    def test_dropped_eye_is_exactly_zero_after_noise(self):
        """Noise must be applied BEFORE masking.

        A blind eye has to read exactly 0 -- that is what the network learned
        "blind" means. If noise were added after the mask, a dropped eye would
        carry a nonzero noise floor and stop being a dropout signal at all.
        """
        w = _make_wrapper(vision_noise_std=0.2, eye_dropout_rate=1.0)
        vision = jnp.full((32, 8, 8, 2), 0.5)

        noised = w._apply_vision_noise(vision, jax.random.PRNGKey(14))
        masked = w._apply_eye_mask(noised, jax.random.PRNGKey(15))

        left_sum = masked[..., 0].sum(axis=(1, 2))
        right_sum = masked[..., 1].sum(axis=(1, 2))
        # rate=1.0 => exactly one eye zeroed per world.
        exactly_one_zero = (left_sum == 0) ^ (right_sum == 0)
        assert bool(jnp.all(exactly_one_zero))
