"""Tests for stochastic eye dropout in BinocularVisionRenderWrapper."""

import jax
import jax.numpy as jnp
import pytest


class TestApplyEyeMask:
    """Unit tests for the eye masking logic.

    These test the masking functions in isolation using simple synthetic
    binocular vision tensors, without requiring MuJoCo rendering.
    """

    def _make_wrapper_with_masking(
        self, eye_dropout_rate=0.4, eval_eye_mode="binocular"
    ):
        """Create a minimal mock wrapper with masking methods.

        We can't easily instantiate the full BinocularVisionRenderWrapper
        without MuJoCo models, so we import the class and test the masking
        methods by creating a partial instance.
        """
        from vnl_playground.tasks.rodent.vision_jax import BinocularVisionRenderWrapper

        class MockWrapper:
            pass

        w = MockWrapper()
        w._eye_dropout_rate = eye_dropout_rate
        w._eval_eye_mode = eval_eye_mode
        w._apply_eye_mask = BinocularVisionRenderWrapper._apply_eye_mask.__get__(w)
        w._apply_eval_eye_mask = (
            BinocularVisionRenderWrapper._apply_eval_eye_mask.__get__(w)
        )
        return w

    def test_no_dropout_when_rate_zero(self):
        """With eye_dropout_rate=0.0, vision should be unchanged."""
        w = self._make_wrapper_with_masking(eye_dropout_rate=0.0)
        vision = jnp.ones((4, 32, 32, 2))
        rng = jax.random.PRNGKey(0)
        result = w._apply_eye_mask(vision, rng)
        assert jnp.allclose(result, vision)

    def test_full_dropout_always_masks_one_eye(self):
        """With eye_dropout_rate=1.0, every world should have exactly one eye zeroed."""
        w = self._make_wrapper_with_masking(eye_dropout_rate=1.0)
        nworld = 128
        vision = jnp.ones((nworld, 32, 32, 2))
        rng = jax.random.PRNGKey(42)
        result = w._apply_eye_mask(vision, rng)

        left = result[..., 0]
        right = result[..., 1]
        left_sum = left.sum(axis=(1, 2))
        right_sum = right.sum(axis=(1, 2))

        both_zero = (left_sum == 0) & (right_sum == 0)
        neither_zero = (left_sum > 0) & (right_sum > 0)
        assert not jnp.any(both_zero), "Both eyes zeroed in some world"
        assert not jnp.any(neither_zero), "Neither eye zeroed in some world"

    def test_dropout_rate_statistical(self):
        """Check that dropout frequency roughly matches the configured rate."""
        w = self._make_wrapper_with_masking(eye_dropout_rate=0.4)
        nworld = 4096
        vision = jnp.ones((nworld, 8, 8, 2))
        rng = jax.random.PRNGKey(123)
        result = w._apply_eye_mask(vision, rng)

        left = result[..., 0].sum(axis=(1, 2))
        right = result[..., 1].sum(axis=(1, 2))
        pixels = 8 * 8

        n_bino = jnp.sum((left == pixels) & (right == pixels))
        n_left_only = jnp.sum((left == pixels) & (right == 0))
        n_right_only = jnp.sum((left == 0) & (right == pixels))

        assert n_bino / nworld > 0.5, f"Too few binocular: {n_bino/nworld:.2f}"
        assert n_bino / nworld < 0.7, f"Too many binocular: {n_bino/nworld:.2f}"
        assert (
            n_left_only / nworld > 0.1
        ), f"Too few left-only: {n_left_only/nworld:.2f}"
        assert (
            n_right_only / nworld > 0.1
        ), f"Too few right-only: {n_right_only/nworld:.2f}"

    def test_dropout_symmetric_between_eyes(self):
        """Left-only and right-only should occur with roughly equal frequency."""
        w = self._make_wrapper_with_masking(eye_dropout_rate=1.0)
        nworld = 4096
        vision = jnp.ones((nworld, 8, 8, 2))
        rng = jax.random.PRNGKey(7)
        result = w._apply_eye_mask(vision, rng)

        left = result[..., 0].sum(axis=(1, 2))
        right = result[..., 1].sum(axis=(1, 2))
        pixels = 8 * 8

        n_left_only = jnp.sum((left == pixels) & (right == 0))
        n_right_only = jnp.sum((left == 0) & (right == pixels))

        ratio = n_left_only / (n_left_only + n_right_only)
        assert 0.4 < ratio < 0.6, f"Asymmetric eye dropout: left_frac={ratio:.2f}"

    def test_eval_mode_binocular(self):
        """eval_eye_mode='binocular' should not mask anything."""
        w = self._make_wrapper_with_masking(eval_eye_mode="binocular")
        vision = jnp.ones((4, 32, 32, 2))
        result = w._apply_eval_eye_mask(vision)
        assert jnp.allclose(result, vision)

    def test_eval_mode_left_only(self):
        """eval_eye_mode='left_only' should zero right eye channels."""
        w = self._make_wrapper_with_masking(eval_eye_mode="left_only")
        vision = jnp.ones((4, 32, 32, 2))
        result = w._apply_eval_eye_mask(vision)
        assert jnp.allclose(result[..., 0], 1.0), "Left eye should be active"
        assert jnp.allclose(result[..., 1], 0.0), "Right eye should be zeroed"

    def test_eval_mode_right_only(self):
        """eval_eye_mode='right_only' should zero left eye channels."""
        w = self._make_wrapper_with_masking(eval_eye_mode="right_only")
        vision = jnp.ones((4, 32, 32, 2))
        result = w._apply_eval_eye_mask(vision)
        assert jnp.allclose(result[..., 0], 0.0), "Left eye should be zeroed"
        assert jnp.allclose(result[..., 1], 1.0), "Right eye should be active"

    def test_rgb_binocular_masking(self):
        """Eye dropout should work with RGB (C=3, total 6 channels)."""
        w = self._make_wrapper_with_masking(eye_dropout_rate=1.0)
        nworld = 64
        vision = jnp.ones((nworld, 16, 16, 6))
        rng = jax.random.PRNGKey(99)
        result = w._apply_eye_mask(vision, rng)

        left = result[..., :3]
        right = result[..., 3:]

        for i in range(nworld):
            left_active = jnp.any(left[i] > 0)
            right_active = jnp.any(right[i] > 0)
            assert (
                left_active != right_active
            ), f"World {i}: expected exactly one eye active"

    def test_eval_mode_invalid_raises(self):
        """Invalid eval_eye_mode should raise ValueError."""
        w = self._make_wrapper_with_masking(eval_eye_mode="invalid_mode")
        vision = jnp.ones((4, 32, 32, 2))
        with pytest.raises(ValueError, match="Unknown eval_eye_mode"):
            w._apply_eval_eye_mask(vision)

    def test_different_rng_gives_different_masks(self):
        """Different RNG keys should produce different masking patterns."""
        w = self._make_wrapper_with_masking(eye_dropout_rate=0.5)
        vision = jnp.ones((256, 8, 8, 2))

        result1 = w._apply_eye_mask(vision, jax.random.PRNGKey(0))
        result2 = w._apply_eye_mask(vision, jax.random.PRNGKey(1))

        assert not jnp.allclose(result1, result2)
