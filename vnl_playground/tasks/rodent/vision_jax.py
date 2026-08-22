"""JAX-native vision rendering using the official MJX render API.

Provides two main components:

1. ``JaxVisionRenderer`` -- wraps ``mjx.create_render_context``,
   ``mjx.refit_bvh``, and ``mjx.render`` to produce batched vision
   observations from ``mjx.Data``.

2. ``VisionRenderWrapper`` -- a Brax-compatible environment wrapper that
   renders vision observations on batched data after each vmapped step.

The rendering is JAX-traceable: it works inside ``jax.jit`` and
``jax.lax.scan`` with no Python-level sync needed.

Usage::

    from vnl_playground.tasks.rodent.vision_jax import VisionRenderWrapper

    raw_env = RunGapVision(config=cfg)
    brax_env = wrap_for_brax_training(raw_env, ...)
    env = VisionRenderWrapper(
        brax_env, raw_env.mj_model, mjx_model=mx, nworld=num_envs,
        width=32, height=32, grayscale=True,
        camera_name="egocentric-rodent",
    )
    # env.step() now produces real vision observations
"""

import collections
import logging
import os
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import warp as wp

from mujoco_playground._src import mjx_env

logger = logging.getLogger(__name__)


def _get_warp_cuda_devices() -> list[str]:
    """Return warp device strings for all visible CUDA GPUs.

    Respects ``CUDA_VISIBLE_DEVICES`` so that single-GPU runs
    (e.g. ``CUDA_VISIBLE_DEVICES=0``) only register one device.
    """
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_vis is not None:
        # CUDA_VISIBLE_DEVICES remaps ordinals: "2,3" → runtime sees cuda:0, cuda:1
        n = len([x.strip() for x in cuda_vis.split(",") if x.strip()])
    else:
        n = wp.get_cuda_device_count()
    return [f"cuda:{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Batch-aware pixel unpacking helpers
# ---------------------------------------------------------------------------
# The official mjx.get_rgb works per-world. These helpers operate on the full
# (nworld, total_pixels) packed output for efficient batch unpacking.
#
# Pixel format is ABGR uint32: B in bits 0-7, G in bits 8-15, R in bits 16-23.
# ---------------------------------------------------------------------------


def _unpack_rgb(rgb_packed: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
    """Unpack uint32 packed ABGR to float32 RGB array.

    Args:
        rgb_packed: (nworld, total_pixels) uint32 packed pixel data.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        (nworld, height, width, 3) float32 array with values in [0, 1].
    """
    r = ((rgb_packed >> 16) & 0xFF).astype(jnp.float32) / 255.0
    g = ((rgb_packed >> 8) & 0xFF).astype(jnp.float32) / 255.0
    b = (rgb_packed & 0xFF).astype(jnp.float32) / 255.0
    nworld = rgb_packed.shape[0]
    rgb = jnp.stack([r, g, b], axis=-1)
    return rgb.reshape(nworld, height, width, 3)


def _unpack_grayscale(rgb_packed: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
    """Unpack uint32 packed ABGR to float32 grayscale array.

    Uses the standard luminance formula: 0.299*R + 0.587*G + 0.114*B.

    Args:
        rgb_packed: (nworld, total_pixels) uint32 packed pixel data.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        (nworld, height, width, 1) float32 array with values in [0, 1].
    """
    r = ((rgb_packed >> 16) & 0xFF).astype(jnp.float32)
    g = ((rgb_packed >> 8) & 0xFF).astype(jnp.float32)
    b = (rgb_packed & 0xFF).astype(jnp.float32)
    gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    nworld = rgb_packed.shape[0]
    return gray.reshape(nworld, height, width, 1)


class JaxVisionRenderer:
    """JAX-native vision renderer using the official MJX render API.

    Uses ``mjx.create_render_context`` for setup, ``mjx.refit_bvh`` to
    update the BVH for the current pose, and ``mjx.render`` to produce
    packed pixel buffers. The render context handles memory management
    automatically via its destructor.

    The render function is JAX-traceable and works inside ``jax.jit``
    and ``jax.lax.scan``.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mjx_model: mjx.Model,
        nworld: int,
        width: int = 32,
        height: int = 32,
        grayscale: bool = True,
        render_depth: bool = False,
        use_textures: bool = False,
        use_shadows: bool = False,
        enabled_geom_groups: list[int] | None = None,
        camera_name: str | None = None,
    ):
        """Initialize the JAX vision renderer.

        Args:
            mj_model: Host MuJoCo model.
            mjx_model: MJX model (warp backend). Needed for refit_bvh/render.
            nworld: Number of parallel worlds (batch size).
            width: Render width in pixels.
            height: Render height in pixels.
            grayscale: If True, output single-channel grayscale images.
            render_depth: If True, also output depth buffer.
            use_textures: Enable texture rendering.
            use_shadows: Enable shadow rendering.
            enabled_geom_groups: Geom groups to render (default [0, 1, 2]).
            camera_name: If set, only render this camera. Otherwise render all.
        """
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._render_depth = render_depth
        self._nworld = nworld
        self._mjx_model = mjx_model

        # Determine which cameras to render
        cam_active = None
        if camera_name is not None:
            cam_active = [False] * mj_model.ncam
            cam_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if cam_id < 0:
                raise ValueError(f"Camera '{camera_name}' not found in model")
            cam_active[cam_id] = True

        if enabled_geom_groups is None:
            enabled_geom_groups = [0, 1, 2]

        # Determine number of active cameras for per-camera flag lists
        if cam_active is not None:
            ncam_active = sum(cam_active)
        else:
            ncam_active = mj_model.ncam

        # Create the render context on ALL visible CUDA devices so that
        # pmap'd training can call refit_bvh / render from any device.
        devices = _get_warp_cuda_devices()
        logger.info(f"Creating render context on devices: {devices}")
        rc = mjx.create_render_context(
            mjm=mj_model,
            nworld=nworld,
            devices=devices,
            cam_res=(width, height),
            render_rgb=[True] * ncam_active,
            render_depth=[render_depth] * ncam_active,
            use_textures=use_textures,
            use_shadows=use_shadows,
            enabled_geom_groups=enabled_geom_groups,
            cam_active=cam_active,
        )
        # MJX now requires RenderContextPytree for refit_bvh/render.
        # Keep a reference to the original RenderContext so its __del__
        # doesn't remove warp buffers from the global registry.
        if hasattr(rc, "pytree"):
            self._rc_owner = rc
            self._ctx = rc.pytree()
        else:
            self._ctx = rc

        logger.info(
            f"JaxVisionRenderer initialized: {nworld} worlds, "
            f"{width}x{height}, {'grayscale' if grayscale else 'RGB'}"
        )

    @property
    def vision_shape(self) -> tuple[int, int, int]:
        """Output image shape: (height, width, channels)."""
        channels = 1 if self._grayscale else 3
        return (self._height, self._width, channels)

    def render(self, data: mjx.Data) -> jnp.ndarray:
        """Render from mjx.Data.

        This is the main entry point for rendering during env.step().
        JAX-traceable -- works inside jax.jit and jax.lax.scan.

        Calls refit_bvh to update the BVH for the current pose, then
        renders the scene to produce packed pixel buffers.

        Args:
            data: mjx.Data with kinematic state after physics step.
                Must have batch dimension (nworld, ...) -- not single-world.

        Returns:
            Float32 images of shape (nworld, height, width, channels).
            If render_depth=True, returns (images, depth) tuple.
        """
        mx = self._mjx_model
        ctx = self._ctx

        # Refit BVH for current pose (returns updated data)
        data = mjx.refit_bvh(mx, data, ctx)

        # Render -- always returns (rgb_packed, depth_packed) tuple
        rgb_packed, depth_packed = mjx.render(mx, data, ctx)

        # Unpack pixels
        if self._grayscale:
            images = _unpack_grayscale(rgb_packed, self._height, self._width)
        else:
            images = _unpack_rgb(rgb_packed, self._height, self._width)

        if self._render_depth:
            return images, depth_packed
        return images


class VisionRenderWrapper:
    """Brax-compatible wrapper that adds vision rendering to a batched env.

    Wraps an already-batched environment (after VmapWrapper) and renders
    vision observations on the full batch of worlds at each step.

    The rendering is JAX-traceable -- it works inside jax.jit and
    jax.lax.scan. No Python-level rendering loop is needed.

    This wrapper must go OUTSIDE the Brax training wrappers (VmapWrapper,
    EpisodeWrapper, AutoResetWrapper) because the warp renderer needs
    all worlds at once (not per-world via vmap).

    The renderer is lazily initialized on the first ``reset()`` call,
    detecting ``nworld`` from the actual batch size.  This allows the
    same ``wrap_for_training`` callable to be used for both training and
    eval environments (which may have different batch sizes).
    """

    def __init__(
        self,
        env: Any,
        mj_model: mujoco.MjModel,
        mjx_model: mjx.Model,
        nworld: int | None = None,
        width: int = 32,
        height: int = 32,
        grayscale: bool = True,
        render_depth: bool = False,
        use_textures: bool = False,
        use_shadows: bool = False,
        camera_name: str | None = None,
    ):
        """Initialize the vision render wrapper.

        Args:
            env: Inner (already-batched) environment.
            mj_model: Host MuJoCo model.
            mjx_model: MJX model (warp backend). Needed for render calls.
            nworld: Number of parallel worlds.  If ``None``, detected
                automatically from the batch size on the first ``reset()``.
            width: Render width in pixels.
            height: Render height in pixels.
            grayscale: If True, output single-channel grayscale.
            render_depth: If True, also output depth buffer.
            use_textures: Enable texture rendering.
            use_shadows: Enable shadow rendering.
            camera_name: Camera to render (None = all cameras).
        """
        self.env = env
        self._mj_model = mj_model
        self._mjx_model = mjx_model
        self._renderer_kwargs = dict(
            width=width,
            height=height,
            grayscale=grayscale,
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
            camera_name=camera_name,
        )

        if nworld is not None:
            self._renderer = JaxVisionRenderer(
                mj_model=mj_model,
                mjx_model=mjx_model,
                nworld=nworld,
                **self._renderer_kwargs,
            )
        else:
            self._renderer = None  # lazy init on first reset

    def _ensure_renderer(self, nworld: int) -> None:
        """Create the renderer if it hasn't been created yet."""
        if self._renderer is None:
            self._renderer = JaxVisionRenderer(
                mj_model=self._mj_model,
                mjx_model=self._mjx_model,
                nworld=nworld,
                **self._renderer_kwargs,
            )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the inner env."""
        return getattr(self.env, name)

    @property
    def renderer(self) -> JaxVisionRenderer | None:
        """The underlying JaxVisionRenderer (None until first reset)."""
        return self._renderer

    @staticmethod
    def _inject_vision(obs, vision):
        """Replace vision placeholders in the obs dict with rendered images.

        Handles two obs layouts:

        1. Raw env (OrderedDict)::

            OrderedDict(
                state=OrderedDict(task_obs, proprioception, vision=zeros),
                privileged_state=OrderedDict(...)
            )

        2. HighLevelWrapper (plain dict)::

            {"imitation_target": ..., "proprioception": ..., "vision": zeros}

        Preserves the exact container types for lax.scan compatibility.
        """
        # Top-level "vision" key (e.g. HighLevelWrapper output)
        if "vision" in obs:
            new_obs = type(obs)(
                [(k, vision if k == "vision" else v) for k, v in obs.items()]
            )
            return new_obs

        # Nested "vision" key inside sub-dicts (raw env output)
        new_obs = type(obs)()
        for key, val in obs.items():
            if isinstance(val, dict) and "vision" in val:
                new_inner = type(val)(
                    [(k, vision if k == "vision" else v) for k, v in val.items()]
                )
                new_obs[key] = new_inner
            else:
                new_obs[key] = val
        return new_obs

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment and render initial vision observations."""
        state = self.env.reset(rng)
        self._ensure_renderer(rng.shape[0])
        vision = self._renderer.render(state.data)
        state = state.replace(obs=self._inject_vision(state.obs, vision))
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment and render vision observations.

        Calls the inner env's step (which handles physics, episode wrapping,
        and auto-reset), then renders vision on the batched data.
        """
        state = self.env.step(state, action)
        vision = self._renderer.render(state.data)
        state = state.replace(obs=self._inject_vision(state.obs, vision))
        return state


class BinocularVisionRenderWrapper:
    """Brax-compatible wrapper that renders binocular (stereo) vision.

    Creates two JaxVisionRenderer instances — one for each eye camera —
    and concatenates the rendered images along the channel dimension.
    Output shape per world: (H, W, 2*C) where C=1 (grayscale) or C=3 (RGB).

    The concatenated image is injected into the observation dict under
    the standard "vision" key, maintaining compatibility with
    HighLevelWrapper and _inject_vision.

    Channel layout: [left_channels..., right_channels...]
    For grayscale: (H, W, 2) — channel 0 = left eye, channel 1 = right eye.
    """

    def __init__(
        self,
        env,
        mj_model,
        mjx_model,
        nworld=None,
        width=32,
        height=32,
        grayscale=True,
        render_depth=False,
        use_textures=False,
        use_shadows=False,
        left_camera_name="eye_left-rodent",
        right_camera_name="eye_right-rodent",
        eye_dropout_rate=0.0,
        eval_eye_mode="binocular",
        vision_noise_std=0.0,
    ):
        self.env = env
        self._mj_model = mj_model
        self._mjx_model = mjx_model
        self._left_camera_name = left_camera_name
        self._right_camera_name = right_camera_name
        self._eye_dropout_rate = eye_dropout_rate
        self._eval_eye_mode = eval_eye_mode
        self._vision_noise_std = vision_noise_std
        self._renderer_kwargs = dict(
            width=width,
            height=height,
            grayscale=grayscale,
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
        )

        if nworld is not None:
            self._left_renderer = JaxVisionRenderer(
                mj_model=mj_model,
                mjx_model=mjx_model,
                nworld=nworld,
                camera_name=left_camera_name,
                **self._renderer_kwargs,
            )
            self._right_renderer = JaxVisionRenderer(
                mj_model=mj_model,
                mjx_model=mjx_model,
                nworld=nworld,
                camera_name=right_camera_name,
                **self._renderer_kwargs,
            )
        else:
            self._left_renderer = None
            self._right_renderer = None

    def _ensure_renderers(self, nworld):
        if self._left_renderer is None:
            self._left_renderer = JaxVisionRenderer(
                mj_model=self._mj_model,
                mjx_model=self._mjx_model,
                nworld=nworld,
                camera_name=self._left_camera_name,
                **self._renderer_kwargs,
            )
            self._right_renderer = JaxVisionRenderer(
                mj_model=self._mj_model,
                mjx_model=self._mjx_model,
                nworld=nworld,
                camera_name=self._right_camera_name,
                **self._renderer_kwargs,
            )

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def left_renderer(self):
        """The left-eye JaxVisionRenderer (None until first reset)."""
        return self._left_renderer

    @property
    def right_renderer(self):
        """The right-eye JaxVisionRenderer (None until first reset)."""
        return self._right_renderer

    def _render_binocular(self, data):
        left = self._left_renderer.render(data)  # (nworld, H, W, C)
        right = self._right_renderer.render(data)  # (nworld, H, W, C)
        return jnp.concatenate([left, right], axis=-1)  # (nworld, H, W, 2C)

    def _apply_vision_noise(self, vision, rng):
        """Add per-step i.i.d. Gaussian sensor noise: ``v + sigma*N(0,1)``.

        E3 (`_implementation_log/DMPO/NEXT_EXPERIMENTS.md`): degrade vision to
        see whether the rat starts deliberating at gap edges instead of
        committing ballistically.

        Clipped to [0, 1] because that is the renderer's range -- the packed
        RGB is divided by 255.0 at the top of this module -- so the noisy image
        stays in the same domain the CNN encoder was trained on.

        i.i.d. over (world, row, col, channel): one `jax.random.normal` call
        over the full shape, so the two eyes are independent and there is no
        spatial correlation. Applied to BOTH training and eval, since
        deliberation has to be trained under noise, not merely probed under it.

        Noise is added BEFORE `_apply_eye_mask`, so a dropped-out eye still
        reads exactly 0 rather than acquiring a noise floor.

        NOTE ON QUANTIZATION: with `vision_uint8_storage: true` the replay
        buffer stores 8-bit pixels, so noise below 1/255 ~ 0.004 is quantized
        away. The E3 ladder starts at sigma=0.05, an order of magnitude above
        that, so the stored noise is faithful.

        Args:
            vision: (nworld, H, W, 2*C) rendered binocular images in [0, 1].
            rng: JAX PRNG key for this step's draw.

        Returns:
            Noisy vision, same shape/dtype, still within [0, 1].
        """
        # <= 0 rather than == 0: sigma=0 is the E3 control arm and must be a
        # bit-identical no-op, and a negative sigma is a config typo, not a
        # request to flip the sign of the noise.
        if self._vision_noise_std <= 0.0:
            return vision

        noise = jax.random.normal(rng, vision.shape, dtype=vision.dtype)
        return jnp.clip(vision + self._vision_noise_std * noise, 0.0, 1.0)

    def _apply_eye_mask(self, vision, rng):
        """Stochastically zero out one eye's channels for monocular dropout.

        With probability ``eye_dropout_rate``, one eye is randomly selected
        and its channels are zeroed. Each eye has equal probability (50/50)
        of being the one masked.

        Args:
            vision: (nworld, H, W, 2*C) rendered binocular images.
            rng: JAX PRNG key for sampling.

        Returns:
            Masked vision with same shape as input.
        """
        if self._eye_dropout_rate <= 0.0:
            return vision

        nworld = vision.shape[0]
        c = vision.shape[-1] // 2  # channels per eye

        rng1, rng2 = jax.random.split(rng)

        # Per-world: should we apply dropout?
        do_dropout = (
            jax.random.uniform(rng1, (nworld, 1, 1, 1)) < self._eye_dropout_rate
        )

        # Per-world: which eye to zero? True -> zero left, False -> zero right
        zero_left = jax.random.uniform(rng2, (nworld, 1, 1, 1)) < 0.5

        # Build per-channel masks
        left_mask = jnp.where(do_dropout & zero_left, 0.0, 1.0)
        right_mask = jnp.where(do_dropout & ~zero_left, 0.0, 1.0)

        mask = jnp.concatenate(
            [
                jnp.broadcast_to(left_mask, (*vision.shape[:-1], c)),
                jnp.broadcast_to(right_mask, (*vision.shape[:-1], c)),
            ],
            axis=-1,
        )
        return vision * mask

    def _apply_eval_eye_mask(self, vision):
        """Deterministically mask one eye for evaluation.

        Used at eval time to test monocular performance fairly (since the
        network was trained with stochastic eye dropout).

        Args:
            vision: (nworld, H, W, 2*C) rendered binocular images.

        Returns:
            Masked vision. Unchanged if mode is "binocular".
        """
        if self._eval_eye_mode == "binocular":
            return vision

        c = vision.shape[-1] // 2
        if self._eval_eye_mode == "left_only":
            # Zero right eye channels
            mask = jnp.concatenate(
                [
                    jnp.ones((*vision.shape[:-1], c)),
                    jnp.zeros((*vision.shape[:-1], c)),
                ],
                axis=-1,
            )
        elif self._eval_eye_mode == "right_only":
            # Zero left eye channels
            mask = jnp.concatenate(
                [
                    jnp.zeros((*vision.shape[:-1], c)),
                    jnp.ones((*vision.shape[:-1], c)),
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Unknown eval_eye_mode: {self._eval_eye_mode!r}. "
                f"Expected 'binocular', 'left_only', or 'right_only'."
            )
        return vision * mask

    def reset(self, rng):
        state = self.env.reset(rng)
        self._ensure_renderers(rng.shape[0])
        vision = self._render_binocular(state.data)

        # Sensor noise FIRST, so a dropped eye below still reads exactly 0.
        # The immediate draw uses split[0] while the stream stored for future
        # steps is split[1]: reusing one key for both would make the reset
        # frame and the first step frame carry identical noise.
        if self._vision_noise_std > 0.0:
            noise_rng, _ = jax.random.split(rng[0])
            vision = self._apply_vision_noise(vision, noise_rng)
            state.info["vision_noise_rng"] = jax.vmap(jax.random.split)(rng)[:, 1]

        # Stochastic eye dropout (training) or deterministic masking (eval)
        if self._eye_dropout_rate > 0.0:
            mask_rng, _ = jax.random.split(rng[0])
            vision = self._apply_eye_mask(vision, mask_rng)
            # Store per-world RNG keys (must be batched for AutoResetWrapper)
            state.info["eye_mask_rng"] = jax.vmap(jax.random.split)(rng)[:, 0]
        else:
            vision = self._apply_eval_eye_mask(vision)

        state = state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        vision = self._render_binocular(state.data)

        # Sensor noise FIRST (see reset). World 0's key seeds the whole batch's
        # draw -- jax.random.normal over the full (nworld, H, W, C) shape still
        # gives every world independent pixels -- then every world's key is
        # advanced, mirroring the eye-dropout stream.
        if self._vision_noise_std > 0.0:
            noise_rng = state.info["vision_noise_rng"][0]
            vision = self._apply_vision_noise(vision, noise_rng)
            state.info["vision_noise_rng"] = jax.vmap(
                lambda k: jax.random.split(k)[0]
            )(state.info["vision_noise_rng"])

        # Stochastic eye dropout (training) or deterministic masking (eval)
        if self._eye_dropout_rate > 0.0:
            # Use world 0's key for batch masking, advance all worlds' keys
            mask_rng = state.info["eye_mask_rng"][0]
            vision = self._apply_eye_mask(vision, mask_rng)
            state.info["eye_mask_rng"] = jax.vmap(lambda k: jax.random.split(k)[0])(
                state.info["eye_mask_rng"]
            )
        else:
            vision = self._apply_eval_eye_mask(vision)

        state = state.replace(obs=VisionRenderWrapper._inject_vision(state.obs, vision))
        return state
