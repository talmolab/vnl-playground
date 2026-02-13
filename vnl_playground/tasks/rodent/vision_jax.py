"""JAX-native vision rendering using the MJX warp renderer.

Provides two main components:

1. ``JaxVisionRenderer`` — wraps ``create_mjx_render_fn`` from render_jax.py
   to produce a JAX-callable render function.

2. ``VisionRenderWrapper`` — a Brax-compatible environment wrapper that
   renders vision observations on batched data after each vmapped step.

The rendering is JAX-traceable: it works inside ``jax.jit`` and
``jax.lax.scan`` with no Python-level sync needed.

Usage::

    from vnl_playground.tasks.rodent.vision_jax import VisionRenderWrapper

    raw_env = RunGapVision(config=cfg)
    brax_env = wrap_for_brax_training(raw_env, ...)
    env = VisionRenderWrapper(
        brax_env, raw_env.mj_model, nworld=num_envs,
        width=32, height=32, grayscale=True,
        camera_name="egocentric-rodent",
    )
    # env.step() now produces real vision observations
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
import mujoco

from mujoco_playground._src import mjx_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import mujoco_warp with rendering support (same strategy as vision.py)
# ---------------------------------------------------------------------------

def _import_mujoco_warp():
    """Import mujoco_warp with rendering support."""
    import os
    import sys

    # Strategy 1: Direct import
    try:
        import mujoco_warp as mjw
        if hasattr(mjw, "create_render_context"):
            return mjw
    except ImportError:
        pass

    sys.modules.pop("mujoco_warp", None)

    # Strategy 2: MUJOCO_WARP_PATH env var
    mjw_path = os.environ.get("MUJOCO_WARP_PATH")
    if mjw_path and os.path.isdir(mjw_path):
        if mjw_path not in sys.path:
            sys.path.insert(0, mjw_path)
        try:
            import mujoco_warp as mjw
            if hasattr(mjw, "create_render_context"):
                return mjw
        except ImportError:
            pass

    sys.modules.pop("mujoco_warp", None)

    # Strategy 3: Common dev locations
    workspace_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
    )
    for name in ("mujoco_warp", "mujoco-warp"):
        path = os.path.join(workspace_root, name)
        if os.path.isdir(path) and os.path.isfile(
            os.path.join(path, "mujoco_warp", "__init__.py")
        ):
            if path not in sys.path:
                sys.path.insert(0, path)
            sys.modules.pop("mujoco_warp", None)
            try:
                import mujoco_warp as mjw
                if hasattr(mjw, "create_render_context"):
                    return mjw
            except ImportError:
                pass

    raise ImportError(
        "Could not find mujoco_warp with rendering support. "
        "Install the standalone mujoco_warp package or set MUJOCO_WARP_PATH."
    )


def _import_render_jax():
    """Import the render_jax module from the MJX warp backend."""
    try:
        from mujoco.mjx.warp import render_jax
        return render_jax
    except ImportError:
        pass

    # Fallback: try relative to SalkResearch workspace
    import os
    import sys
    workspace_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
    )
    mjx_warp_path = os.path.join(workspace_root, "mujoco", "mjx")
    if mjx_warp_path not in sys.path:
        sys.path.insert(0, mjx_warp_path)
    from mujoco.mjx.warp import render_jax
    return render_jax


class JaxVisionRenderer:
    """JAX-native vision renderer using the MJX warp FFI bridge.

    Unlike the old VisionRenderer, this renderer produces a JAX-callable
    function that works inside jax.jit and jax.lax.scan. No Python-level
    sync_state/render calls are needed.

    The render function is created once at init time, capturing the warp
    Model/Data/RenderContext in a closure. Only dynamic kinematic arrays
    (geometry, camera, light poses) flow through JAX.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
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

        mjwarp = _import_mujoco_warp()
        render_jax = _import_render_jax()

        # Create host MjData for warp initialization
        mjd = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mjd)

        # Transfer to warp GPU memory
        m_warp = mjwarp.put_model(mj_model)
        d_warp = mjwarp.put_data(mj_model, mjd, nworld=nworld)
        mjwarp.forward(m_warp, d_warp)

        # Determine which cameras to render
        cam_active = None
        if camera_name is not None:
            cam_active = [False] * mj_model.ncam
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                raise ValueError(f"Camera '{camera_name}' not found in model")
            cam_active[cam_id] = True

        # Create JAX-callable render function
        self._render_fn, self._info = render_jax.create_mjx_render_fn(
            mj_model,
            m_warp,
            d_warp,
            cam_res=(width, height),
            render_depth=render_depth,
            use_textures=use_textures,
            use_shadows=use_shadows,
            enabled_geom_groups=enabled_geom_groups,
            cam_active=cam_active,
        )

        # Store unpack functions
        self._unpack_rgb = render_jax.unpack_rgb
        self._unpack_grayscale = render_jax.unpack_grayscale

        logger.info(
            f"JaxVisionRenderer initialized: {nworld} worlds, "
            f"{width}x{height}, {'grayscale' if grayscale else 'RGB'}, "
            f"total_pixels={self._info['total_pixels']}"
        )

    @property
    def info(self) -> dict:
        """Render info dict with shapes and metadata."""
        return self._info

    @property
    def vision_shape(self) -> tuple[int, int, int]:
        """Output image shape: (height, width, channels)."""
        channels = 1 if self._grayscale else 3
        return (self._height, self._width, channels)

    def render_from_arrays(
        self,
        geom_xpos: jnp.ndarray,
        geom_xmat: jnp.ndarray,
        cam_xpos: jnp.ndarray,
        cam_xmat: jnp.ndarray,
        light_xpos: jnp.ndarray,
        light_xdir: jnp.ndarray,
    ) -> jnp.ndarray:
        """Render from explicit kinematic arrays.

        All inputs are JAX arrays. This function is JAX-traceable and can be
        used inside jax.jit and jax.lax.scan.

        Args:
            geom_xpos: Geometry positions (nworld, ngeom, 3).
            geom_xmat: Geometry rotation matrices (nworld, ngeom, 3, 3).
            cam_xpos: Camera positions (nworld, ncam, 3).
            cam_xmat: Camera rotation matrices (nworld, ncam, 3, 3).
            light_xpos: Light positions (nworld, nlight, 3).
            light_xdir: Light directions (nworld, nlight, 3).

        Returns:
            Float32 images of shape (nworld, height, width, channels).
            If render_depth=True, returns (images, depth) tuple.
        """
        if self._render_depth:
            rgb_packed, depth_packed = self._render_fn(
                geom_xpos, geom_xmat, cam_xpos, cam_xmat,
                light_xpos, light_xdir,
            )
        else:
            rgb_packed = self._render_fn(
                geom_xpos, geom_xmat, cam_xpos, cam_xmat,
                light_xpos, light_xdir,
            )

        if self._grayscale:
            images = self._unpack_grayscale(rgb_packed, self._height, self._width)
        else:
            images = self._unpack_rgb(rgb_packed, self._height, self._width)

        if self._render_depth:
            return images, depth_packed
        return images

    def render(self, data) -> jnp.ndarray:
        """Render from mjx.Data, extracting the needed kinematic arrays.

        This is the main entry point for rendering during env.step().
        JAX-traceable — works inside jax.jit and jax.lax.scan.

        Args:
            data: mjx.Data with kinematic state after physics step.
                Must have batch dimension (nworld, ...) — not single-world.

        Returns:
            Float32 images of shape (nworld, height, width, channels).
            If render_depth=True, returns (images, depth) tuple.
        """
        return self.render_from_arrays(
            data.geom_xpos,
            data.geom_xmat,
            data.cam_xpos,
            data.cam_xmat,
            data._impl.light_xpos,
            data._impl.light_xdir,
        )


class VisionRenderWrapper:
    """Brax-compatible wrapper that adds vision rendering to a batched env.

    Wraps an already-batched environment (after VmapWrapper) and renders
    vision observations on the full batch of worlds at each step.

    The rendering is JAX-traceable — it works inside jax.jit and
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
                mj_model=mj_model, nworld=nworld, **self._renderer_kwargs,
            )
        else:
            self._renderer = None  # lazy init on first reset

    def _ensure_renderer(self, nworld: int) -> None:
        """Create the renderer if it hasn't been created yet."""
        if self._renderer is None:
            self._renderer = JaxVisionRenderer(
                mj_model=self._mj_model,
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

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment and render initial vision observations."""
        state = self.env.reset(rng)
        self._ensure_renderer(rng.shape[0])
        vision = self._renderer.render(state.data)
        state = state.replace(obs={**state.obs, "vision": vision})
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment and render vision observations.

        Calls the inner env's step (which handles physics, episode wrapping,
        and auto-reset), then renders vision on the batched data.
        """
        state = self.env.step(state, action)
        vision = self._renderer.render(state.data)
        state = state.replace(obs={**state.obs, "vision": vision})
        return state
