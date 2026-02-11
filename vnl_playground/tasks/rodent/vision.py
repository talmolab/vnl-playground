"""Vision rendering utilities using mujoco_warp GPU batch renderer.

Provides a VisionRenderer class that wraps the mujoco_warp rendering API
to produce egocentric camera observations for batched environments.

Note: mujoco_warp rendering is NOT differentiable and runs outside JAX.
Images are injected as external observations into the JAX computation.

The standalone ``mujoco_warp`` package (https://github.com/google-deepmind/mujoco_warp)
is required for the rendering functions (``create_render_context``, ``render``,
``refit_bvh``). These are not included in the vendored mujoco_warp inside
``mujoco-mjx``.

If ``mujoco_warp`` is not pip-installed, set the ``MUJOCO_WARP_PATH`` environment
variable to point to the cloned repo directory.
"""

import logging
import os
import sys

import numpy as np
import mujoco
import warp as wp

logger = logging.getLogger(__name__)


def _import_mujoco_warp():
    """Import mujoco_warp with rendering support.

    Tries multiple strategies:
      1. Direct ``import mujoco_warp`` (pip-installed or on sys.path).
      2. ``MUJOCO_WARP_PATH`` environment variable pointing to the repo.
      3. Common development locations relative to the SalkResearch workspace.

    Returns:
        The ``mujoco_warp`` module with rendering functions.

    Raises:
        ImportError: If mujoco_warp with rendering support cannot be found.
    """
    # Strategy 1: Try direct import
    try:
        import mujoco_warp as mjw

        if hasattr(mjw, "create_render_context"):
            return mjw
        else:
            logger.warning(
                "Found mujoco_warp but it lacks rendering functions "
                "(create_render_context, render, refit_bvh). "
                "This may be the vendored version from mujoco-mjx."
            )
    except ImportError:
        pass

    # Strategy 2: Check MUJOCO_WARP_PATH environment variable
    mjw_path = os.environ.get("MUJOCO_WARP_PATH")
    if mjw_path and os.path.isdir(mjw_path):
        if mjw_path not in sys.path:
            sys.path.insert(0, mjw_path)
        try:
            import mujoco_warp as mjw

            if hasattr(mjw, "create_render_context"):
                logger.info(f"Using mujoco_warp from MUJOCO_WARP_PATH={mjw_path}")
                return mjw
        except ImportError:
            pass

    # Strategy 3: Try common development locations
    search_paths = []

    # Relative to this file's workspace
    workspace_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
    )
    search_paths.append(os.path.join(workspace_root, "mujoco_warp"))
    search_paths.append(os.path.join(workspace_root, "mujoco-warp"))

    for path in search_paths:
        if os.path.isdir(path) and os.path.isfile(
            os.path.join(path, "mujoco_warp", "__init__.py")
        ):
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                import mujoco_warp as mjw

                if hasattr(mjw, "create_render_context"):
                    logger.info(f"Using mujoco_warp from {path}")
                    return mjw
            except ImportError:
                pass

    raise ImportError(
        "Could not import mujoco_warp with rendering support. "
        "The standalone mujoco_warp package is required for GPU rendering. "
        "Install it from https://github.com/google-deepmind/mujoco_warp "
        "or set MUJOCO_WARP_PATH to the cloned repo directory."
    )


mjw = _import_mujoco_warp()


class VisionRenderer:
    """GPU-accelerated batch renderer for egocentric vision observations.

    Wraps mujoco_warp's ray-tracing renderer to produce RGB images from
    the egocentric camera defined in the rodent MJCF.

    This renderer maintains its own native mujoco_warp Model and Data objects
    on the GPU. Before each render call, the caller must sync the physics
    state from JAX/mjx Data into the warp Data via the ``sync_state`` method,
    then call ``render``.

    Typical usage in a training loop::

        renderer = VisionRenderer(mj_model, nworld=num_envs)

        # After mjx.step():
        renderer.sync_state(mjx_data)
        rgb, depth = renderer.render()

    Args:
        mj_model: Compiled MuJoCo model (``mujoco.MjModel``).
        nworld: Number of parallel simulation worlds (batch size).
        camera_name: Name of the camera in the MJCF (default: "egocentric-rodent").
        width: Image width in pixels (default: 64).
        height: Image height in pixels (default: 64).
        render_depth: Whether to also render depth (default: False).
        use_textures: Whether to use textures in rendering (default: True).
        use_shadows: Whether to use shadow rendering (default: False).
        nconmax: Maximum contacts per world for warp Data. If None, uses default.
        njmax: Maximum constraints per world for warp Data. If None, uses default.
        naconmax: Maximum contacts across all worlds. If None, computed from nconmax.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        nworld: int = 1,
        camera_name: str = "egocentric-rodent",
        width: int = 64,
        height: int = 64,
        render_depth: bool = False,
        use_textures: bool = True,
        use_shadows: bool = False,
        nconmax: int | None = None,
        njmax: int | None = None,
        naconmax: int | None = None,
    ):
        self._mj_model = mj_model
        self._nworld = nworld
        self._width = width
        self._height = height
        self._render_depth = render_depth

        # Find camera index by name
        cam_names = [
            mj_model.camera(i).name for i in range(mj_model.ncam)
        ]
        if camera_name not in cam_names:
            raise ValueError(
                f"Camera '{camera_name}' not found in model. "
                f"Available cameras: {cam_names}"
            )
        self._cam_idx = cam_names.index(camera_name)

        # Create native mujoco_warp model on GPU
        self._mjw_model = mjw.put_model(mj_model)

        # Create native mujoco_warp data on GPU (batch of nworld)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        put_data_kwargs = {"nworld": nworld}
        if nconmax is not None:
            put_data_kwargs["nconmax"] = nconmax
        if njmax is not None:
            put_data_kwargs["njmax"] = njmax
        if naconmax is not None:
            put_data_kwargs["naconmax"] = naconmax
        self._mjw_data = mjw.put_data(mj_model, mj_data, **put_data_kwargs)

        # Create render context - only activate the egocentric camera
        cam_active = [False] * mj_model.ncam
        cam_active[self._cam_idx] = True

        self._render_context = mjw.create_render_context(
            mj_model,
            self._mjw_model,
            self._mjw_data,
            cam_res=(width, height),
            render_rgb=True,
            render_depth=render_depth,
            cam_active=cam_active,
            use_textures=use_textures,
            use_shadows=use_shadows,
        )

        # Pre-compute the address of our camera in the render context output
        rgb_adr = self._render_context.rgb_adr.numpy()
        self._rgb_adr = int(rgb_adr[0])  # We only have 1 active camera (index 0)
        self._n_pixels = width * height

        if render_depth:
            depth_adr = self._render_context.depth_adr.numpy()
            self._depth_adr = int(depth_adr[0])
        else:
            self._depth_adr = -1

    def sync_state(self, mjx_data) -> None:
        """Sync physics state from JAX mjx.Data into native warp Data.

        Copies the geometry, camera, and light positions/orientations from
        the batched JAX arrays into the native mujoco_warp Data arrays.
        This must be called after ``mjx.step()`` and before ``render()``.

        The following fields are synced (all needed for rendering):
          - ``geom_xpos``, ``geom_xmat`` (geometry positions and orientations)
          - ``cam_xpos``, ``cam_xmat`` (camera positions and orientations)
          - ``light_xpos``, ``light_xdir`` (light positions and directions)

        Args:
            mjx_data: Batched mjx.Data object from the simulation. Must have
                shape ``(nworld, ...)`` for each field.
        """
        # Copy rendering-relevant fields from JAX -> warp
        # wp.from_jax creates zero-copy warp array views of JAX arrays,
        # then wp.copy does a device-to-device memcpy into our warp data.
        field_specs = [
            ("geom_xpos", wp.vec3),
            ("geom_xmat", wp.mat33),
            ("cam_xpos", wp.vec3),
            ("cam_xmat", wp.mat33),
            ("light_xpos", wp.vec3),
            ("light_xdir", wp.vec3),
        ]

        for field_name, dtype in field_specs:
            jax_arr = getattr(mjx_data, field_name)
            if jax_arr.size == 0:
                continue
            src = wp.from_jax(jax_arr, dtype=dtype)
            dst = getattr(self._mjw_data, field_name)
            wp.copy(dst, src)

    def render(self):
        """Render the current scene from the egocentric camera.

        Must be called AFTER ``sync_state()`` to ensure the warp data
        contains up-to-date physics state.

        Returns:
            rgb: ``np.ndarray`` of shape ``(nworld, height, width, 3)``,
                dtype ``uint8``. RGB pixel values.
            depth: ``np.ndarray`` of shape ``(nworld, height, width)``,
                dtype ``float32``, or ``None`` if ``render_depth=False``.
                Depth values in meters.
        """
        # Update BVH for the new geometry positions
        mjw.refit_bvh(self._mjw_model, self._mjw_data, self._render_context)
        # Render all worlds in parallel
        mjw.render(self._mjw_model, self._mjw_data, self._render_context)

        # Extract and unpack RGB data
        # rgb_data is shape (nworld, total_rgb_pixels) dtype uint32
        rgb_all = self._render_context.rgb_data.numpy()
        # Slice to our camera's pixel range
        rgb_packed = rgb_all[:, self._rgb_adr : self._rgb_adr + self._n_pixels]

        # Unpack uint32 RGBA: (a << 24) | (r << 16) | (g << 8) | b
        rgb_flat = np.zeros(
            (self._nworld, self._n_pixels, 3), dtype=np.uint8
        )
        rgb_flat[..., 0] = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)  # R
        rgb_flat[..., 1] = ((rgb_packed >> 8) & 0xFF).astype(np.uint8)   # G
        rgb_flat[..., 2] = (rgb_packed & 0xFF).astype(np.uint8)          # B

        rgb = rgb_flat.reshape(self._nworld, self._height, self._width, 3)

        depth = None
        if self._render_depth and self._depth_adr >= 0:
            depth_all = self._render_context.depth_data.numpy()
            depth_flat = depth_all[
                :, self._depth_adr : self._depth_adr + self._n_pixels
            ]
            depth = depth_flat.reshape(
                self._nworld, self._height, self._width
            )

        return rgb, depth

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Shape of a single RGB image: ``(height, width, 3)``."""
        return (self._height, self._width, 3)

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._height

    @property
    def nworld(self) -> int:
        """Number of parallel worlds."""
        return self._nworld

    @property
    def render_context(self):
        """The underlying mujoco_warp RenderContext."""
        return self._render_context

    @property
    def mjw_model(self):
        """The underlying mujoco_warp Model."""
        return self._mjw_model

    @property
    def mjw_data(self):
        """The underlying mujoco_warp Data."""
        return self._mjw_data
