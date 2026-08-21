"""Offline render harness for ``MazeForageVision`` validation figures.

Three things this script exists to get right, all of which were measured on an
RTX 5090 with ``mujoco 3.9.0`` / ``warp 1.12.1`` before being written down here:

1. **Third-person views of the whole maze.**  Every camera compiled into this
   model is attached to the rodent body -- there is no world-fixed overview
   camera -- so a maze overview has to come from a *free* ``MjvCamera``.  The
   framing constants below are calibrated against the maze's own extent, not
   guessed (see :func:`overview_camera`).

2. **The pixels the CNN actually receives.**  ``mujoco.Renderer`` pointed at
   ``egocentric-rodent`` produces a *different image* from the one the policy
   sees: the policy is fed the Warp ray-tracer (``mjx.render``) at 32x32
   grayscale with ``use_textures`` / ``use_shadows`` / ``enabled_geom_groups``
   from the config.  Only :class:`JaxVisionRenderer` reproduces it, and it is
   the same class the training wrapper uses, so ``--what ego`` is a real
   check on the observation and not an illustration of it.

3. **A batch of genuinely different reset states.**  ``jax.vmap(env.reset)``
   gives a batched ``mjx.Data``; ``qpos[i]`` + ``mj_forward`` is enough to
   redraw world ``i`` exactly (matched ``mjx.Data.xpos`` to 7e-8 m), because
   every per-episode degree of freedom in this task -- spawn pose *and* every
   treat position -- lives in ``qpos``.  That route is ~28x cheaper than
   ``mjx.get_data``.

Usage::

    MUJOCO_GL=egl python scripts/render_maze_forage.py --what all
    MUJOCO_GL=egl python scripts/render_maze_forage.py --what ego --batch 8

EGL, not osmesa: osmesa is ~80x slower on a GPU box and oversubscribes CPU
threads.  ``MUJOCO_GL`` is forced below because it has to be set *before*
``mujoco`` is imported.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import imageio.v2 as imageio
import jax
import mujoco
import numpy as np

from vnl_playground.tasks.rodent.maze_forage_vision import (
    MazeForageVision,
    default_config,
)
from vnl_playground.tasks.rodent.vision_jax import JaxVisionRenderer

# The free-camera vertical FOV MuJoCo uses is ``model.vis.global_.fovy`` (45 deg
# here), NOT any ``cam_fovy`` entry -- those belong to the model cameras.
#
# Perspective: ``distance = margin * half_extent / tan(fovy / 2)`` puts the maze
# at ``1 / margin`` of the frame height; margin 1.15 measured at 83% of frame.
#
# Orthographic: the visible vertical extent is ~0.86-0.90 * ``cam.distance``
# (measured 2.59 m at distance 3.0, 4.41 m at distance 4.9), so
# ``distance = 1.22 * full_extent`` fills ~95% of a square frame.  Orthographic
# is the better choice for a maze *map*: perspective at elevation -90 splays the
# 0.3 m walls outward and hides floor next to the outer wall.
PERSPECTIVE_MARGIN = 1.15
ORTHO_DISTANCE_FACTOR = 1.22


def build_env(**overrides) -> MazeForageVision:
    """Builds the env from ``default_config()`` plus keyword overrides."""
    cfg = default_config()
    for key, value in overrides.items():
        cfg[key] = value
    return MazeForageVision(config=cfg)


def camera_table(env: MazeForageVision) -> str:
    """One line per compiled camera: name, mode, fovy, parent body."""
    modes = {0: "FIXED", 1: "TRACK", 2: "TRACKCOM", 3: "TARGETBODY", 4: "TARGETCOM"}
    m = env.mj_model
    lines = []
    for i in range(m.ncam):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.cam_bodyid[i]))
        lines.append(
            f"  [{i:2d}] {name:26s} mode={modes.get(int(m.cam_mode[i])):9s} "
            f"fovy={m.cam_fovy[i]:5.1f} body={body}"
        )
    return "\n".join(lines)


def overview_camera(env: MazeForageVision, orthographic: bool = True,
                    elevation: float = -90.0, azimuth: float = 90.0):
    """Free camera framed on the maze footprint, looking down by default.

    ``lookat`` is the world origin because ``maze_utils.grid_to_world`` centres
    the grid there; ``+x`` grows with the column index and ``+y`` *shrinks* with
    the row index, so an ``elevation=-90, azimuth=90`` image has world ``+x`` to
    the right and world ``+y`` up -- i.e. ``maze_grid`` row 0 at the top.
    """
    m = env.mj_model
    half = max(env._maze_half_extent)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.azimuth = azimuth
    cam.elevation = elevation
    if orthographic:
        cam.orthographic = 1
        cam.distance = ORTHO_DISTANCE_FACTOR * 2.0 * half
    else:
        fovy = float(m.vis.global_.fovy)
        cam.distance = PERSPECTIVE_MARGIN * half / np.tan(np.deg2rad(fovy) / 2.0)
    return cam


class Stage:
    """Offscreen EGL renderer bound to one compiled model.

    Drawing is stateless in ``qpos``: write it, ``mj_forward`` to resolve
    kinematics (no physics), draw.  ~0.9 ms/frame at 960x720 on a 5090.
    """

    def __init__(self, model: mujoco.MjModel, width: int, height: int,
                 shadows: bool = True):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.opt = mujoco.MjvOption()
        self.shadows = shadows

    def frame(self, qpos, cam) -> np.ndarray:
        self.data.qpos[:] = np.asarray(qpos)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self.shadows)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def reset_batch(env: MazeForageVision, batch: int, seed: int = 0):
    """``jax.vmap``'d reset -> batched ``mjx.Data`` with ``(batch, ...)`` leaves.

    Warp's FFI contract keeps ``contact__*`` / ``nacon`` / ``ncollision``
    *unbatched*; that is expected and is exactly what ``JaxVisionRenderer``
    wants, so do not "fix" it with a blanket ``tree.map(x[None])``.
    """
    rngs = jax.random.split(jax.random.PRNGKey(seed), batch)
    return jax.jit(jax.vmap(env.reset))(rngs)


def tile(images, ncol: int) -> np.ndarray:
    """Row-major grid of equally sized frames."""
    rows = [np.concatenate(images[i:i + ncol], axis=1)
            for i in range(0, len(images), ncol)]
    width = min(r.shape[1] for r in rows)
    return np.concatenate([r[:, :width] for r in rows], axis=0)


def render_ego(env: MazeForageVision, data, batch: int, width: int, height: int,
               grayscale: bool):
    """The Warp pixels, through the same class the training wrapper uses.

    Two traps live here:

    * ``nworld`` is baked into the render context.  Feeding it a differently
      sized batch is *silently accepted* and returns garbage -- build one
      renderer per batch size.
    * ``jax.jit(renderer.render)`` inside a loop retraces every call because a
      bound method is a fresh object each time: 78 ms/call instead of 0.85
      ms/call for a batch of 6 at 32x32.  Hoist the jit (``train_highlvl``
      caches it by ``id(renderer)`` in ``_make_render_all_fn``).
    """
    cfg = env._config
    renderer = JaxVisionRenderer(
        mj_model=env.mj_model,
        mjx_model=env.mjx_model,
        nworld=batch,
        width=width,
        height=height,
        grayscale=grayscale,
        render_depth=False,
        use_textures=bool(cfg.use_textures),
        use_shadows=bool(cfg.use_shadows),
        camera_name=str(cfg.vision_camera_name),
    )
    render = jax.jit(renderer.render)
    return np.asarray(jax.block_until_ready(render(data)))


def to_uint8(images: np.ndarray, scale: int = 1) -> list:
    """``(N, H, W, C)`` float [0, 1] -> list of ``(H*scale, W*scale, 3)`` uint8.

    Matches ``train_highlvl._prepare_ego_overlay``: grayscale is repeated to
    RGB and the upscale is nearest-neighbour, so no pixel is invented.
    """
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    out = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    if scale > 1:
        out = np.repeat(np.repeat(out, scale, axis=1), scale, axis=2)
    return list(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", default="all",
                        choices=["all", "cameras", "overview", "resets", "ego"])
    parser.add_argument("--out", default="maze_forage_figs")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = build_env()

    print(f"maze grid {env.maze_grid.shape}, corridor_width={env.corridor_width:.3f} m, "
          f"half extent {env._maze_half_extent}, {env.free_cell_positions.shape[0]} free cells, "
          f"{env.n_treats} treats, vision {env.vision_shape}")

    if args.what in ("all", "cameras"):
        print("compiled cameras:")
        print(camera_table(env))

    if args.what == "cameras":
        return

    states = reset_batch(env, args.batch, args.seed)
    qpos = np.asarray(states.data.qpos)

    if args.what in ("all", "overview"):
        stage = Stage(env.mj_model, args.width, args.height)
        for tag, cam in [
            ("ortho", overview_camera(env, orthographic=True)),
            ("persp", overview_camera(env, orthographic=False)),
            ("oblique", overview_camera(env, orthographic=False, elevation=-55.0)),
        ]:
            imageio.imwrite(out / f"overview_{tag}.png", stage.frame(qpos[0], cam))
        for name in ("close_profile-rodent", "top-rodent"):
            imageio.imwrite(out / f"camera_{name}.png", stage.frame(qpos[0], name))
        stage.close()
        print(f"wrote overview_* and camera_* to {out}")

    if args.what in ("all", "resets"):
        stage = Stage(env.mj_model, args.width, args.height)
        cam = overview_camera(env, orthographic=True)
        frames = [stage.frame(qpos[i], cam) for i in range(args.batch)]
        stage.close()
        ncol = 3 if args.batch >= 3 else args.batch
        imageio.imwrite(out / "reset_grid.png", tile(frames, ncol))
        root = env._rodent_root_qpos
        print("spawn xy per world:\n", np.round(qpos[:, root:root + 2], 3))
        print(f"wrote reset_grid.png ({args.batch} independent resets) to {out}")

    if args.what in ("all", "ego"):
        cfg = env._config
        native = render_ego(env, states.data, args.batch,
                            int(cfg.vision_width), int(cfg.vision_height),
                            bool(cfg.grayscale))
        print(f"policy vision: {native.shape} in "
              f"[{native.min():.3f}, {native.max():.3f}], mean {native.mean():.3f}, "
              f"{(native < 0.15).mean():.0%} of pixels below 0.15")
        ncol = 3 if args.batch >= 3 else args.batch
        imageio.imwrite(out / "ego_policy_pixels.png",
                        tile(to_uint8(native, scale=8), ncol))
        legible = render_ego(env, states.data, args.batch, 256, 256, False)
        imageio.imwrite(out / "ego_warp_hires.png", tile(to_uint8(legible), ncol))
        print(f"wrote ego_policy_pixels.png (native, 8x nearest) and "
              f"ego_warp_hires.png (256x256 RGB, same ray-tracer) to {out}")


if __name__ == "__main__":
    main()
