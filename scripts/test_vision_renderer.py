"""Test the VisionRenderer with the RunGap environment.

Renders egocentric camera views from the rodent's skull camera during
a zero-action rollout. Saves a grid of rendered frames.

IMPORTANT: Requires Warp backend (mujoco_impl="warp") and GPU.
"""

import os
import sys

os.environ["MUJOCO_GL"] = "egl"

# ---------------------------------------------------------------------------
# Work around a mujoco_warp BLEEDING_EDGE_MUJOCO detection issue.
#
# When the installed mujoco has a dev-suffixed version string
# (e.g. 3.4.1.dev854365689) but the actual MjModel class does NOT expose
# newer attributes like `flexedge_J_rownnz`, mujoco_warp's `put_model`
# crashes with an AttributeError.  We fix this by pre-loading the standalone
# mujoco_warp and patching the flag before anything calls `put_model`.
# ---------------------------------------------------------------------------
import mujoco

_STANDALONE_MJW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "mujoco_warp",
)
if os.path.isdir(_STANDALONE_MJW_PATH):
    # Evict vendored mujoco_warp so the standalone one takes priority
    sys.modules.pop("mujoco_warp", None)
    if _STANDALONE_MJW_PATH not in sys.path:
        sys.path.insert(0, _STANDALONE_MJW_PATH)

# Now import the standalone mujoco_warp and patch the bleeding-edge flag
import mujoco_warp  # noqa: E402
import mujoco_warp._src.io as _mjw_io  # noqa: E402

if not hasattr(mujoco.MjModel, "flexedge_J_rownnz"):
    _mjw_io.BLEEDING_EDGE_MUJOCO = False
    print(f"Patched BLEEDING_EDGE_MUJOCO=False (mujoco {mujoco.__version__} "
          f"lacks MjModel.flexedge_J_rownnz)")
# ---------------------------------------------------------------------------

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vnl_playground.tasks.rodent import run_gap  # noqa: E402
from vnl_playground.tasks.rodent.vision import VisionRenderer  # noqa: E402


def main():
    print("Initializing RunGap environment (warp backend)...")
    cfg = run_gap.default_config()
    cfg.mujoco_impl = "warp"
    env = run_gap.RunGap(config=cfg)
    mj_model = env.mj_model

    print(f"  Cameras in model: {[mj_model.camera(i).name for i in range(mj_model.ncam)]}")
    print(f"  Action size: {env.action_size}")

    # Create VisionRenderer
    print("\nSetting up VisionRenderer...")
    renderer = VisionRenderer(
        mj_model=mj_model,
        nworld=1,
        camera_name="egocentric-rodent",
        width=64,
        height=64,
    )
    print(f"  Image shape: {renderer.image_shape}")

    # Run zero-action rollout with rendering
    print("\nRunning zero-action rollout with rendering...")
    rng = jax.random.PRNGKey(42)
    state = jax.jit(env.reset)(rng)
    step_fn = jax.jit(env.step)

    frames = []
    render_every = 10
    n_steps = 100

    for step_i in range(n_steps):
        action = jp.zeros(env.action_size)
        state = step_fn(state, action)

        if step_i % render_every == 0:
            # Sync physics state to warp and render
            renderer.sync_state(state.data)
            rgb, _ = renderer.render()
            frames.append(rgb[0])  # World 0
            print(f"  Step {step_i}: rendered frame, shape={rgb[0].shape}, "
                  f"mean_pixel={rgb[0].mean():.1f}")

    # Plot rendered frames in a grid
    n_frames = len(frames)
    cols = min(n_frames, 5)
    rows = (n_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame)
        ax.set_title(f"Step {i * render_every}")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[n_frames:]:
        ax.set_visible(False)

    plt.suptitle("Egocentric camera - Zero action rollout (64x64)")
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "vision_test_egocentric.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
