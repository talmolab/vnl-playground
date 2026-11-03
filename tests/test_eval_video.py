"""render_video writes a playable MP4 from a fake 3-frame rollout (headless EGL)."""

import os
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import pytest

from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.rodent.eval_video import render_video

pytestmark = pytest.mark.gpu

# Exception types genuinely raised when a headless GL context can't be built
# (see mujoco/egl/__init__.py: ImportError for "no EGL device display", or
# RuntimeError for framebuffer-config/context-creation failures; the compiled
# renderer raises mujoco.FatalError, e.g. via a glad GL-function-pointer load
# failure). Anything else -- a real bug in render_video, a bad model path,
# etc. -- must propagate as a test failure, not vanish into a skip.
_GL_INIT_ERROR_TYPES = (mujoco.FatalError, RuntimeError, ImportError, OSError)
_GL_INIT_KEYWORDS = ("egl", "gl", "glad", "display", "context", "opengl")


def _is_gl_init_failure(exc: Exception) -> bool:
    return isinstance(exc, _GL_INIT_ERROR_TYPES) and any(
        kw in str(exc).lower() for kw in _GL_INIT_KEYWORDS
    )


def _fake_rollout(model, n=3):
    frames = []
    for i in range(n):
        qpos = model.qpos0.copy()
        qpos[0] += 0.01 * i
        frames.append(
            SimpleNamespace(
                data=SimpleNamespace(qpos=qpos, qvel=np.zeros(model.nv)),
                obs={},
                metrics={},
                info={},
                reward=np.float32(0.0),
                done=np.float32(0.0),
            )
        )
    return frames


@pytest.mark.parametrize(
    "cameras", [None, [{"label": "side"}, {"elevation": -90, "label": "top"}]]
)
def test_render_video_writes_mp4(tmp_path, cameras):
    try:
        model = mujoco.MjModel.from_xml_path(str(consts.RODENT_NO_TAIL_COLLISION_XML))
        renderer = mujoco.Renderer(model, height=64, width=96)
    except Exception as e:
        if not _is_gl_init_failure(e):
            raise
        pytest.skip(f"no headless GL: {e}")
    out = tmp_path / "v.mp4"
    render_video(
        _fake_rollout(model),
        model,
        mujoco.MjData(model),
        renderer,
        str(out),
        fps=10,
        cameras=cameras,
    )
    assert out.exists() and out.stat().st_size > 0
    n_panels = 1 if cameras is None else len(cameras)
    frame = imageio.v3.imread(out, index=0)
    assert frame.shape[1] == 96 * n_panels
