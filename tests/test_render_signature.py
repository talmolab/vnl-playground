"""Task ``render`` signatures must match the mujoco_playground wrapper.

``mujoco_playground.Wrapper.render`` forwards its arguments positionally as
``(trajectory, height, width, camera, scene_option, modify_scene_fns)``. A task
whose sixth parameter is anything else silently receives ``None`` there; the
mouse arm lost its reference ghost this way (``render_ghost=None``).
"""

import inspect
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

WRAPPER_PARAMS = [
    "trajectory",
    "height",
    "width",
    "camera",
    "scene_option",
    "modify_scene_fns",
]


def _imitation_envs():
    from vnl_playground import tasks

    return sorted(
        {cls for name, cls in tasks._envs.items() if name.endswith("Imitation")},
        key=lambda cls: cls.__module__,
    )


@pytest.mark.parametrize("env_cls", _imitation_envs(), ids=lambda c: c.__module__)
def test_render_positional_params_match_wrapper(env_cls):
    from mujoco_playground._src import wrapper

    wrapper_params = list(inspect.signature(wrapper.Wrapper.render).parameters)[1:]
    assert wrapper_params == WRAPPER_PARAMS

    params = list(inspect.signature(env_cls.render).parameters)[1:]
    assert params[: len(WRAPPER_PARAMS)] == WRAPPER_PARAMS
