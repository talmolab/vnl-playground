"""Monkey-patch for mujoco_warp make_data/put_data to default naccdmax to naconmax.

When using MuJoCo MJX with the warp backend, the top-level ``mjx.make_data()`` API
does NOT forward the ``naccdmax`` parameter to the warp backend's ``make_data()``.
This causes CCD (Convex Collision Detection) buffer overflow when ``naconmax`` is set
to a large value but ``naccdmax`` falls back to a tiny heuristic (~150).

Importing this module patches ``mujoco.mjx.third_party.mujoco_warp.make_data`` and
``put_data`` so that ``naccdmax`` defaults to the value of ``naconmax`` when it is not
explicitly provided.

Usage::

    import vnl_playground.naccdmax_patch  # noqa: F401  -- patch applied on import
"""

from __future__ import annotations

import functools
from typing import Callable

try:
    import mujoco.mjx.third_party.mujoco_warp as _mjwp

    # ------------------------------------------------------------------ #
    # Save references to the original (unpatched) functions.
    # ------------------------------------------------------------------ #
    _original_make_data: Callable = _mjwp.make_data
    _original_put_data: Callable = _mjwp.put_data

    # ------------------------------------------------------------------ #
    # Wrapper helpers
    # ------------------------------------------------------------------ #
    def _inject_naccdmax(kwargs: dict) -> dict:
        """If naccdmax is missing or None, set it to naconmax."""
        if kwargs.get("naccdmax") is None:
            kwargs["naccdmax"] = kwargs.get("naconmax")
        return kwargs

    @functools.wraps(_original_make_data)
    def _patched_make_data(*args, **kwargs):
        kwargs = _inject_naccdmax(kwargs)
        return _original_make_data(*args, **kwargs)

    @functools.wraps(_original_put_data)
    def _patched_put_data(*args, **kwargs):
        kwargs = _inject_naccdmax(kwargs)
        return _original_put_data(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Apply the patch by replacing the module-level attributes.
    # ------------------------------------------------------------------ #
    _mjwp.make_data = _patched_make_data
    _mjwp.put_data = _patched_put_data

    print(
        "[naccdmax_patch] Patched warp make_data/put_data: "
        "naccdmax defaults to naconmax"
    )

except ImportError:
    # Warp backend is not installed -- nothing to patch.
    print(
        "[naccdmax_patch] mujoco_warp not available; skipping patch"
    )
