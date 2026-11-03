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

No module in this repository imports this file -- that is expected. It is
applied by an external training launch harness, which imports it before the
warp backend builds its collision buffers (i.e. before the first
``mjx.make_data``/``put_data`` call of a run). If it is not applied,
``naccdmax`` silently falls back to its tiny heuristic default, and any run
using a large ``naconmax`` either overflows the CCD buffer at runtime or
silently corrupts collision data instead.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import mujoco.mjx.third_party.mujoco_warp as _mjwp

    # ------------------------------------------------------------------ #
    # Guard: verify the patch-target assumption before patching anything.
    # ------------------------------------------------------------------ #
    # The real mjx call path resolves the warp backend through
    # ``mujoco.mjx.warp.mujoco_warp``. This patch only works because, on the
    # pinned stack, that attribute is the *same module object* as
    # ``mujoco.mjx.third_party.mujoco_warp`` (checked below) -- so patching
    # the latter's ``make_data``/``put_data`` also intercepts calls made
    # through the former. If a future mujoco/warp release breaks that
    # identity (e.g. by vendoring a separate copy, or moving/renaming the
    # attribute), this patch would silently stop intercepting the real call
    # path -- a failure that would only surface much later, as an
    # out-of-memory or corrupted-collision bug deep inside training, far
    # from its actual cause. Fail loudly here instead.
    import mujoco.mjx.warp as _mjx_warp_pkg

    _resolved_mujoco_warp = getattr(_mjx_warp_pkg, "mujoco_warp", None)
    if _resolved_mujoco_warp is not _mjwp:
        raise RuntimeError(
            "naccdmax_patch: expected mujoco.mjx.warp.mujoco_warp to be the "
            "same module object as mujoco.mjx.third_party.mujoco_warp "
            f"(target={_mjwp!r}), but found {_resolved_mujoco_warp!r}. The "
            "patch's identity assumption no longer holds on this "
            "mujoco/warp version -- refusing to apply a patch that would "
            "silently fail to intercept the real call path."
        )
    if not (hasattr(_mjwp, "make_data") and hasattr(_mjwp, "put_data")):
        raise RuntimeError(
            "naccdmax_patch: expected mujoco.mjx.third_party.mujoco_warp to "
            "define both `make_data` and `put_data`, but found "
            f"make_data={hasattr(_mjwp, 'make_data')!r}, "
            f"put_data={hasattr(_mjwp, 'put_data')!r}."
        )

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
    print("[naccdmax_patch] mujoco_warp not available; skipping patch")
