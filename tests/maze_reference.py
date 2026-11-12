"""Reference-env helpers for the maze default-off guards.

An opt-in maze feature must leave every config that does not ask for it
exactly as it was.  The guards compare the current env with the env modules as
they were in the git commit before the feature's config key first appeared.
That commit is found from the history (``git log -S <key>``), so the guards
survive a rebase.  They skip when git or that commit is not available, e.g. in
a shallow clone.

The guards pin the default env to that commit, so a later, intended change to
the default maze or spout env fails them.  Point ``VNL_MAZE_REFERENCE_REV`` at
the commit before such a change to compare against it instead.

The reference modules are the old ``maze_forage_vision.py`` and
``spout_maze.py`` sources, executed as fresh modules.  While the old spout
module is executed, the old maze module stands in for the current one, so the
reference spout env subclasses the reference maze env.  Every other module
(``base``, ``consts``, the spout geometry and trial logic) is the current one.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import subprocess
import sys
import types

import jax
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
MAZE_MODULE = "vnl_playground.tasks.rodent.maze_forage_vision"
MAZE_PATH = "vnl_playground/tasks/rodent/maze_forage_vision.py"
SPOUT_PATH = "vnl_playground/tasks/rodent/spout_maze.py"

# Public ``Data`` fields that differ between two runs of the SAME code on the
# same inputs (warp solver scratch); nothing in the obs, reward or render reads
# them.
NONDETERMINISTIC_LEAVES = (
    "qacc_smooth",
    "moment_colind",
    "moment_rowadr",
    "qLD",
    "qLDiagInv",
    "efc__",
    "solver_niter",
)

# Warp-internal buffers.  ``make_data`` leaves some of them uninitialised
# (island maps, sparse moment storage), so two instances of the same code differ
# there; the trace comparison covers how they are computed.
WARP_INTERNALS = ".data._impl."


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def reference_rev(key: str) -> tuple[str | None, str]:
    """``(rev, reason)``: the commit before ``key`` first appeared in the maze env.

    ``rev`` is None when it cannot be found, and ``reason`` then says why.
    """
    override = os.environ.get("VNL_MAZE_REFERENCE_REV")
    if override:
        return override, ""
    try:
        commits = _git(
            "log", "--format=%H", "--reverse", "-S", key, "--", MAZE_PATH
        ).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"git history is not available: {exc}"
    if not commits:
        return None, f"no commit in the history adds {key!r} to {MAZE_PATH}"
    rev = f"{commits[0]}^"
    try:
        _git("cat-file", "-e", f"{rev}:{MAZE_PATH}")
    except (OSError, subprocess.CalledProcessError):
        return None, f"{MAZE_PATH} does not exist before the commit that adds {key!r}"
    return rev, ""


def _source(rev: str, path: str) -> str:
    return _git("show", f"{rev}:{path}")


@contextlib.contextmanager
def _maze_module_replaced(module: types.ModuleType):
    """Makes ``from vnl_playground.tasks.rodent import maze_forage_vision`` give ``module``."""
    import vnl_playground.tasks.rodent as package

    old_module = sys.modules[MAZE_MODULE]
    old_attr = package.maze_forage_vision
    sys.modules[MAZE_MODULE] = module
    package.maze_forage_vision = module
    try:
        yield
    finally:
        sys.modules[MAZE_MODULE] = old_module
        package.maze_forage_vision = old_attr


def _exec_module(source: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


def load_reference(rev: str) -> types.SimpleNamespace:
    """The maze and spout env modules as of ``rev``, as ``.maze`` and ``.spout``."""
    tag = re.sub(r"\W", "_", rev)
    maze = _exec_module(_source(rev, MAZE_PATH), f"maze_forage_vision_ref_{tag}")
    with _maze_module_replaced(maze):
        spout = _exec_module(_source(rev, SPOUT_PATH), f"spout_maze_ref_{tag}")
    return types.SimpleNamespace(maze=maze, spout=spout)


def model_arrays(model) -> dict:
    """Every public numeric attribute of an ``MjModel``, as numpy arrays."""
    out = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:  # some accessors raise on purpose
            continue
        if isinstance(value, np.ndarray | int | float) and not callable(value):
            out[name] = np.asarray(value)
    return out


def model_mismatches(new_model, ref_model) -> list[str]:
    """Names of the model arrays that differ (or exist on one side only)."""
    new, ref = model_arrays(new_model), model_arrays(ref_model)
    names = sorted(set(new) ^ set(ref))
    names += [
        k
        for k in sorted(set(new) & set(ref))
        if not np.array_equal(new[k], ref[k], equal_nan=True)
    ]
    return names


@contextlib.contextmanager
def zeroed_make_data():
    """Makes ``mjx.make_data`` return all-zero leaves of the same shapes and dtypes.

    Under the warp backend ``make_data`` leaves some island buffers
    uninitialised, so two calls on the same model differ there, and a traced
    ``reset`` bakes those bytes in as constants.  ``make_data``'s values are a
    function of the model and its size arguments only; the model is compared
    array by array and the sizes show up in the leaf shapes.  So two traces
    compare the rest of the computation once the values are zeroed.
    """
    from mujoco import mjx

    original = mjx.make_data

    def make_data(*args, **kwargs):
        # Concrete arrays even under a trace (make_data takes no traced input).
        with jax.ensure_compile_time_eval():
            return jax.tree.map(jax.numpy.zeros_like, original(*args, **kwargs))

    mjx.make_data = make_data
    try:
        yield
    finally:
        mjx.make_data = original


def normalised_jaxpr(fn, *args) -> tuple[str, list[np.ndarray]]:
    """``(text, consts)`` of ``make_jaxpr(fn)`` with per-instance names normalised."""
    closed = jax.make_jaxpr(fn)(*args)
    text = re.sub(r"func_wrapper_\d+", "func_wrapper_N", str(closed))
    text = re.sub(r" at 0x[0-9a-f]+", " at 0xN", text)
    return text, [np.asarray(c) for c in closed.consts]


def jaxpr_mismatch(fn_new, fn_ref, *args) -> str:
    """Empty when both functions trace to the same jaxpr and constants, else why not.

    Traced under `zeroed_make_data`.
    """
    with zeroed_make_data():
        text_new, consts_new = normalised_jaxpr(fn_new, *args)
        text_ref, consts_ref = normalised_jaxpr(fn_ref, *args)
    if text_new != text_ref:
        return "the jaxpr text differs"
    if len(consts_new) != len(consts_ref):
        return f"{len(consts_new)} constants vs {len(consts_ref)}"
    for i, (c_new, c_ref) in enumerate(zip(consts_new, consts_ref)):
        if c_new.dtype != c_ref.dtype or not np.array_equal(
            c_new, c_ref, equal_nan=True
        ):
            return f"constant {i} differs"
    return ""


def shape_tree(tree) -> dict:
    """``{path: (shape, dtype)}`` of every leaf of ``tree``."""
    return {
        jax.tree_util.keystr(path): (tuple(leaf.shape), str(leaf.dtype))
        for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
    }


def state_mismatches(new, ref, exact: bool = True) -> tuple[list[str], list[str]]:
    """``(compared, bad)`` leaf names of two env states.

    A leaf is bad when dtype or shape differ, when it differs bitwise
    (``exact``), or, for float leaves of a non-exact comparison, when it differs
    by more than ``1e-4 + 1e-3 * max|ref leaf|``.  The warp solver is not
    bitwise reproducible from run to run, so only reset states are compared
    exactly; integer and bool leaves are always compared exactly.
    `NONDETERMINISTIC_LEAVES` and `WARP_INTERNALS` are not compared.
    """
    assert jax.tree.structure(new) == jax.tree.structure(ref)
    compared, bad = [], []
    for (path, n), r in zip(
        jax.tree_util.tree_leaves_with_path(new), jax.tree.leaves(ref)
    ):
        name = jax.tree_util.keystr(path)
        if name.startswith(WARP_INTERNALS) or any(
            token in name for token in NONDETERMINISTIC_LEAVES
        ):
            continue
        compared.append(name)
        n, r = np.asarray(n), np.asarray(r)
        if n.dtype != r.dtype or n.shape != r.shape:
            bad.append(name)
        elif exact or not np.issubdtype(n.dtype, np.floating):
            if not np.array_equal(n, r, equal_nan=True):
                bad.append(name)
        elif not np.allclose(
            n,
            r,
            rtol=0.0,
            atol=1e-4 + 1e-3 * float(np.nanmax(np.abs(r), initial=0.0)),
            equal_nan=True,
        ):
            bad.append(name)
    return compared, bad


def rollout(env, key, actions) -> list:
    """``[reset(key), step, step, ...]`` of ``env`` under ``actions``, jitted."""
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    states = [reset(key)]
    for action in actions:
        states.append(step(states[-1], action))
    return states


def run_mismatches(new_env, ref_env, key, actions) -> list[tuple[list, list]]:
    """``[(compared, bad), ...]`` per state of a rollout of both envs.

    The reset state is compared bitwise, the step states to the solver's
    run-to-run noise (`state_mismatches`).
    """
    runs = [rollout(e, key, actions) for e in (new_env, ref_env)]
    return [
        state_mismatches(s_new, s_ref, exact=(i == 0))
        for i, (s_new, s_ref) in enumerate(zip(*runs))
    ]
