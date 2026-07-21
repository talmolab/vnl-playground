"""Thermal-gradient (scalar temperature field) factory for the thermotaxis task.

A temperature field maps a planar position ``(x, y)`` to a scalar temperature. The
field must be evaluated every simulation step on *traced* positions and may differ
per vmapped parallel environment, so a field cannot be stored as a Python object in
``mjx_env.State.info`` (info must be a pytree of arrays). Instead:

- :meth:`Gradient.make_gradient` (called once at ``reset``) resolves a gradient
  *type* and its parameters into a fixed-length array ``params`` plus an integer
  ``shape_id``. Both are stored in ``info``.
- :meth:`Gradient.evaluate` (called every step) is a pure function that
  ``jax.lax.switch`` es on ``shape_id`` to the right per-type evaluator.

Convention: any argument passed as a length-2 iterable of scalars is treated as
``[low, high]`` bounds for a uniform sample; a plain scalar is used as-is. This is
applied at the *leaf* level, so coordinate arguments like ``setpoint_loc`` are
``(x, y)`` pairs whose elements each follow the rule
(``setpoint_loc=(5.0, 0.0)`` -> fixed point; ``setpoint_loc=([3, 7], 0.0)`` ->
``x ~ U(3, 7)``, ``y = 0``).
"""

from typing import Any, List, Sequence, Tuple

import jax
import jax.numpy as jp

# Width of the (padded) parameter vector stored in ``info``. Must be >= the number
# of parameters any single gradient type packs.
_PARAM_WIDTH = 6


def _is_scalar(x: Any) -> bool:
    """True if ``x`` is a plain Python number (not an iterable/array)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_bounds(arg: Any) -> bool:
    """True if ``arg`` is a length-2 iterable of scalars, i.e. ``[low, high]``."""
    return (
        isinstance(arg, (list, tuple))
        and len(arg) == 2
        and all(_is_scalar(x) for x in arg)
    )


def _resolve(arg: Any, key: jax.Array) -> jp.ndarray:
    """Resolve a leaf argument to a scalar array.

    A length-2 iterable of scalars is uniform-sampled in ``[arg[0], arg[1]]`` using
    ``key``; any other value is returned as a fixed scalar array. This is a
    trace-time Python branch on the (static) config value, so ``key`` is only
    consumed for the random case.
    """
    if _is_bounds(arg):
        lo, hi = float(arg[0]), float(arg[1])
        return jax.random.uniform(key, (), minval=lo, maxval=hi)
    return jp.asarray(arg, dtype=jp.float32)


def _pack(vals: Sequence[Any]) -> jp.ndarray:
    """Pack a list of scalars into a fixed-length ``(_PARAM_WIDTH,)`` param vector."""
    stacked = jp.stack([jp.asarray(v, dtype=jp.float32) for v in vals])
    out = jp.zeros(_PARAM_WIDTH, dtype=jp.float32)
    return out.at[: stacked.shape[0]].set(stacked)


class Gradient:
    """Factory + evaluator for thermal gradient fields.

    ``TYPES`` is the ordered registry of implemented field shapes; a field's index
    into this tuple is its ``shape_id``. Add a new shape by implementing a
    ``_build_<name>`` / ``_eval_<name>`` pair and appending ``<name>`` to ``TYPES``
    (and to the branch lists in :meth:`_build` / :meth:`evaluate`).
    """

    TYPES: Tuple[str, ...] = ("linear", "gaussian")

    # ------------------------------------------------------------------ factory
    @classmethod
    def make_gradient(
        cls,
        rng: jax.Array,
        *,
        gradient_type: Any = "linear",
        setpoint: Any,
        setpoint_loc: Sequence[Any],
        min_temp: Any,
        max_temp: Any,
        arena_size: Sequence[Any],
        **type_kwargs: Any,
    ) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """Resolve a gradient type + parameters for one episode.

        Args:
            rng: Random key (consumed for iterable-bound sampling and, when the type
                set has >1 candidate, for random type selection).
            gradient_type: ``str`` in :attr:`TYPES`, a ``list``/``tuple`` of such
                strings to choose from at random, ``"random"`` to choose from all
                implemented types, or ``None`` -> ``"linear"``.
            setpoint: Target temperature *at* ``setpoint_loc`` (scalar or bounds).
            setpoint_loc: ``(x, y)`` anchor location (each element scalar or bounds).
            min_temp, max_temp: Temperature range controlling gradient steepness
                (scalar or bounds).
            arena_size: ``(Lx, Ly)`` arena half-extents (each element scalar or bounds).
            **type_kwargs: Extra per-type params, e.g. ``sigma`` for gaussian.

        Returns:
            ``(shape_id, params, setpoint_value)`` where ``shape_id`` is an int32
            scalar indexing :attr:`TYPES`, ``params`` is a ``(_PARAM_WIDTH,)`` float
            array, and ``setpoint_value`` is the resolved target temperature.
        """
        candidates = cls._candidate_types(gradient_type)

        keys = jax.random.split(rng, 9)
        setpoint_v = _resolve(setpoint, keys[0])
        lx = _resolve(setpoint_loc[0], keys[1])
        ly = _resolve(setpoint_loc[1], keys[2])
        min_t = _resolve(min_temp, keys[3])
        max_t = _resolve(max_temp, keys[4])
        arena_lx = _resolve(arena_size[0], keys[5])
        arena_ly = _resolve(arena_size[1], keys[6])
        sigma = _resolve(type_kwargs.get("sigma", 3.0), keys[7])

        build_args = dict(
            setpoint=setpoint_v,
            lx=lx,
            ly=ly,
            min_temp=min_t,
            max_temp=max_t,
            Lx=arena_lx,
            Ly=arena_ly,
            sigma=sigma,
        )

        cand_ids = [cls.TYPES.index(t) for t in candidates]
        if len(cand_ids) == 1:
            shape_id = jp.asarray(cand_ids[0], dtype=jp.int32)
        else:
            local_idx = jax.random.randint(keys[8], (), 0, len(cand_ids))
            shape_id = jp.asarray(cand_ids, dtype=jp.int32)[local_idx]

        params = cls._build(shape_id, build_args)
        return shape_id, params, setpoint_v

    @classmethod
    def _candidate_types(cls, gradient_type: Any) -> List[str]:
        """Resolve ``gradient_type`` (str/list/"random"/None) to a candidate list."""
        if gradient_type is None:
            return ["linear"]
        if gradient_type == "random":
            return list(cls.TYPES)
        if isinstance(gradient_type, (list, tuple)):
            candidates = list(gradient_type)
        else:
            candidates = [gradient_type]
        for t in candidates:
            if t not in cls.TYPES:
                raise ValueError(
                    f"Unknown gradient_type '{t}'. Available: {cls.TYPES}"
                )
        return candidates

    @classmethod
    def _build(cls, shape_id: jp.ndarray, build_args: dict) -> jp.ndarray:
        """Build the param vector for ``shape_id`` via a switch over all builders."""
        builders = [cls._build_linear, cls._build_gaussian]
        return jax.lax.switch(
            shape_id,
            [(lambda ba, fn=fn: fn(**ba)) for fn in builders],
            build_args,
        )

    @classmethod
    def evaluate(
        cls, shape_id: jp.ndarray, params: jp.ndarray, xy: jp.ndarray
    ) -> jp.ndarray:
        """Evaluate temperature ``T(xy)`` for the field ``(shape_id, params)``."""
        return jax.lax.switch(
            shape_id, [cls._eval_linear, cls._eval_gaussian], params, xy
        )

    # --------------------------------------------------------------- linear field
    @staticmethod
    def _build_linear(setpoint, lx, ly, min_temp, max_temp, Lx, Ly, sigma):
        """Pack params for a planar gradient anchored at ``setpoint`` on ``(lx, ly)``.

        v1 gradient points along +x with slope ``(max - min) / (2*Lx)``.
        """
        del Ly, sigma
        gx = (max_temp - min_temp) / (2.0 * Lx)
        gy = jp.asarray(0.0, dtype=jp.float32)
        return _pack([setpoint, lx, ly, gx, gy])

    @staticmethod
    def _eval_linear(params, xy):
        setpoint, lx, ly, gx, gy = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
        return setpoint + gx * (xy[0] - lx) + gy * (xy[1] - ly)

    # ------------------------------------------------------------- gaussian field
    @staticmethod
    def _build_gaussian(setpoint, lx, ly, min_temp, max_temp, Lx, Ly, sigma):
        """Pack params for a radial peak of value ``setpoint`` at ``(lx, ly)``.

        Decays to ``min_temp`` far away with width ``sigma``. ``max_temp`` unused.
        """
        del max_temp, Lx, Ly
        return _pack([setpoint, lx, ly, min_temp, sigma])

    @staticmethod
    def _eval_gaussian(params, xy):
        setpoint, lx, ly, min_temp, sigma = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
        r2 = (xy[0] - lx) ** 2 + (xy[1] - ly) ** 2
        return min_temp + (setpoint - min_temp) * jp.exp(-r2 / (2.0 * sigma**2))
