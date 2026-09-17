"""Enander muscle-spindle afferents as a proprioceptive observation.

Thin adapter onto the `enander` package in the `spindle-models` repo -- the
model itself is NOT forked in here, so there is one implementation and one set
of validation. The `spindle-models` repo must be importable. This repo is run off PYTHONPATH,
so add it the same way:

    PYTHONPATH=/path/to/vnl-playground:/path/to/spindle-models

or drop a `.pth` naming both into the venv's site-packages, which survives
scripts that overwrite PYTHONPATH (the handoff's `_common.sh` does).

Drive is **alpha-coupled**: the policy emits one motor command per muscle and
that same value is `ctrl`, `gamma_static` and `gamma_dynamic`. The action space
is therefore unchanged at 52. Because gamma_static == gamma_dynamic always,
this reduces exactly to Enander's published single-drive Eqs. 6-7 -- the
paper's own all-beta-innervated model.

Velocity normalisation uses the frozen per-muscle table in
`enander/data/v_scale_scott_v2.json`, not `L0 * vmax`. See that file's
`_provenance` and `experiments/imitation/PLAN.md` decision D8: `vmax` is
MuJoCo's force-velocity parameter and clips 23.7% of real samples.
"""

import json
import os
from typing import Tuple

import jax.numpy as jp
import numpy as np

try:
    import enander
    from enander import make_params, spindle_model
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "spindle observations need the `spindle-models` repo importable: add "
        "it to PYTHONPATH, or name it in a .pth in the venv's site-packages"
    ) from exc

# Frozen velocity-normaliser table, shipped with the spindle model.
V_SCALE_TABLE = os.path.join(
    os.path.dirname(os.path.abspath(enander.__file__)),
    "data",
    "v_scale_scott_v2.json",
)

SPINDLE_MODES = ("add", "replace")


def _match_key(name: str) -> str:
    """Normalise an actuator name for table lookup.

    MuJoCo attaches the walker into the arena, which suffixes every actuator
    (e.g. `bicepsbrachii` -> `bicepsbrachii-mouse`). The frozen table is
    generated from the bare walker XML, so strip one trailing `-<token>`.
    """
    return name.rsplit("-", 1)[0] if "-" in name else name


def load_velocity_scale(mj_model, table_path: str = V_SCALE_TABLE) -> np.ndarray:
    """Per-muscle velocity normaliser, ordered to match this model.

    Matches by actuator *name*, not index, so a model whose actuator order
    changed cannot silently pick up the wrong scales -- which matters because
    a wrong scale does not raise, it just produces plausible saturated
    afferents.
    """
    with open(table_path) as handle:
        table = json.load(handle)

    by_name = dict(zip(table["muscle_order"], table["velocity_scale"]))
    by_key = {}
    for raw, value in by_name.items():
        by_key.setdefault(_match_key(raw), value)

    names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    scales, missing = [], []
    for name in names:
        if name in by_name:
            scales.append(by_name[name])
        elif _match_key(name) in by_key:
            scales.append(by_key[_match_key(name)])
        else:
            missing.append(name)
            scales.append(np.nan)

    if missing:
        raise ValueError(
            f"{len(missing)} actuator(s) absent from {table_path}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}. The table was "
            f"frozen for {table['_provenance']['model_xml']}; regenerate it "
            "for this model rather than falling back to L0*vmax silently."
        )
    return np.array(scales)


def build_spindle_params(mj_model, table_path: str = V_SCALE_TABLE):
    """`SpindleParams` for every muscle actuator, built once at env setup."""
    return make_params(
        mj_model, velocity_scale=load_velocity_scale(mj_model, table_path)
    )


def afferents(data, params, drive) -> Tuple[jp.ndarray, jp.ndarray]:
    """Return `(Ia, II)`, each `(nu,)`, from post-step muscle kinematics.

    Args:
        data: `mjx.Data` after the step.
        params: from `build_spindle_params`.
        drive: the policy's motor command, `(nu,)` in [0, 1]. Used for both
            gamma channels (alpha-coupled).

    Uses `data._impl.actuator_*`; the direct `data.actuator_length` spelling is
    deprecated on mujoco 3.4.0 and warns every step.
    """
    impl = getattr(data, "_impl", data)
    ia, ii, _ = spindle_model(
        impl.actuator_length,
        impl.actuator_velocity,
        enander.init_state(),
        params.with_drive(drive, drive),
    )
    return ia, ii


def raw_normalised_velocity(data, params) -> jp.ndarray:
    """Unclipped normalised velocity, for logging the saturation rate.

    The clip hides over-range velocity, so the only way to see whether the
    frozen scale still fits is to log this and watch |V| > 1.
    """
    impl = getattr(data, "_impl", data)
    return impl.actuator_velocity / params.v_scale
