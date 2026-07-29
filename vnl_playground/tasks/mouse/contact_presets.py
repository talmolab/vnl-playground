"""Contact-parameter presets for the hand<->joystick models.

Ported verbatim in behaviour from the kinematic contact probe
(`analysis/2026-07-28-hand-joystick-contact-probe/scripts/probe_hand_joystick.py`),
which measured what these settings actually deliver with the hand held rigid:

| preset                  | press transmission | push while holding still | ringing |
|-------------------------|--------------------|--------------------------|---------|
| shipped                 | 0.024              | 1.1 mN                   | --      |
| harder, k=30x, dt 0.25ms| 0.920              | 62 mN                    | 10 mN   |
| harder, k=3000x, dt 10us| 0.964              | 1036 mN                  | 687 mN  |

The joystick's return spring needs ~54 mN, so k=30x is the setting where the
contact does the work the physics asks for. k=3000x delivers 19x that,
oscillating at +-687 mN, and halving the timestep does not fix it (687 -> 674),
so it is genuine stiff-contact ringing rather than discretisation.

Nothing here is applied unless `cfg.contact_preset` is set, so every existing
config (v22/v22x/v23/v25) keeps its shipped contact behaviour untouched.
"""

import mujoco
import numpy as np

# contype bitmasks the walker XMLs use to pair hand against joystick. The hand's
# collision proxies carry contype=4/conaffinity=8 and the joystick's three geoms
# the mirror image, so these two values select exactly the geoms that can
# participate in a grip contact -- and nothing in the arena.
GRIP_CONTYPE = 4
JOYSTICK_CONTYPE = 8

# The joystick's own translational dofs. Their limits inherit the *global*
# solref because the XML never sets jnt_solref, so once the contact is stiffened
# they become the weakest link in the chain and the stick gets squeezed straight
# through its own +-6 mm range.
_JOYSTICK_SLIDE_JOINTS = ("x_slide", "y_slide")


def shipped_contact_k(model, geom_id):
    """The contact stiffness k (1/s^2) `geom_id` currently delivers.

    Read off the model rather than tabulated, so it stays correct for any XML.
    Mirrors MuJoCo's `constraint.py::_kbi` including the REFSAFE clamp -- which
    is the whole reason solref's timeconst looks like a dead knob here: it is
    floored at 2*dt, and v25 and scott_v1 both ship 0.002 s at dt = 1 ms, i.e.
    sitting exactly on the floor.
    """
    timeconst = max(float(model.geom_solref[geom_id][0]),
                    2.0 * float(model.opt.timestep))
    dampratio = float(model.geom_solref[geom_id][1])
    dmax = float(model.geom_solimp[geom_id][1])
    return 1.0 / (dmax**2 * timeconst**2 * dampratio**2)


def direct_solref(k, dmax, dampratio=1.0):
    """A solref that sets (k, b) directly, bypassing timeconst and its clamp.

    `solref[0] <= 0` switches MuJoCo to `k = -solref[0]/dmax^2` and
    `b = -solref[1]/dmax`. b is chosen to match what the positive branch would
    give at this dampratio, so only the stiffness changes.
    """
    return np.array([-k * dmax * dmax, -2.0 * dampratio * np.sqrt(k) * dmax])


def _find_joint(model, base_name):
    """Joint id for `base_name`, tolerating the arena-attach name suffix.

    MouseBaseEnv attaches the walker into the arena, which renames every
    element to `<name><suffix>` (e.g. "x_slide-mouse"). This is called both on
    a bare walker XML (tests, the contact probe) and on the attached model
    (compile()), so match either form rather than requiring the caller to know
    which one it holds.
    """
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_name)
    if j >= 0:
        return j
    matches = [
        i
        for i in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).startswith(base_name)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return -1
    raise ValueError(
        f"joint name {base_name!r} is ambiguous in this model: {matches}"
    )


def apply_contact_preset(model, preset, stiffness_mult=30.0, dmax=0.999):
    """Apply a contact preset in place; returns the same model.

    'shipped' -- no-op.

    'hard'    -- `gap=0` and impedance dmax on every grip and joystick geom.
                 Deliberately leaves solref alone: at dt = 1 ms the timeconst
                 clamp is already active, so impedance is the only free
                 stiffness left. Costs nothing in timestep.

    'harder'  -- 'hard' plus a direct (negative) solref at `stiffness_mult` x
                 the shipped k, which is the only way past the timeconst clamp,
                 applied to the grip/joystick geoms *and* to the joystick's own
                 slide limits.

    Must be called after `model.opt.timestep` is final: `shipped_contact_k`
    reads the timestep through the REFSAFE clamp.
    """
    if preset in (None, "shipped"):
        return model
    if preset not in ("hard", "harder"):
        raise ValueError(
            f"unknown contact_preset {preset!r}; expected one of "
            "None/'shipped'/'hard'/'harder'"
        )

    grip_or_joystick = np.flatnonzero(
        (model.geom_contype == GRIP_CONTYPE)
        | (model.geom_contype == JOYSTICK_CONTYPE)
    )
    if grip_or_joystick.size == 0:
        raise ValueError(
            "contact_preset was set but no geom carries contype "
            f"{GRIP_CONTYPE} or {JOYSTICK_CONTYPE} -- this model has no "
            "hand/joystick contact pair to harden."
        )

    # Read the shipped stiffness BEFORE the loop below overwrites solimp dmax,
    # which shipped_contact_k depends on.
    solref = None
    if preset == "harder":
        k = stiffness_mult * shipped_contact_k(model, int(grip_or_joystick[0]))
        solref = direct_solref(k, dmax)

    for g in grip_or_joystick:
        model.geom_gap[g] = 0.0
        model.geom_solimp[g][0] = dmax
        model.geom_solimp[g][1] = dmax
        if solref is not None:
            model.geom_solref[g] = solref

    if solref is not None:
        for name in _JOYSTICK_SLIDE_JOINTS:
            j = _find_joint(model, name)
            if j < 0:
                raise ValueError(
                    f"contact_preset='harder' needs joint {name!r} to harden "
                    "the joystick's own slide limits, but it is not in this "
                    "model."
                )
            model.jnt_solref[j] = solref
            model.jnt_solimp[j][0] = dmax
            model.jnt_solimp[j][1] = dmax
    return model
