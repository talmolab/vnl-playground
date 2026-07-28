# v25's `joystick_base` is ~16.7mm from stac-mjx's calibration-derived value

## TL;DR

`stac-mjx` computes v25's `joystick_base` anchor from a real static-calibration
measurement of the physical joystick rig, re-derived through v24/v25's own
registration rotation (`compute_joystick_geometry_v25.py`), and bakes it into
the STAC-fit XML it actually fits against
(`models/mouse_forelimb_right_janelia_arm_hand_v25_stac.xml`). The reference
clips in `refined_STACed_data_v25/` (and their `xpos`, which vnl-playground
uses directly since `recompute_kinematics=False`) all assume that anchor.

vnl-playground's own copy of the v25 walker XML
(`vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml`)
has a **different** `joystick_base` anchor — a placeholder Eric built on
2026-07-17 before the real calibrated v25 joystick geometry existed (see
`docs/2026-07-17-v24-v25-buildout-and-full-run.md`: "Eric separately built v25
(v24 + a placeholder joystick, **untargeted**) himself"). It was never synced
to the calibration-derived value once `compute_joystick_geometry_v25.py`
landed.

**The two anchors are ~16.7mm apart — nearly 3x the joystick's own ±6mm
`x_slide`/`y_slide` travel range.** The simulated joystick physically cannot
reach the position the reference/reward data says it should be at, relative
to the tracked hand. This is almost certainly why the joystick's physical
placement looks wrong in vnl-playground runs.

## The two values

stac-mjx, `build_v25_stac_xml.py` (from `compute_joystick_geometry_v25.py`,
12-reliable-trial calibration average, v24/v25's own registered frame):

```xml
<body name="joystick_base" pos="-0.01114158 -0.02948650 0.04407920">
```

vnl-playground, `mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml` line 13:

```xml
<body name="joystick_base" pos="0.00029639 -0.02926438 0.05621514">
```

Everything else about the joystick body is byte-identical between the two
files — same local geom offsets (`joystick_geom` at local z=0.0071815,
`joystick_ball` at local z=0.016184), same child-body structure, same
`x_slide`/`y_slide` joints (±6mm each). **The only difference is this one
anchor position.**

```
delta = vnl - stac = (+11.44mm, +0.22mm, +12.14mm)   |delta| ≈ 16.7mm
```

y matches closely (0.2mm); x and z are each off by more than a centimeter.

## Why this matters at runtime

- `default_config_v25()` sets `cfg.recompute_kinematics = False`
  (`imitation_arm_hand.py`, "Per Eric 2026-07-19: STAC-fit xpos is already
  correct; recompute was introducing the error, not fixing one"). That means
  the reference `xpos` used for reward comes straight from stac-mjx's own
  frame — the frame where `joystick_base` sits at `-0.01114158 -0.02948650
  0.04407920`.
- `"joystick"` is in `_TRACKED_BODIES_V25`, so `_bodies_pos_reward()`
  (`imitation.py`) compares the **simulated** joystick body's `xpos` against
  that reference `xpos` every step.
- In vnl-playground's sim, the joystick body can only move within ±6mm of
  its own anchor (`0.00029639 -0.02926438 0.05621514`). It can never close a
  16.7mm gap. The result: a permanently large `body_errors/joystick` term,
  and — more visibly — the joystick sitting over a centimeter away from
  where hand-joystick contact should actually happen relative to the
  (correctly IK-tracked) hand.

## How confident is stac-mjx's number

`compute_joystick_geometry_v25.py` triangulates the same static calibration
recording used for v21/v22
(`static_joystick_labels_withbottom.v001.slp`: `js_ball`/`js_base`/`js_bottom`
markers) and transforms it into v24/v25's frame via the averaged registration
rotation from 12 gauge-reliable trials (`register_v25_mocap.py`). The derived
`shaft_length` (14.363mm) and `ball_offset` (1.822mm) match v21/v22's known
physical-rig constants **exactly**, which is a solid cross-check that the
pipeline (triangulation + registration transform) is self-consistent for v25
too — the same kind of validation that flagged v22's now-fixed 52.6mm
joystick bug (`docs/2026-07-16-v22-joystick-base-placement-bug.md`).

## Suggested fix

Update `joystick_base`'s `pos` in
`vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml`
from:

```
0.00029639 -0.02926438 0.05621514
```

to stac-mjx's calibration-derived value:

```
-0.01114158 -0.02948650 0.04407920
```

If there's a deliberate reason to want a shifted anchor (v22 did this once,
intentionally, and documented why — see the "hand-shifted joystick_base that
trained cleanly" comments in `consts.py`), re-derive that shift explicitly
against v25's real calibrated anchor and document the reasoning, rather than
carrying forward the pre-calibration placeholder value.
