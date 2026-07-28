# v22 joystick geometry: what vnl-playground changed, and what stac-mjx should fix upstream

## TL;DR

stac-mjx's `mouse_forelimb_right_janelia_arm_hand_v22_stac.xml` gives the
joystick a shaft tilt (`quat="0.94389458 -0.21721094 0.24876180 0"`), derived
from a static calibration recording. **That tilt is wrong** — the real rig's
mounting bracket sits flush to the x,y plane and the shaft rises straight up
along Z, confirmed directly against the physical setup. Removing the tilt
(keeping stac-mjx's registration-bug-fixed `joystick_base` position) leaves
the hand short of the joystick by ~7-8mm at closest approach — not real
contact. vnl-playground worked around this locally by translating (not
rotating) `joystick_base`'s position so the hand and ball actually touch,
but this was derived from **one trial only** and should be redone properly
upstream, in stac-mjx, against the full 15-trial set, with the joystick
constrained to stay upright rather than fit as a free rotation.

## What was wrong

`mouse_forelimb_right_janelia_arm_hand_v22_stac.xml`'s `joystick` body:

```xml
<body name="joystick_base" pos="-0.01264354 -0.03211015 0.04737563">
  <body name="joystick" pos="0 0 0" quat="0.94389458 -0.21721094 0.24876180 0.00000000">
    <geom name="joystick_geom" type="cylinder" size="0.001 0.0071815" pos="0 0 0.0071815" .../>
    <geom name="joystick_ball" type="sphere" size="0.002" pos="0 0 0.016184" .../>
  </body>
</body>
```

The `quat` on the inner `joystick` body tilts the shaft+ball relative to
`joystick_base`'s mounting plate (`circular_base`, a sibling geom on
`joystick_base` itself, unrotated). Two problems with this, found when
rendering it in vnl-playground:

1. **Structurally implausible even on its own terms.** Because the tilt is
   applied only to the shaft+ball body and not to the mounting plate, the
   rendered shaft comes out of its own base at a non-perpendicular angle —
   not how a rigid, single-piece mounted joystick would look regardless of
   what angle the whole assembly sits at.
2. **Doesn't match the physical rig.** Per direct confirmation (Eric): the
   mounting bracket is flush to the x,y plane and the shaft rises straight
   up along Z. No tilt, on the base or the shaft.

We tested applying the same tilt to `joystick_base` instead (rotating base
+ shaft together, so they stay mutually perpendicular) — this fixed problem
(1) and gave an equally good numeric fit (fingertip-to-ball distance ~1.9mm
at closest approach, essentially unchanged from the original tilted
derivation, since the intermediate body has zero translation offset —
rotating at either level produces the same result). But the whole assembly
was still visibly non-vertical, which doesn't match the physical rig, so we
rejected this too.

## What we actually want: upright, always

No rotation anywhere in the joystick assembly — `joystick_base` and
`joystick` both at identity orientation. Just `joystick_base`'s XYZ
position as the free parameter.

**Problem: with stac-mjx's own registration-bug-fixed position
(`-0.01264354 -0.03211015 0.04737563`) and no tilt, the hand doesn't reach
the joystick.** Checked against `CFL_35_20240128_trial_0001`'s real STAC-fit
qpos (native v22 STAC data, not the earlier v21-transplant approach):
minimum fingertip-to-ball distance across the full 126-frame trial was
**7.89mm** (ball radius is 2mm, so nowhere near actual contact). This is
after stac-mjx's registration-rotation-sharing bugfix (v5) was already
applied — so either that fix isn't fully sufficient, or the joystick
position derivation itself (median of 12 trials' `js_base` position,
combined with static-calibration shaft geometry) has more error left in it
than the tilted derivation's apparent ~1-5mm accuracy suggested, once you
take the tilt back out.

## What vnl-playground did as a stopgap (NOT a proper fix, please redo upstream)

Kept the joystick upright (no quat anywhere) and translated `joystick_base`
to make contact happen, using ONE trial as ground truth:

1. Loaded `CFL_35_20240128_trial_0001`'s real STAC-fit qpos.
2. With `x_slide`/`y_slide` zeroed (to probe the rest anchor, not a
   particular frame's deflection), found the frame/fingertip where the
   reach naturally comes closest to `joystick_base`'s rest position:
   frame 107, `Phalanx_hand_3_4_right`, 16.27mm away.
3. Shifted `joystick_base` so the ball (which sits at local
   `(0, 0, 0.016184)` above `joystick_base` when upright) would land at
   that fingertip's position.
4. Re-checked across the full trial with real (non-zeroed) `x_slide`/
   `y_slide` values: closest approach was now 3.21mm at frame 102 — better,
   but Eric wanted actual contact, not just "close."
5. Computed the residual delta at that specific frame (102) between the
   ball and the closest fingertip (`Phalanx_hand_3_5_right`), and shifted
   `joystick_base` by that exact residual.

**Final position**: `pos="-0.00447064 -0.02640225 0.04534320"` — a
translation of **(+8.17mm, +5.71mm, -2.03mm)** from stac-mjx's
registration-bugfix-derived value. No rotation anywhere.

**Verified**: minimum fingertip-to-ball distance = **0.00mm** at frame 102,
max 3.02mm across the full 126-frame trial. Rendered and visually confirmed
— fingers genuinely wrapped around and touching the ball.

## Why this is NOT good enough as a permanent fix, and what stac-mjx should do

- **Derived from one trial only** (`CFL_35_20240128_trial_0001`). Not
  checked against the other 4 available CFL_35 trials, let alone the
  pending CFL_36/37 trials (10 more). It may not generalize — different
  trials/animals could need a different correction if there's any
  per-animal or per-session registration drift.
- **It's a translation hack fit to make one trial's numbers work**, not a
  principled re-derivation. The right fix is upstream: re-derive
  `joystick_base`'s position properly in stac-mjx, **with the joystick
  constrained to stay upright** (no rotation as a free parameter — that's
  now confirmed physically wrong), across the full trial set, and validate
  that the resulting position gives real hand-joystick contact (not just a
  "stable, low-scatter" fit — stability doesn't mean *correct*, as this
  whole saga demonstrated).
- Possible places to look for the remaining error, now that rotation is
  off the table as an explanation: whether the "12 gauge-reliable trials"
  used for the position median are actually representative; whether the
  static-calibration-recording's own frame (used for shaft geometry) still
  has a small residual offset from the per-trial `calibration_refined.toml`
  frame (a distinct, smaller issue already flagged in the prior
  registration-bug writeup); whether `js_base`/`js_ball` marker placement
  on the physical rig during calibration matches where the model's
  `joystick_base`/`joystick_ball` sites are actually defined.

## For reference: files on the vnl-playground side

- Patched model (the fix described above, applied):
  `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_arm_hand_v22_contacts_patched.xml`
- Full narrative/derivation log:
  `docs/superpowers/plans/2026-07-16-janelia-v22-arm-hand-stac-v21-training.md`
  (see the last three sections, all dated 2026-07-16)
