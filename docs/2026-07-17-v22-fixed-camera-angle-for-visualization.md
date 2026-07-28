# Fixed camera angle for v22 arm+joystick visualization (shaft rendered vertical)

## TL;DR

For any diagnostic/visualization rendering of the v22 arm+hand+joystick model
in MuJoCo (not the calibration-matched reprojection-onto-video renders,
which use their own per-trial camera derived from each trial's real
calibration — see `stac_mjx/reproject_stac_model.py`), use the fixed camera
below. It's rolled so the joystick shaft renders **vertical in the image**,
regardless of the shaft's actual ~38.6° tilt in the model's own coordinate
frame (see `2026-07-16-v22-registration-fix-and-joystick-anchor-update.md`
for why that tilt exists and why it's currently unresolved/disputed — this
doc is unrelated to that argument, it's purely about picking a legible,
reproducible viewing angle for any future renders, independent of how that
tilt question gets resolved). It also frames both the shoulder/arm and the
joystick in the same shot.

## Camera parameters

As a MuJoCo `<camera>` element (add to `<worldbody>`, or construct via
`mujoco.MjvCamera`/xyaxes at render time):

```xml
<camera name="arm_joystick_fixed_view"
        pos="-0.03940696 -0.04889572 0.09061815"
        xyaxes="0.65772284 -0.75326002 0.00000000  0.46960983 0.41004846 0.78187395"
        fovy="45"/>
```

- `pos`: camera position in world/model frame (meters).
- `xyaxes`: first 3 numbers = camera "right" axis, last 3 = camera "up" axis
  (MuJoCo xyaxes convention — forward/view direction is `-cross(right, up)`).
- `fovy`: 45 degrees, arbitrary/adjustable to taste — not load-bearing for
  the "shaft vertical" property, only affects zoom.

This was derived and verified against
`stac-mjx/refined_STACed_data_v22/CFL_35_20240128_trial_0001/CFL_35_20240128_trial_0001_ik.h5`,
rendered directly against `models/mouse_forelimb_right_janelia_arm_hand_v22_stac.xml`
(no video compositing — this is a clean, from-scratch MuJoCo render, so it
should transfer directly to any v22 model file with the same `joystick`
body orientation and same `humerus`/`joystick_base` body positions; if
either of those change, e.g. from re-deriving `joystick_base` upright per
the open item in the registration-fix doc, this camera should be
re-derived — see "How to reproduce/re-derive" below).

## How it was derived

1. **`up` axis = the joystick shaft's actual world-frame direction.** Read
   the `joystick` body's world orientation (`xquat`) via `mj_forward` at
   rest (qpos=0), and rotate its local `+Z` (the shaft's own long axis) into
   world frame: `shaft_dir = R(xquat) @ [0,0,1]`, normalized. Setting the
   camera's `up` to this vector is what makes the shaft render vertical —
   it doesn't matter whether that tilt is "real" or an artifact; the camera
   simply rolls to match whatever the model currently does.
2. **`right` axis**: `cross(shaft_dir, world_ref)`, normalized, where
   `world_ref = [0,0,1]` (or `[1,0,0]` as a fallback if `shaft_dir` is
   nearly parallel to `[0,0,1]`, to avoid a degenerate cross product — not
   triggered for this model's actual tilt, but kept for robustness if the
   tilt changes).
3. **`forward`** (the direction the camera looks into the scene):
   `-cross(right, up)`, normalized (MuJoCo xyaxes convention).
4. **`lookat`**: midpoint between the `humerus` body's world position
   (`[-0.0071983, -0.01418874, 0.07143588]` — this is the same point
   `register_v22_mocap.py` calls `targets["shoulder"]`/`model_shoulder`,
   just read off the `humerus` body directly here since it's the visible
   mesh body nearest that anchor) and the `joystick_base` body's world
   position (`[-0.01264354, -0.03211015, 0.04737563]`).
5. **`distance`**: `span * 1.15 + 0.015` meters, where `span` is the
   distance between those same two points (`~0.0305m` for this model) —
   tuned empirically so both the shoulder/arm mesh and the joystick fill
   most of the frame without clipping either. `pos = lookat - forward * distance`.

## How to reproduce/re-derive

If the model changes (new `joystick_base` position/orientation, e.g. from
the pending upright re-derivation), re-run the same steps in a MuJoCo
Python session — `mj_forward` at qpos=0 to get body world positions/orientations,
then steps 1–5 above. This is a ~30-line script; ask for it again if useful
rather than hand-deriving new numbers, since the reasoning depends on
whichever XML you're pointing at matching the actual body names used above
(`joystick`, `joystick_base`, `humerus`).
