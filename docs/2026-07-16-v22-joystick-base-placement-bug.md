# v22 joystick_base is ~52.6mm from where v21 (and the real rig) puts it

## TL;DR

`stac-mjx`'s v21 model places `joystick_base` from a direct, physical
static-calibration measurement, and its STAC fits show the hand's fingertips
coming within **0.3-1.9mm of the joystick ball** (the ball itself is 2mm
radius, so that's real contact) in every one of the 15 CFL_35/36/37 trials.

v22's `joystick_base` was **not** copied from that measured value. It was
independently re-derived via a wrist-relative-offset transfer, and it lands
**52.6mm away** from v21's position — roughly the length of the whole
forearm. That's almost certainly why the hand never looks close to the
joystick in vnl-playground: the joystick itself is planted in the wrong
place, off the end of the arm's reach.

## How the 0.3-2mm number was reached (stac-mjx side)

Source: `refined_STACed_data_v21/<trial>/<trial>_ik.h5`, produced by
`run_stac_v21_refined.py` (STAC fit, kinematics-only, no dynamics) against
`models/mouse_forelimb_right_janelia_arm_hand_v21.xml`.

Each `_ik.h5` has `marker_sites` (the STAC-fitted site positions, shape
`(n_frames, 19, 3)`, meters) and `kp_data` (the mocap registration target,
same shape), in this fixed keypoint order:

```
Shoulder, Elbow, Wrist, D1_K, D2_K, D3_K, D4_K, D5_K, D2_M, D3_M, D4_M, D5_M,
D1_T, D2_T, D3_T, D4_T, D5_T, js_base, js_ball
```

For every trial: took the 5 fingertip sites (`D1_T..D5_T`), computed Euclidean
distance to `js_ball` at every frame, took the min over fingers (closest
finger that frame), then the min over all frames (closest approach in the
whole clip). Ran on both `marker_sites` (what STAC actually fit) and
`kp_data` (the raw registration target, i.e. the mocap-derived ground truth)
as a cross-check:

| trial | min fit dist (mm) | min mocap dist (mm) |
|---|---|---|
| CFL_35_trial_0001 | 1.48 | 1.70 |
| CFL_35_trial_0101 | 1.77 | 2.30 |
| CFL_35_trial_0201 | 1.86 | 1.68 |
| CFL_35_trial_0301 | 1.72 | 1.57 |
| CFL_35_trial_0401 | 1.14 | 1.55 |
| CFL_36_trial_0001 | 1.58 | 1.37 |
| CFL_36_trial_0101 | 1.08 | 1.75 |
| CFL_36_trial_0201 | 0.71 | 0.96 |
| CFL_36_trial_0301 | 1.51 | 2.08 |
| CFL_36_trial_0401 | 1.61 | 1.96 |
| CFL_37_trial_0001 | 0.35 | 1.02 |
| CFL_37_trial_0101 | 0.43 | 1.15 |
| CFL_37_trial_0201 | 0.83 | 0.78 |
| CFL_37_trial_0301 | 0.46 | 0.68 |
| CFL_37_trial_0401 | 1.01 | 2.01 |

Both the fitted model *and* the underlying mocap agree: the hand reaches
essentially all the way to the ball in every trial. (Visually this is easy to
miss in the reprojection videos — the finger/phalanx meshes are thin and get
visually buried under the marker overlay right at the contact moment — but
the numbers are unambiguous.)

## Where v21 actually puts the joystick (MuJoCo terms)

`models/mouse_forelimb_right_janelia_arm_hand_v21.xml`:

```xml
<body name="joystick_base" pos="0.02754416 0.00213547 0.04439829">
  ...
  <body name="joystick" pos="0 0 0" quat="0.94147265 -0.29173666 -0.16887560 0.00000000">
    <geom name="joystick_geom" .../>   <!-- shaft, length 14.363mm -->
    <geom name="joystick_ball" pos="0 0 0.016184" .../>  <!-- ball center -->
  </body>
</body>
```

This position was **not** guessed or transferred — it comes directly from
triangulating a static calibration recording of the physical joystick rig
(base/ball/bottom markers, `static_joystick_labels_withbottom.v001.slp`),
in the *same* real-world/model frame that the arm keypoints (`shoulder`,
`elbow`, `Wrist`, ...) are registered into (`register_v21_mocap.py`, shared
Procrustes rotation across all 15 trials + per-trial scale correction,
documented in `CALIBRATION_REFINEMENT_REPORT.md`). Since `joystick_base` and
the arm skeleton are registered into the *same* frame by the *same*
pipeline, their relative position is exactly what was physically measured —
there's no independent transfer step that could introduce drift.

The hand keypoints that matter here map to these v21 bodies/site offsets
(`configs/model/mouse_arm_hand_joystick_v21.yaml`):

```
D1_T -> Phalanx_hand_2_1_right, site offset (0, 0.00066, 0)
D2_T -> Phalanx_hand_3_2_right, site offset (0, 0.00127, 0)
D3_T -> Phalanx_hand_3_3_right, site offset (0, 0.00146, 0)
D4_T -> Phalanx_hand_3_4_right, site offset (0, 0.00141, 0)
D5_T -> Phalanx_hand_3_5_right, site offset (0, 0.00098, 0)
js_base -> joystick, site offset (0, 0, 0.014363)   # shaft/ball junction
js_ball -> joystick, site offset (0, 0, 0.016184)   # ball center
```

## What v22 did differently, and why it's the likely bug

`janelia_model/v22/mouse_forelimb_right_janelia_arm_hand_v22.xml`, header
comment at the `joystick_base` body:

```xml
<body name="joystick_base" pos="-0.01685188 -0.02594152 0.04776544">
```

vs. v21's `pos="0.02754416 0.00213547 0.04439829"`.

**Difference: dx=-44.4mm, dy=-28.1mm, dz=+3.4mm → 52.6mm total.** That's not
noise — it's on the order of the forearm's own length.

The XML comment documents the method: rather than copying v21's measured
`joystick_base` position directly (which can't be done naively because v22's
*rest pose* differs from v21's — different neutral joint configuration), the
value was derived by:

1. At each of 240 STAC-fitted frames (all 3 animals), compute the joystick's
   position as a local offset from the `wrist` body, using **that frame's
   fitted wrist orientation** (in v21's model/frame).
2. Re-apply that same local offset using **v22's own wrist xpos/xquat** at
   the same qpos values (joint angles carry over by name; `z_slide` was
   dropped since v22 has no equivalent DOF).
3. Take the resulting joystick position as v22's new `joystick_base` anchor
   (x_slide=0/y_slide=0 rest position).

The comment reports this was stable across all 240 frames (~1-3mm per-animal
std, ~1.5mm cross-animal agreement) as evidence the transfer is sound. **That
stability check does not rule out a systematic bug** — a consistent rotation-
convention error (e.g. local-vs-world frame mixed up, or a transpose/inverse
applied on the wrong side when re-expressing the offset in v22's wrist frame)
would reproduce a stable-but-wrong answer every single frame, since the same
bug fires identically each time.

Given the size of the discrepancy (~52.6mm, roughly one forearm-length) and
that v21's own registered data has the joystick within ~2mm of the fingers,
a rotation-frame mismatch in step 2 (re-expressing the wrist-relative offset
using v22's wrist orientation instead of v21's, or a sign/transpose error in
that rotation) is the most likely explanation. Also worth checking:
v22 added a new `radius` body between `ulna` and `wrist` to fix forearm
pronation/supination (see v22.xml's own header, "v22 fix, 2026-07-15") —
if that changed the `wrist` body's local orientation convention (not just its
position), any offset transfer keyed on wrist orientation would silently
break even though wrist's *position* still matches v21's to within a few mm.

## Suggested fix

The simplest, lowest-risk fix: **don't transfer via wrist-relative offset at
all.** Since v21's arm skeleton and joystick are registered into the same
real-world frame by the same static-calibration + shared-rotation pipeline,
and v22's rest pose differs from v21's only in joint *configuration* (not in
the fixed, non-joint anchor points like `clavicle`/`shoulder_base`), it should
be possible to re-derive v22's `joystick_base` position directly against
v22's own rest-pose FK relative to `shoulder_base`/`clavicle`, using the same
static-calibration numbers v21 used (`joystick_base pos="0.02754416
0.00213547 0.04439829"`), rather than transferring through a per-frame wrist
offset at all. If v22's rest pose truly requires a different anchor (e.g.
because `shoulder_base` itself moved), the fix is to redo the transfer but
sanity-check it against the *known-good* v21 answer: pick one frame, compute
the wrist-relative offset in v21, re-apply it in v21 itself (a no-op round
trip) and confirm you get back `joystick_base`'s known v21 position, before
trusting the same code path applied to v22.
