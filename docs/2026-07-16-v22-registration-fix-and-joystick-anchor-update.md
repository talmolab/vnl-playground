# v22 registration bug fixed on the stac-mjx side; vnl-playground's training model needs reconciling

## TL;DR

This follows on from `2026-07-16-v22-joystick-base-placement-bug.md` (same
day, earlier finding). That doc diagnosed a 52.6mm `joystick_base` placement
error and recommended re-deriving it from v22's own static calibration. That
re-derivation was done, but the joystick still looked crooked/offset/
inconsistently-sized trial-to-trial in reprojection videos. The **actual**
root cause turned out to be one level upstream: `register_v22_mocap.py` (in
`stac-mjx`) was forcing every one of the 15 CFL_35/36/37 trials — spanning
three recording sessions months apart — to share ONE rotation matrix when
converting raw triangulated mocap into model-frame coordinates. That's now
fixed (details below), registration has been re-run for all 15 trials, and
`joystick_base`'s position has been re-derived again against the corrected
data.

**None of this touches vnl-playground directly** — it's all upstream, in
`stac-mjx`. But vnl-playground's actual training model
(`vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_arm_hand_v22_contacts_patched.xml`)
currently has its own hand-patched `joystick_base` position and geom sizes,
independent of stac-mjx's model file, and they now disagree. Someone (or a
Claude session in this repo) needs to decide how to reconcile them. See
"Action items" at the bottom.

## What was actually wrong (stac-mjx side)

`register_v22_mocap.py`'s previous recipe (labeled `recipe =
"register_v22_mocap_shared_rotation_v4"` in its output h5 attrs) averaged
every trial's own independently-fit Procrustes rotation (calibration frame →
model frame) into one shared rotation, then used that same shared rotation
for every trial's registration. The stated justification (in the old
docstring, now corrected) was: "the joystick and cameras are physically
identical across all 15 trials (confirmed byte-identical calibration.toml)."

That premise is false. Directly checked:

- `calibration.toml` (the original, non-refined calibration) *is* byte-identical
  within a session (all 5 CFL_35 trials share one file).
- `calibration_refined.toml` (the per-trial bundle-adjustment-refined
  calibration, which is what's actually used for triangulating mocap and for
  reprojection rendering) is a **different file, with measurably different
  camera extrinsics, for every single trial** — even within one same-day
  session. Directly measured rotation differences of 8–21° between trials'
  `Cam00` extrinsics, just within CFL_35's 5 trials.

Forcing a shared rotation onto trials whose true calibration-refined frame
diverges from it by that much doesn't average out noise — it bakes in a
real, trial-dependent registration error. Quantified directly: comparing the
joystick target position each trial's *own* rotation would produce vs. what
the shared rotation produced, for the CFL_35 session:

| trial | target shift, shared vs. own rotation (mm) | visual report |
|---|---|---|
| trial_0001 | 3.53 | "crooked, too large" |
| trial_0101 | 0.89 | "good" |
| trial_0201 | 1.28 | "good" |
| trial_0301 | 3.80 | "offset negatively on y axis" |

This tracks the reported visual quality almost exactly. A 3.5–3.8mm target
error consumes more than half of the joystick's `x_slide`/`y_slide` travel
(±6mm) just compensating for the wrong target, leaving little room to also
capture the real per-frame deflection — hence "crooked"/"offset" rendering
for exactly the trials with the largest rotation mismatch. STAC's own qpos
solver is not at fault here (`tol=1e-12` in `stac_mjx/stac_core.py`, about
as tight as float precision allows) — it was accurately hitting a
systematically wrong target.

## The fix (stac-mjx, already applied and re-run)

`register_v22_mocap.py` now registers every trial using **its own
independently-fit rotation** instead of a shared/averaged one (recipe label
now `register_v22_mocap_own_rotation_v5`). The old shared-rotation average is
still computed and printed as a diagnostic only, never applied. This does
**not** reintroduce cross-trial physics inconsistency: the spring rest point
for `x_slide`/`y_slide` is `joystick_base`'s one fixed position in the
shared MJCF (`springref` defaults to 0), completely independent of how any
given trial's mocap gets registered into that frame. Registration accuracy
only affects how well STAC's fit matches the true per-frame joystick
deflection against that one fixed anchor.

All 15 trials were re-registered with this fix
(`*_registered_v22_refined.h5`, recipe `v5`).

## joystick_base anchor, re-derived again

The position anchor was re-derived a third time, this time combining two
sources rather than relying on either alone:

1. **Shaft orientation, shaft length (14.363mm), ball offset (1.822mm)**
   from the dedicated static joystick calibration recording
   (`compute_joystick_geometry_v22.py`, using
   `static_joystick_labels_withbottom.v001.slp`) — a single precise
   measurement, not per-frame-tracking-noise-prone, so reliable for these.
2. **Position anchor** taken as the median of the 12 gauge-reliable trials'
   own v5-registered `js_base` resting position directly — not the static
   calibration's own placement into model frame, which turned out to carry
   a ~3.6mm systematic bias from mixing `calibration_refined.toml`-based
   per-trial rotations with the static recording's own plain
   `calibration.toml` frame (a separate, smaller frame-mismatch issue,
   distinct from the main rotation-sharing bug above).

This combination leaves an essentially zero mean residual (<0.3mm every
axis) with tight per-trial scatter (0.7–3.7mm, on par with v21's own
documented ~2–5mm accuracy).

Current values in `stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v22_stac.xml`
(built by `build_v22_stac_xml.py`, see that file for full derivation
comments):

```xml
<body name="joystick_base" pos="-0.01264354 -0.03211015 0.04737563">
  ...
  <body name="joystick" pos="0 0 0" quat="0.94389458 -0.21721094 0.24876180 0.00000000">
    <geom name="joystick_geom" type="cylinder" size="0.001 0.0071815" pos="0 0 0.0071815" .../>
    <geom name="joystick_ball" type="sphere" size="0.002" pos="0 0 0.016184" .../>
  </body>
</body>
```

(The `quat` didn't change from the previous derivation — only `joystick_base`'s
`pos` did.)

## A separate, smaller bug also fixed (visualization only)

`stac_mjx/reproject_stac_model.py`'s virtual-camera-position computation
wasn't applying the same per-trial `scale_factor` correction that the mocap
marker overlay was already applying, causing the rendered arm+joystick mesh
to appear at inconsistent sizes across trials in reprojection *videos*. This
is purely a rendering/diagnostic-visualization issue — it does not affect
STAC's fit, the registered data, or anything vnl-playground consumes. Fixed
by applying `scale_factor` consistently in both places (`multicam.py`,
`reproject_stac_model.py`).

Separately: `scale_factor` itself (a per-trial uniform correction for
triangulation/calibration-scale error, computed as a simple ratio, not an
iterative fit) legitimately varies quite a bit across trials (0.50–1.50
across the 15 CFL_35/36/37 trials) — trials at the extreme end (e.g.
`CFL_37_trial_0201` at 0.50) are already-known poor-calibration-convergence
trials, flagged and excluded from the joystick anchor averaging above.

## Status as of writing

A full-length (126-frame, not the 15-frame smoke-test truncation) STAC run
across all 15 CFL_35/36/37 trials, with the v5 registration and the new
`joystick_base` anchor, was kicked off in `stac-mjx` and was still running
when this doc was written. The 15-frame smoke test with this same fix showed
consistently good results (no more `x_slide`/`y_slide` saturation, fit
distances in the same 1–5mm range across every previously-"bad" trial,
visually consistent alignment) — this doc will likely be updated, or a
follow-up written, once the full-length run's results are confirmed.

## Action items for vnl-playground

### 1. `joystick_base` position MUST be updated — this is not optional or open for debate

`mouse_forelimb_right_janelia_arm_hand_v22_contacts_patched.xml`
(`vnl_playground/tasks/mouse/xmls/`) — the model actually used by
`train_mouse_janelia_arm_hand.py` — currently has `joystick_base`
`pos="0.02754416 0.00213547 0.04439829"`.

**That number is v21's measured position, sitting in a v22 model file.** It
is not a v22 value of any vintage (old or new) — it's a direct transplant
from `mouse_forelimb_right_janelia_arm_hand_v21.xml`, used as a stopgap (see
`2026-07-16-v22-joystick-base-placement-bug.md`, which documents this same
transplant being tried and shown wrong on the stac-mjx side: v21's world
frame and v22's world frame are empirically ~87° apart, so a position
measured in one means nothing in the other — it's not "slightly off," it's
a different coordinate system). There is no scenario in which this specific
number is correct for v22; it was always a placeholder pending a proper
v22-frame measurement, which now exists.

**Action:** once the full-length stac-mjx run (in progress as of this
writing) confirms the anchor holds up, replace `contacts_patched.xml`'s
`joystick_base` `pos` with stac-mjx's newly-derived value (currently
`-0.01264354 -0.03211015 0.04737563` — see "joystick_base anchor, re-derived
again" above for the derivation, and check `stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v22_stac.xml`
for whatever the final confirmed value is, in case it shifts slightly after
the full-length run). The `joystick` child body's tilt `quat`
(`0.94389458 -0.21721094 0.24876180 0`) should be copied over too — it comes
from the same static-calibration measurement and hasn't changed across the
last two derivations.

### 2. Geom sizes and spring stiffness — genuinely need judgment, not a copy-paste

Unlike the position above, these are **not** a clear-cut "old value replaced
by new value" situation — investigate before changing:

1. `contacts_patched.xml`'s `joystick_geom`/`joystick_ball` sizes are
   `0.0015`/`0.003` — bigger than even stac-mjx's `0.001`/`0.002` (which
   were themselves bumped up from v22's original `0.0005`/`0.001` purely so
   the ball visually touches the shaft in stac-mjx's reprojection videos).
   These are **collision geometry** (`contype`/`conaffinity` set on both),
   not purely cosmetic — changing them changes contact/reach distance during
   training, not just how things look. `contacts_patched.xml`'s sizes may
   have been deliberately tuned larger for training (e.g. to make contact
   easier for early-stage RL) rather than simply copied from an older
   stac-mjx value — check history/intent before overwriting.
2. There is currently **no single source of truth** for this model —
   `stac-mjx`'s copy, a "shared reference" copy at
   `/root/vast/eric/janelia_model/v22/mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml`,
   and vnl-playground's `contacts_patched.xml` are three independently
   hand-edited files that have already diverged from each other. Worth
   deciding whether to keep patching all three by hand or set up something
   that keeps them in sync (even just a documented "copy these lines from
   X" step).
3. Training consumes STAC's `_ik.h5` qpos/qvel directly and re-does forward
   kinematics against the training model
   (`recompute_kinematics=True` in the reference-clip loading path) — so
   even with corrected STAC output, if `contacts_patched.xml`'s
   `joystick_base` anchor doesn't match what STAC fit against, the joystick
   will render/simulate in the wrong place relative to the hand, silently.
   `x_slide`/`y_slide` pass through unchanged from the STAC fit, but they're
   *relative to* the anchor, so the anchor has to match for those values to
   mean the same thing in both models. (This is the reason item 1's position
   update is mandatory, not optional.)
4. Also worth double-checking (not yet verified either way):
   `x_slide`/`y_slide` `stiffness`/`damping` values in `contacts_patched.xml`
   vs. stac-mjx's `v22_stac.xml` (currently `stiffness="14.8"`/`"11.2"`,
   `damping="0.0099"`/`"0.0082"`, `springref` defaulting to 0) — if these
   differ, the spring dynamics themselves differ between what STAC fit
   against and what training simulates, independent of the anchor-position
   question.

Once the full-length stac-mjx run is validated, the practical next step is
likely: regenerate the v22 reference clips vnl-playground trains on (via
whatever the v22-native equivalent of `scripts/convert_stac_v21_to_v22.py`
is — that script currently converts v21 output, not v22 output, worth
checking if that's even still the right path now that stac-mjx has its own
working v22 STAC pipeline), and update `contacts_patched.xml`'s
`joystick_base`/geom-size/spring values to match stac-mjx's `v22_stac.xml`,
after resolving the open questions above.
