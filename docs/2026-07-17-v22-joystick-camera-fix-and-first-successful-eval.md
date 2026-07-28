# v22 arm+hand joystick/camera fix — root cause, fix, and first successful eval

Written 2026-07-17. Documents how the v22x/v23 "zero eval past 0" regression
was diagnosed and fixed (Track A), and the smoke-test run that confirmed it.

## TL;DR

`v22x_no_joystick.xml` and `v23_simplified_collision.xml` never produced a
useful eval past the 0-step mark. Root cause: reverting `joystick_base` to
"the raw STAC-fit position" (in an attempt to stop hand-patching the
joystick) silently reintroduced a real ~38.6-degree shaft tilt that **no
run that ever actually trained successfully has had** — not the checkpoint
at `checkpoints/janelia-v22-arm-hand-20260716-155637-full-run-v1`, and not
the later `contacts_patched.xml` run in
`logs/janelia_v22_arm_hand_full_run.log` (reward -49 -> 305, 0% NaN, 5/5
evals). `v23` inherited that tilt but not the `ccd_iterations=100`
compensation a sibling experimental file had already found necessary for
it, which is almost certainly why it was unstable.

Fix: keep the joystick **untilted** (matches every run that's ever actually
trained), and add *only* a fixed diagnostic camera on top of the already-
proven `contacts_patched.xml` geometry — this is what "fix the camera, not
the joystick" should have meant. New file:
`mouse_forelimb_right_janelia_arm_hand_v22_final_camera.xml`.

A short training smoke test (5M steps, 6-clip subset) on this file
completed clean: no NaN, bounded contacts, and the eval-render video at
step 5,242,880 shows a physically plausible arm+joystick interaction with
the joystick rendered vertical in-frame — clearly better than v22x/v23's
behavior. One known, separate, pre-existing issue was also confirmed by
this run (not fixed yet, see "Open follow-up" below): the policy oscillates
at ~400Hz because of a control-rate/reference-rate mismatch that's
unrelated to the joystick/camera fix.

## Root cause, in detail

Two full 500M-step runs of the v22 arm+hand+joystick model actually
succeeded before this investigation:

1. `checkpoints/janelia-v22-arm-hand-20260716-155637-full-run-v1` — used
   the **unmodified** `/root/vast/eric/janelia_model/v22/mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml`.
   `joystick_base` at `pos="-0.01685188 -0.02594152 0.04776544"`, no `quat`
   (upright).
2. The run behind `logs/janelia_v22_arm_hand_full_run.log` — used
   `mouse_forelimb_right_janelia_arm_hand_v22_contacts_patched.xml`, whose
   `joystick_base` was hand-shifted to
   `pos="-0.00447064 -0.02640225 0.04534320"` (still **no** `quat`,
   i.e. still upright) to force closer fingertip-to-ball contact.

Both trained cleanly to reward -49 -> 305 with 0% NaN termination across
all 5 evals. **Neither ever had a tilted joystick.**

Later, believing the hand-shift itself was "the big mistake," the joystick
was reverted to the raw stac-mjx-synced pose in
`mouse_forelimb_right_janelia_arm_hand_v22_raw_joystick_camera.xml`:
`joystick_base` at `pos="-0.01264354 -0.03211015 0.04737563"`, **`quat="0.94389458 -0.21721094 0.24876180 0"`** — a genuine ~38.6-degree tilt
(`2*acos(0.94389458)`). That file's own comment already flags the
consequence: the tilt makes hand-vs-cylinder contact oblique instead of
face-on, "a slow-converging case for iterative convex collision," and adds
`ccd_iterations="100"` (MuJoCo default is 35) to compensate — an
explicitly **unverified** mitigation. `v23_simplified_collision.xml`
inherited the exact same tilted `joystick_base` pose but **not** the
`ccd_iterations` bump, and its own handoff doc
(`docs/2026-07-17-v23-handoff.md`) confirms no long training run had ever
validated it.

So: translating the joystick (either raw or hand-shifted) was never the
problem — both proven runs did that, in different ways. The tilt was the
new, untested variable, and it landed in exactly the two configs
(`raw_joystick_camera`/`v23`) that never worked.

## The fix

New file:
`vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_arm_hand_v22_final_camera.xml`

- Byte-identical to `mouse_forelimb_right_janelia_arm_hand_v22_contacts_patched.xml`
  (the proven, untilted, hand-shifted joystick) **except** for one addition:
  a fixed diagnostic camera, `arm_joystick_fixed_view`.
- The camera was **re-derived from scratch** for this file's untilted
  joystick pose — it does *not* reuse the tilted variant's camera (that one
  is rolled specifically to compensate for a 38.6-degree tilt this file
  doesn't have, and would frame the wrong thing here). Derivation method
  (same recipe as `docs/2026-07-17-v22-fixed-camera-angle-for-visualization.md`,
  computed via `mj_forward` at qpos=0 against this file):
  - `up` = joystick shaft's world +Z direction (trivially `[0,0,1]` since
    untilted; used the documented `[1,0,0]` fallback for the `right`-axis
    cross product to avoid the degenerate parallel case).
  - `right` = `normalize(cross(up, world_ref))`.
  - `forward` = `-cross(right, up)`.
  - `lookat` = midpoint of `humerus.xpos` and `joystick_base.xpos`.
  - `distance` = `span*1.15 + 0.015`, `span` = distance between those two
    points.
  - Result:
    ```xml
    <camera name="arm_joystick_fixed_view"
            pos="0.04244483 -0.02029550 0.05838954"
            xyaxes="0.00000000 1.00000000 0.00000000  0.00000000 0.00000000 1.00000000"
            fovy="45"/>
    ```
  - Verified by rendering a frame: joystick shaft renders vertical,
    shoulder/arm and joystick both fit in the same shot.

Wiring changes:
- `vnl_playground/tasks/mouse/consts.py`: `JANELIA_MOUSE_ARM_HAND_V22_XML_PATH`
  now points at `..._v22_final_camera.xml` instead of
  `..._contacts_patched.xml` directly (same physics, plus the camera).
- `vnl_playground/tasks/mouse/imitation_arm_hand.py`: `default_config()`
  now also sets `cfg.keep_clips_idx = [1, 2, 3, 4, 13, 14]` — the same
  6-trial subset already used for v22x (re-verified against
  `MOUSE_REFERENCE_DATA_JANELIA_V22_PATH`'s actual directory listing,
  15 entries, same trial names/order as documented in
  `docs/2026-07-17-v23-handoff.md`). Previously the with-joystick configs
  trained on all 15 trials; now they match the intended subset.
  `default_config_v23()` and `default_config_raw_joystick()` inherit this
  automatically since they build on `default_config()`.

## Validation performed

1. **Model compiles, camera loads**: `nq=27`, `ngeom=48`, `ncam=4`
   (`arm_joystick_fixed_view`, `my_camera`, `close-profile`, `janelia_cam`).
   `ncon` at rest = 14, identical to the unmodified `contacts_patched.xml`
   baseline (confirms the camera addition is physically inert).
2. **300-step random-action stress test** (`MouseImitationArmHand`,
   `mujoco_impl="warp"`, full `default_config()`): **zero NaN** over 300
   steps of uniform random ±1 torques on all 35 muscles (a harder stress
   test than early real PPO exploration). Contact count stayed bounded
   (0-13 active contacts, well under the `naconmax=16384` budget).
   `done` flag never fired (no early termination).
3. **Camera render check**: rendered a frame from `arm_joystick_fixed_view`
   directly (required `LD_LIBRARY_PATH=/root/miniforge3/envs/track_mjx/lib`
   for a working EGL loader in this shell — the repo's own `.venv` is
   missing the system `libegl1` package `ctypes.util.find_library` needs;
   unrelated to this fix, just a local shell quirk) — joystick vertical,
   arm+joystick both framed, as designed.
4. **Training smoke test**: `train_mouse_janelia_arm_hand.py --tag
   track-a-smoke --no-wandb --num-timesteps 5000000 --eval-every 1000000`.
   Completed cleanly (process exited normally, no traceback). Checkpoint:
   `checkpoints/janelia-v22-arm-hand-20260717-060110-track-a-smoke/`.
   - `config.json` confirms: `walker_xml_path` = the new final_camera file,
     `keep_clips_idx=[1,2,3,4,13,14]`, `termination_criteria` has only
     `nan_termination` (no `pose_error`), `njmax=512`/`naconmax=16384`.
   - Two eval videos saved: `0.mp4` (initial/untrained policy) and
     `5242880.mp4` (after ~5.24M env steps). Both rendered successfully.
   - `5242880.mp4` shows plausible arm+hand+joystick interaction — no limbs
     flying apart, no joystick clipping through the mesh, joystick renders
     vertical as intended. Qualitatively much better than what v22x/v23
     were producing.
   - `broadphase overflow -- please increase nconmax/naconmax` warnings
     appeared during training (e.g. "naconmax to 86453"), same warning
     class both historically-successful runs threw hundreds of thousands
     of times without it ever blocking training — present, not alarming
     on its own.

This is the first with-joystick, 6-clip-subset training run to produce a
legible, physically-plausible eval past step 0 since the regression began.

## Open follow-up (not yet fixed, tracked separately from this fix)

The trained policy in `5242880.mp4` visibly oscillates very fast (looks
like ~400-500Hz chatter). Root cause, confirmed from `config.json`:
`ctrl_dt=0.0025` (400Hz control rate) while `mocap_hz=25` means the STAC
reference target only updates every ~16 control steps — a "staircase"
target with nothing pulling the policy toward a smoothly changing goal in
between, while the fast control loop still lets it chatter freely between
updates. This is a **pre-existing, already-documented** tension (see the
comment block in `imitation_arm_hand.py`'s `default_config()` around
`cfg.ctrl_dt`), not a regression introduced by the joystick/camera fix —
it was present in the same form in the proven v1/v3 runs, just less
visible there.

Proposed next step (discussed, not yet applied as of this doc): move to
2 control steps per mocap frame instead of 16, i.e. `ctrl_dt=0.02`
(`0.04s mocap frame / 2`), paired with `sim_dt=0.001` rather than
`sim_dt=0.00125` — the existing code comment notes `sim_dt=0.00125` at a
`ctrl_dt=0.02`-like rate previously showed a 24.2% NaN rate in one test,
while `sim_dt=0.001` had one clean eval (though still unproven at full
scale). This also requires updating
`train_mouse_janelia_arm_hand.py`'s hardcoded `ppo_params.episode_length`
from `2016` down to `252` (`= 126 frames / (0.02*25) frames-per-step`), per
that file's own comment noting the two must move together or episodes
stop partway through each clip. **Not implemented yet** — flagged here so
it isn't lost, and so any future smoke test re-validates NaN rate at scale
before trusting it, exactly as the existing comments already caution.

## Files touched by this fix

- `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_arm_hand_v22_final_camera.xml` (new)
- `vnl_playground/tasks/mouse/consts.py` (`JANELIA_MOUSE_ARM_HAND_V22_XML_PATH` repointed)
- `vnl_playground/tasks/mouse/imitation_arm_hand.py` (`default_config()` gains `keep_clips_idx`)

## Files deprioritized, not deleted

- `mouse_forelimb_right_janelia_arm_hand_v22_raw_joystick_camera.xml`,
  `..._raw_joystick_capsule.xml`, `..._v23_simplified_collision.xml` — all
  carry the untested tilt. Worth revisiting only if the (currently benign)
  broadphase-overflow warning volume ever becomes an actual wall-clock
  problem, and only ever with the tilt+mitigation combination validated at
  full scale first.
- `mouse_forelimb_right_janelia_arm_hand_v22x_no_joystick.xml` — unrelated
  arm-only control experiment; its own separate issues (already-fixed
  `pose_error` termination deletion, `njmax`/`naconmax=64` only validated
  at small scale, missing fixed-camera due to a single-root `attach_body()`
  limitation) are tracked independently of the joystick/camera fix.
