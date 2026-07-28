# Janelia v22 arm+hand model + STAC v21 training — bring-up plan

**Goal:** Bring `eric/new-janelia` up to parity with the `eric/janelia` mouse-imitation
training pipeline, then extend it to train against the new
`mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml` model (35 muscles,
27 joints, full arm+hand+joystick) using the STAC v21 dataset at
`/root/vast/eric/stac-mjx/refined_STACed_data_v21/` (3 animals × 5 trials).

**Architecture:** Port the janelia-specific training modules from `eric/janelia`
onto `eric/new-janelia` (Phase 1). Fix the v22 model's stale macOS mesh path
(Phase 2). Write a one-off data-adaptation script that converts each STAC v21
`_ik.h5` clip to the v22 model's 27-dof qpos layout by **name**, since v22
deliberately dropped the `z_slide` joint the v21 fit used (Phase 3 — this is
already anticipated in the v22 XML's own comments). Add a new `MouseImitation`
task variant sized for the full hand kinematic chain and 35-muscle actuator
space (Phase 4). Add a new training entrypoint script, this time with a real
`--xml-path`/`--data-path` CLI surface instead of hardcoded python constants
(Phase 5). Smoke-test via FK replay before spending GPU time on PPO (Phase 6).

**Tech stack:** Python 3.12, mujoco/mjx, brax PPO, jax, wandb, orbax
checkpoints — all unchanged from the existing janelia pipeline.
`.venv/bin/python` for data/asset scripts (h5py, no jax needed);
`source /root/vast/eric/track-mjx/.venv/bin/activate` for anything that
imports mujoco/brax/jax (matches the convention used by the existing s15-ms
plan).

**Sources of truth used to write this plan:**
- `eric/janelia` branch (`git show eric/janelia:<path>` — do not check it out,
  working tree stays on `eric/new-janelia` with `stash@{0}` pending)
- `stash@{0}` ("On eric/janelia: eric/janelia stash") — optional/secondary;
  it's mostly EMG/figure analysis notebooks and docs, not core training code.
  Not required for this plan; revisit separately if useful later.
- `/root/vast/eric/janelia_model/v22/mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml`
- `/root/vast/eric/stac-mjx/refined_STACed_data_v21/` (15 trial dirs + `refined_rerun_fit_summary.json`)

**Assumption (flag if wrong):** train on all 3 animals (CFL_35/36/37) pooled
in one run for this first pass, matching how the existing pipeline already
pools multiple trials from one animal. Per-animal training (there's already
a per-animal trainer design doc on `eric/janelia`) is a follow-up, not part
of this bring-up.

**Decisions from Eric (2026-07-16), resolving the plan's original open
questions 1 and 3:**
- Converted STAC v21→v22 reference data lives at the root level, outside the
  repo entirely — `/root/vast/eric/refined_STACed_data_v22/` — not nested
  under `vnl_playground/tasks/mouse/...`. (Resolves old open question 2.)
- **Shoulder translation (`sh_tx`, `sh_ty`, `sh_tz`) is IK-driven** — snapped
  from the STAC reference every step, same mechanism as
  `MouseImitationMovingShoulder._override_ik_dims`, generalized to a
  configurable slice of qpos rather than assuming the leading dims (in v22's
  27-dim layout, `sh_tx/ty/tz` sit at indices 2–4, *after* the joystick's
  `x_slide`/`y_slide` at 0–1). The policy only has to actuate everything
  else — elbow, forearm_supination, wrist_flexion, `rz_N_L_C_right`, and the
  16 finger joints. (Resolves old open question 3/Task 4 Step 4.1.)
- **Joystick `x_slide`/`y_slide` are NOT IK-driven** — they stay physically
  simulated, moved only by hand-joystick contact (this is exactly why the
  `_contacts` xml variant was picked). Joystick `z_slide` doesn't exist in
  v22 at all and was never wanted — confirmed, no change needed there, v22
  already matches this intent as-is.

---

## Task 1: Port janelia mouse-imitation training code onto `eric/new-janelia`

**Why:** `new-janelia` is missing the janelia-specific arm-imitation stack
entirely — confirmed via `git diff eric/new-janelia...eric/janelia --stat`.
Everything below depends on this being in place first.

**Files to bring over from `eric/janelia`** (via
`git checkout eric/janelia -- <path>`, reviewed individually — don't blanket
merge, `new-janelia` has since-diverged files like `tests/`,
`eval_metrics/emg.py`, `sweep_sAnimal_*.sh` that must NOT be clobbered):

- [x] `vnl_playground/train_mouse_janelia.py` (plain PPO target-reach, no imitation)
- [x] `vnl_playground/train_mouse_janelia_imitation.py` (PPO imitation, real argparse CLI)
- [x] `vnl_playground/tasks/mouse/imitation_moving_shoulder.py` (`MouseImitationMovingShoulder` — best structural template for "snap some qpos dims kinematically, actuate the rest via muscles")
- [x] `vnl_playground/tasks/mouse/consts.py` — merge by hand, not overwrite: add `JANELIA_MOUSE_XML_PATH`, `JANELIA_AKIRA_XML_PATH`, `JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH`, `MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH` on top of whatever `new-janelia` already has
- [x] `vnl_playground/tasks/mouse/base.py`, `imitation.py` — diff against `new-janelia`'s current versions before overwriting (both have independent changes)
- [x] `vnl_playground/tasks/mouse/reference_clips.py` (`MouseReferenceClips` — the loader we'll reuse for v21 STAC data; only a 2-line diff)

**Do not port:** `vnl_playground/tasks/mouse/reference_data_moving_shoulder/*.h5`
(the old STAC v16 single-animal set) — not needed, we're wiring up STAC v21
instead. `train_mouse_janelia_intention.py`,
`train_mouse_janelia_sigmoid_moving_shoulder.py` — reference material only,
already present in some form on `new-janelia`; leave as-is unless Task 4 needs
to crib from them.

- [x] **Step 1.1:** Ran each `git checkout eric/janelia -- <path>` above; diffed `base.py`/`imitation.py`/`reference_clips.py`/`consts.py` by hand before applying (all were clean, one-directional additive changes from janelia — no new-janelia-only code was clobbered).
- [x] **Step 1.2:** `train_mouse_janelia_imitation.py` has no `__main__` guard — it runs training at module level, so importing it would launch a real run. Used `py_compile` (syntax-only) on it instead, plus a real `import` on the safe modules (`imitation.py`, `imitation_moving_shoulder.py`, `consts.py`). All clean.

---

## Task 2: Fix the v22 model's mesh path and register it in `consts.py`

**Problem found:** `mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml`
line 51 has `<compiler angle="degree" meshdir="/Volumes/talmo/eric/janelia_model/v22"/>`
— a stale macOS mount path. On this box the same tree lives at
`/root/vast/eric/janelia_model/v22`, so meshes currently fail to resolve.

- [x] **Step 2.1:** Edited `meshdir` in `/root/vast/eric/janelia_model/v22/mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml` to `/root/vast/eric/janelia_model/v22` (matches the existing repo convention of pointing at absolute, machine-specific, gitignored external paths — see `EMG_DIR`/`TRIAL_CSV` constants in `train_mouse_janelia_sigmoid_moving_shoulder.py`). Do not copy the mesh tree into the repo; there's no precedent for that and it'd duplicate ~1.3 MB of binary assets that already live in the right place.
- [x] **Step 2.2:** Added `JANELIA_MOUSE_ARM_HAND_V22_XML_PATH` and
  `MOUSE_REFERENCE_DATA_JANELIA_V22_PATH` (pointing at
  `/root/vast/eric/refined_STACed_data_v22`, per the root-level decision
  above) to `vnl_playground/tasks/mouse/consts.py`.
- [x] **Step 2.3:** Smoke-loaded — confirmed `nq=27`, `nv=27`, `nu=35` (muscles, not 53 — the XML changelog's "53 muscles discovered in the rig" counts everything found during rig extraction, not all included here), `nbody=30`. Joint order matches the expected v22 layout (`x_slide, y_slide, sh_tx, sh_ty, sh_tz, sh_rotation, sh_extension, sh_elv, elbow, forearm_supination, wrist_flexion, rz_N_L_C_right`, then 16 finger joints).

---

## Task 3: Adapt STAC v21 reference clips to the v22 model's qpos layout

**The core compatibility issue:** the v21 STAC fit has **28** qpos dims
(`x_slide, y_slide, z_slide, sh_tx, sh_ty, sh_tz, ...`); v22 has **27** —
`z_slide` was deliberately removed (confirmed: `grep z_slide` on the v22 xml
finds nothing under `<joint>`, and the XML's own comment block says
*"joint angles transfer by name; joystick z_slide has no [equivalent] ...
x_slide=0/y_slide=0 at this anchor is now the STAC-aligned rest position"*).
So the fix is a **name-based** column drop, not a raw reshape. Also: v22 adds
one new body (`radius`) not present in v21's `names_xpos` — but
`MouseReferenceClips.recompute_kinematics(mj_model)` already exists precisely
for this (recomputes `xpos`/`xquat` via forward kinematics from `qpos` using
whatever model you pass it), so xpos/xquat don't need manual patching.

- [x] **Step 3.1:** Verified programmatically across all 15 v21 trials:
  every trial's `names_qpos` drops exactly `['z_slide']` and the remaining
  27 names match v22's joint order exactly. Safe to hardcode the name-based
  column drop.
- [x] **Step 3.2:** Wrote `scripts/convert_stac_v21_to_v22.py`:
  - Input: `/root/vast/eric/stac-mjx/refined_STACed_data_v21/<trial>/<trial>_ik.h5`
  - For each file: load `qpos`, `qvel`, `names_qpos`; build an index map from
    v22's joint-name order into the source array (drop any name not present
    in v22, i.e. `z_slide`); write `qpos`/`qvel` sliced+reordered accordingly.
  - Copy through `kp_data`, `kp_names`, `marker_sites`, `offsets` unchanged
    (markers are model-independent).
  - Set `names_qpos` to v22's order. `MouseReferenceClips._load_from_disk`
    unconditionally reads `xpos`/`xquat`/`names_xpos` (no None-check), so
    these can't be omitted — wrote zero-filled placeholders
    (`(n_frames, v22_nbody, 3)` / identity-quaternion `(n_frames, v22_nbody, 4)`)
    plus v22's body names; `recompute_kinematics` overwrites all three at
    load time so the placeholder values themselves don't matter.
  - Update the embedded `config` string's `MJCF_PATH` to point at the v22 xml
    (provenance only, not consumed by `MouseReferenceClips`).
  - Output: one `_ik.h5` per trial under `/root/vast/eric/refined_STACed_data_v22/`
    (root-level, outside the repo, per Eric's 2026-07-16 decision above).
- [x] **Step 3.3:** Ran the converter over all 15 trials -> all wrote
  successfully to `/root/vast/eric/refined_STACed_data_v22/`; every output's
  `qpos.shape == (126, 27)`.
- [x] **Step 3.4:** Loaded all 15 clips via `MouseReferenceClips`, called
  `.recompute_kinematics(v22_model)` — no NaNs, wrist xpos stays in a
  plausible ~1cm-scale reach envelope for clip 0. Full visual FK-replay
  video check deferred to Task 6, Step 6.1 (this was just a numeric sanity
  check, not a visual one).

---

## Task 4: New task class for the full arm+hand+joystick model

**Why not reuse `imitation.py`/`imitation_moving_shoulder.py` as-is:** both
hardcode `tracked_bodies = ["scapula", "humerus", "ulna", "wrist"]` and were
built for a 12-muscle, 4–7 DOF arm. The v22 model has 35 muscles and 27 DOFs
across a full finger chain — reward terms (`joints`, `joints_vel`,
`pose_error`) need to see the whole `names_xpos`/`names_qpos` set, not just
the wrist.

- [x] **Step 4.1:** Implemented generalized IK-driven-dims snapping.
  `MouseImitationMovingShoulder._override_ik_dims` assumes the snapped dims
  are `qpos[:n]` (a leading slice) — not true here, since `x_slide`/`y_slide`
  (joystick, indices 0–1, physically simulated, contact-driven) come *before*
  `sh_tx/sh_ty/sh_tz` (indices 2–4, IK-driven) in v22's qpos layout. Add a
  `config.ik_driven_qpos_idx = [2, 3, 4]` (or similar explicit index list,
  not just a count) to the new task's config, and generalize
  `_override_ik_dims` to snap `qpos.at[idx].set(ref.qpos[idx])` /
  `qvel.at[idx].set(ref.qvel[idx])` for arbitrary indices, and to mask those
  same indices (not just a leading range) out of `joints`/`joints_vel`
  rewards and `pose_error` termination. Everything else (elbow,
  forearm_supination, wrist_flexion, `rz_N_L_C_right`, 16 finger joints) is
  muscle-actuated by the policy; joystick `x_slide`/`y_slide` are left alone
  entirely (not snapped, not masked — fully physically simulated via
  hand-joystick contact, per Eric's 2026-07-16 decision above).
- [x] **Step 4.2:** Created `vnl_playground/tasks/mouse/imitation_arm_hand.py`
  (name TBD) with `default_config()` setting `walker_xml_path =
  JANELIA_MOUSE_ARM_HAND_V22_XML_PATH`, `reference_data_path` = Task 3's
  output dir, `tracked_bodies` = the full 28-body chain (or a curated subset
  — fingertips + wrist + elbow + shoulder, mirroring the STAC `KEYPOINT_MODEL_PAIRS`
  used to fit the data: `humerus, ulna, wrist,
  Phalanx_hand_{1..5}_{1,2,3}_right, joystick`), `end_effector = "wrist"`.
- [x] **Step 4.3:** `recompute_kinematics = True` set in this config.

**Two significant issues discovered and fixed while smoke-testing this task
(neither was anticipated in the original plan):**

1. **`add_mouse()` silently dropped 5 DOFs.** v22's kinematic tree has *two*
   disconnected roots under `worldbody`: `shoulder_base` (carries
   `sh_tx/ty/tz`, parent of `clavicle`) and a separate `joystick_base`
   (carries `x_slide/y_slide`). The shared `add_mouse()`/`add_ghost_mouse()`
   helpers in `base.py` hardcoded attaching a single named root
   (`"clavicle"`, the true root in older single-root models), so attaching
   only `"clavicle"` silently discarded both `shoulder_base`'s and
   `joystick_base`'s joints -- compiled model came back with `nq=22`, not
   27. Generalized both helpers to accept a `root_bodies` sequence (default
   `("clavicle",)`, zero behavior change for every existing single-root
   model); for multi-root callers, uses MjSpec's whole-model `attach()`
   (imports the child's entire `<asset>` table exactly once, so it doesn't
   collide the way two `attach_body()` calls from the same source file do)
   instead of attaching bodies one at a time. `imitation.py`'s `add_mouse()`
   call and its `render()` ghost-overlay builder both now thread
   `self._config.root_bodies` through. New `cfg.root_bodies` field added to
   `base.py`'s `default_config()`; `imitation_arm_hand.py` sets it to
   `("shoulder_base", "joystick_base")`.
2. **MJX's pure-JAX backend doesn't support cylinder-vs-mesh collisions.**
   The `_contacts` xml's joystick geoms are cylinders (`joystick_geom`,
   `circular_base`) meant to collide with the hand's mesh geoms --
   `mjx.put_model(..., impl="jax")` raises
   `NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_MESH) collisions not
   implemented`. The Warp backend (`impl="warp"`, already installed,
   confirmed 2x NVIDIA A40 available) handles it fine --
   `imitation_arm_hand.py`'s `default_config()` sets `cfg.mujoco_impl =
   "warp"`. This means **Task 5's training script needs to run under the
   Warp MJX backend for this task**, not the plain-JAX backend the other
   janelia scripts use.

Smoke-tested end to end: `MouseImitationArmHand(default_config())`
constructs, `reset()` gives a finite reward with no NaNs, 5x `step()` with
the null action stays finite. `qpos[0:5]` (joystick x/y, then sh_tx/ty/tz)
shows the IK-driven shoulder dims snapped near the reference and the
joystick dims free to evolve under contact physics, as intended.

---

## Task 5: New training entrypoint with a real CLI (no more hardcoded xml constants)

**Why:** every existing `train_mouse_janelia*.py` script hardcodes its xml
path via a python import, not a flag — fine when there's one model, painful
now that we have akira / v21-arm-only / v22-arm-hand all live at once.

- [x] **Step 5.1:** Created `vnl_playground/train_mouse_janelia_arm_hand.py`,
  modeled on `train_mouse_janelia_imitation.py`'s CLI + PPO setup, reusing
  its physics-override flags. Added `--xml-path`/`--data-path` flags
  (default to `default_config()`'s v22 paths if omitted). `build_render_model`
  needed the same single-vs-multi-root attach branching as `base.py`/
  `imitation.py` (Task 4's discoveries) -- factored into a local
  `_attach_walker` helper. Verified standalone (without triggering the
  script's top-level argparse/wandb side effects, since importing the
  module runs everything up to the `if __name__ == "__main__"` guard): the
  render model compiles (`nq=54` = 27 main + 27 ghost, 9 cameras correctly
  suffixed `-mouse`/`-ghost`), and `mj_forward` runs clean.
- [ ] **Step 5.2:** Decide multi-animal handling for this run: pool all 15
  clips as one `reference_data_path` (simplest -- `MouseReferenceClips`
  already handles multi-clip directories, and this is what the script does
  today with no `--animal` flag) vs. add a `--animal` filter using
  `keep_clips_idx`. Still pooling for the first pass per this plan's stated
  assumption; revisit if Task 6's smoke run shows cross-animal pose scale
  mismatches.

---

## Task 6: Smoke test before spending GPU time

- [ ] **Step 6.1:** FK replay check — load Task 3's converted clips + v22
  model, step through one trial's `qpos` frames with `mj_forward`, render or
  dump a short video, visually confirm the arm+hand tracks a plausible reach
  (this catches joint-order/sign bugs cheaply; render tooling already exists
  in `vnl_playground/tasks/mouse/visualize.py`).
- [ ] **Step 6.2:** Short PPO smoke run (few thousand timesteps, small batch)
  via Task 5's script — confirm it compiles, steps, and checkpoints without
  NaNs before committing to a full run.

---

## Task 7: Full training run

- [ ] **Step 7.1:** Launch the real run (wandb group/tags per repo
  convention, e.g. `wandb-group=janelia-v22-arm-hand`), sized per available
  GPU budget — hyperparameters TBD from Task 6's smoke run behavior, not
  fixed here.
- [ ] **Step 7.2:** Eval against held-out STAC trials / EMG comparison if
  applicable (mirrors `scripts/emg_comparison.py`'s pattern), once a
  checkpoint exists.

---

## Open questions to confirm before/while executing

1. **Multi-animal pooling vs. per-animal runs** — this plan defaults to
   pooling all 3 animals for the first pass (see Assumption above). Redirect
   me if you want per-animal from the start.
2. **Tracked-bodies set for the new task class** (Task 4, Step 4.2) — full
   28-body chain vs. a curated subset matching the STAC `KEYPOINT_MODEL_PAIRS`.
   Affects reward shaping; worth a deliberate choice, not a default.

**Resolved (2026-07-16):** reference-data location (root-level, outside
repo) and shoulder-translation IK-driving — see Decisions block above.


---

## Investigated and retracted (2026-07-16): hand/joystick "misalignment" was a measurement artifact, not a real issue

Eric observed the joystick relaxing back to qpos=0 during early rollouts and
asked whether the arm and joystick were actually registered correctly in the
reference data. My first check compared body-origin `xpos` (wrist/fingertip
*joint* positions) to the joystick body's `xpos` and found ~11-18mm minimum
separation across all 15 clips -- looked like a real misalignment, logged as
a known issue.

**That check was wrong.** Body `xpos` gives a joint origin, not the physical
marker/fingertip location -- the STAC fit's `D1_T`..`D5_T` fingertip markers
are attached to bodies via a per-marker `offset` (see each clip's `offsets`
dataset) that can put the actual marker centimeters away from that body's
joint origin, especially since some tip markers are assigned to a mid-chain
phalanx body (e.g. `D1_T` -> `Phalanx_hand_2_1_right`), not the most distal
bone.

**Corrected check, using `marker_sites` (the actual fitted keypoint
positions) instead of body origins:** fingertip-to-joystick-marker distance
gets as close as 1.37-1.48mm in `CFL_35_20240128_trial_0001` (joystick ball
is ~1-2mm radius -- real contact), matching what Eric found independently
by inspecting the STAC data directly.

**Confirmed this isn't a v21->v22 conversion artifact either:** ran the same
flawed body-origin check against the *original, unconverted* v21 model with
its own native 28-dim STAC qpos (no Task 3 conversion involved at all) --
got the same ~15-17mm gap. Since the gap is present even in the pristine
original data/model, it was never a rest-pose-mismatch or conversion bug --
just the wrong measurement (body origin vs. marker site) applied
consistently wrong in both cases.

**Conclusion:** the reference data is correctly registered; no data or
model fix needed. The joystick relaxing to qpos=0 in early rollouts is
simply an untrained policy not yet directing the actual hand geometry to
contact the joystick -- expected at this stage, not a bug. The
[[project-v22-joystick-hand-misalignment]] memory has been corrected to
reflect this (see memory file for the retraction).

---

## SUPERSEDED (2026-07-16, later same day): the real bug was upstream in stac-mjx, not vnl-playground

Everything in the two sections above this one (the "misalignment" retraction,
and the later joystick-position/clavicle patching this plan describes) was
downstream troubleshooting of a bug that actually lived in stac-mjx's mocap
registration pipeline, not in this repo. Full picture, in order of what was
actually true:

1. **Real root cause**: `register_v22_mocap.py` (stac-mjx) forced all 15
   CFL_35/36/37 trials to share ONE Procrustes rotation when registering raw
   triangulated mocap into model-frame coordinates, on the false premise
   that "the joystick and cameras are physically identical across all 15
   trials." `calibration_refined.toml` (the per-trial bundle-adjustment-
   refined calibration actually used for triangulation) is measurably
   different per trial -- 8-21 degrees of rotation difference between
   trials' camera extrinsics, even within one same-day session. This baked
   a real, trial-dependent registration error into the joystick's fitted
   position, up to ~3.8mm -- more than half the joystick's +-6mm x_slide/
   y_slide travel.
2. **Fixed upstream**: `register_v22_mocap.py` now uses each trial's own
   independently-fit rotation (recipe `register_v22_mocap_own_rotation_v5`).
   All 15 trials re-registered; `joystick_base`'s position re-derived a
   third time (combining static-calibration shaft geometry with the median
   of 12 gauge-reliable trials' own v5-registered `js_base` position).
3. **This means Task 3's whole v21-to-v22 conversion approach
   (`scripts/convert_stac_v21_to_v22.py`, name-matching/transplanting v21's
   qpos into v22) is superseded.** stac-mjx now fits STAC directly against
   v22 (`mouse_forelimb_right_janelia_arm_hand_v22_stac.xml`), producing
   native 27-dof qpos/xpos/xquat that need no conversion at all. Verified:
   `v22_stac.xml` (the model STAC actually fit against) + this native data
   gives fingertip-to-joystick-ball distance 1.8-4.8mm across a full
   126-frame trial (`CFL_35_20240128_trial_0001`) -- real contact, matching
   v21's own accuracy.
4. **The intermediate bone-frame patches (clavicle, ulna, wrist) documented
   earlier in this plan were based on a wrong diagnosis and have been
   reverted.** Comparing `v22_stac.xml` to the original (unpatched)
   `mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml` shows v22's arm
   bones (`clavicle`, `scapula`, `humerus`, `ulna`, `wrist`, all 5
   `Metacarpal_hand_*_right` bodies) are byte-for-byte (or ~0mm/~0deg)
   identical to what STAC was actually fit against -- they were never
   wrong. The entire ~10-36mm gap chased through several rounds of bone
   patching in this plan was 100% the joystick registration bug; patching
   arm bones was solving a problem that didn't exist there.

**Current state of `contacts_patched.xml`** (the training model): arm bones
match the original, unpatched reference model exactly (all patches to
clavicle/ulna/wrist reverted). Only the joystick block (`joystick_base`
pos, `joystick`'s tilt `quat`, `joystick_geom`/`joystick_ball` sizes) is
patched, synced from `v22_stac.xml`:

```xml
<body name="joystick_base" pos="-0.01264354 -0.03211015 0.04737563">
  ...
  <body name="joystick" pos="0 0 0" quat="0.94389458 -0.21721094 0.24876180 0.00000000">
    <geom name="joystick_geom" type="cylinder" size="0.001 0.0071815" .../>
    <geom name="joystick_ball" type="sphere" size="0.002" pos="0 0 0.016184" .../>
  </body>
</body>
```

**Data pipeline changed**: `consts.py`'s `MOUSE_REFERENCE_DATA_JANELIA_V22_PATH`
now points directly at stac-mjx's own output,
`/root/vast/eric/stac-mjx/refined_STACed_data_v22` (not a vnl-playground-local
copy -- this may grow as more trials finish there; only the 5 CFL_35 trials
were done as of this writing, CFL_36/37 -- 10 more trials -- pending).
`imitation_arm_hand.py`'s `cfg.recompute_kinematics` changed `True` ->
`False` (verified: stored `xpos` in this data matches our model's own FK to
0.0000mm, no recompute needed). `MouseReferenceClips._load_from_disk`'s glob
generalized to recursive `**/*_ik.h5` (was flat `*.h5`) to handle stac-mjx's
per-trial-subdirectory layout, still backward compatible with the old flat
datasets.

**Smoke-tested end to end** with the corrected model + real data: env
constructs, loads 5 clips (126 frames each), reset/step run clean, and a
direct render at the closest-approach frame (clip
`CFL_35_20240128_trial_0001`, frame 104) shows genuine visual hand-joystick
contact matching the 1.80mm numeric result.

**Not yet done**: incorporating CFL_36/37 once those trials finish on the
stac-mjx side (should need no code changes, `MouseReferenceClips` globs
whatever's in the directory); a real training run against this corrected
setup (the completed 500M-step run from earlier today used the buggy
joystick geometry throughout and should not be treated as informative about
tracking quality).


---

## FINAL joystick geometry (2026-07-16, later still): upright, position-adjusted, not rotated

The stac-mjx-derived tilt quat (`0.94389458 -0.21721094 0.24876180 0`)
described above was rejected after visual review: Eric confirmed the real
mounting bracket sits flush to the x,y plane with the shaft rising straight
up along Z -- no tilt anywhere, on either the base plate or the shaft.
Applying the tilt only to the inner "joystick" body (shaft+ball) also
produced a structurally implausible bent joint (shaft not perpendicular to
its own mounting plate); applying it to `joystick_base` instead (rotating
base+shaft together) fixed that structural issue but still left the whole
assembly visibly non-vertical, which Eric rejected on direct knowledge of
the physical rig.

**Resolution, per Eric's explicit instruction** ("we just need to make the
hand and joystick touch... if we need to modify the IK to look that way
then fine"): keep the joystick fully upright (no quat anywhere), and
correct its **position** (not orientation) so the ball actually reaches the
hand's real closest-approach point. Computed by: zeroing `x_slide`/`y_slide`
to find the frame where the reach naturally comes closest to
`joystick_base`'s own rest anchor (`CFL_35_20240128_trial_0001`, frame 107,
`Phalanx_hand_3_4_right`), then setting `joystick_base`'s position so the
ball (at local `(0,0,0.016184)` above `joystick_base` when upright) lands
at that fingertip's position. Net translation from the stac-mjx-derived
value: `(+5.6, +7.5, -2.9)mm`.

**Current (final) values in `contacts_patched.xml`**:

```xml
<body name="joystick_base" pos="-0.00702773 -0.02465269 0.04451328">
  <joint name="x_slide" .../>
  <joint name="y_slide" .../>
  <geom name="circular_base" .../>
  <body name="joystick" pos="0 0 0">
    <geom name="joystick_geom" type="cylinder" size="0.001 0.0071815" .../>
    <geom name="joystick_ball" type="sphere" size="0.002" pos="0 0 0.016184" .../>
  </body>
</body>
```

**Verified against the real, native v22 STAC data**
(`CFL_35_20240128_trial_0001`): min fingertip-to-ball-center distance
3.21mm (ball radius 2mm, so ~1.2mm surface gap) at frame 102, using the
trial's actual recorded `x_slide`/`y_slide` (not zeroed). Rendered and
visually confirmed genuine hand-joystick contact (fingers wrapped around
the ball) from two camera angles.

**Caveat this leaves open**: this position was derived from one trial's
geometry (`CFL_35_20240128_trial_0001`), not validated against all 5
available trials (let alone the 10 CFL_36/37 trials still pending on the
stac-mjx side) -- other trials may need their own small correction, or this
one value may not generalize once more data lands. Re-check against the
full trial set once CFL_36/37 finish. This is a pragmatic training-time fix
prioritizing actual contact over a "measured" pose that didn't hold up to
visual/structural scrutiny; it is not claimed to be the true physical
mounting position.


---

## Refined once more (2026-07-16, same evening): tightened to exact contact

Eric's ask: the closest-approach frame should be within ~2mm (essentially
touching), not just "close." The first position fix (translating from the
zeroed-slide rest anchor) landed at 3.21mm minimum -- close but not quite.
Refined by computing the residual delta at the actual closest frame (102,
using the trial's real recorded `x_slide`/`y_slide`, not zeroed) between the
ball and the closest fingertip, and shifting `joystick_base` by that exact
residual.

**Final value**: `joystick_base` `pos="-0.00447064 -0.02640225 0.04534320"`
(still no quat anywhere -- upright, base flush to x,y plane, per Eric).

**Verified**: min fingertip-to-ball distance = **0.00mm** at frame 102
(`CFL_35_20240128_trial_0001`), max 3.02mm across the full 126-frame trial.
Rendered from two close-up angles: fingers visibly wrapped around and
touching the ball's surface, unambiguous contact.

Note this only moved the joystick's fixed *anchor* position; `x_slide`/
`y_slide` values themselves are untouched, straight from the STAC fit, so
the joystick's motion pattern over time is exactly as recorded -- only
where that motion is anchored in world space changed.

Same caveat as before applies: derived from one trial only
(`CFL_35_20240128_trial_0001`); not yet checked against the other 4
available CFL_35 trials or the pending CFL_36/37 data.
