# v24/v25 buildout, bugs fixed, and the full training run

Written 2026-07-17, covering the session after the v22 joystick/camera fix
(see `2026-07-17-v22-joystick-camera-fix-and-first-successful-eval.md`).
Eric stepped away for ~5 hours; this documents everything done in that
window so it's reviewable on return.

## TL;DR

Built out v24 (a structurally different, newer mouse arm+hand+neck+head
model, no joystick) end-to-end: muscle-only XML, task config, training
script, camera. Found and fixed three real bugs along the way (see below).
Eric separately built v25 (v24 + a placeholder joystick, untargeted) himself
while I was mid-fix on one of the bugs. Both configs pass stability smoke
tests. A full 1B-step training run for v24 is in progress as of this
writing: `checkpoints/janelia-v24-arm-hand-20260717-103129-v24-full-run`,
`--eval-every 50000000`, muscle `force=".5"`, corrected camera.

## What v24 actually is

`/root/vast/eric/janelia_model/v24/forearm_v24.xml`: a much bigger rebuild
than v22 — full spine (T1-T13, C1-C7), skull, mandible, ribs all present as
rigid (njnt=0) geometry down to a single fixed root body `Armature`. Only
25 joints actually move, all anatomically named
(`shoulder_protraction/internal_rotation/abduction`, `elbow_flexion`,
`forearm_supination`, `wrist_deviation/flexion`, per-digit
`mcp/pip/dip/tip_flexion`, plus `neck_flexion`/`head_flexion`) — no
shoulder-translation slides like v22, so none of `imitation_arm_hand.py`'s
IK-snap/multi-root machinery applies. Wrist-equivalent body is
`N_L_C_right` (a fused carpal complex), not a body literally named
"wrist" — confirmed against `register_v24_mocap.py`'s
`KEYPOINT_MODEL_PAIRS`. No joystick body/geoms anywhere in the raw model.

## New files

- `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_v24_muscle_only.xml`
  — muscle-only copy of `forearm_v24.xml`. The source ships 4 actuator
  groups (position/velocity/motor + 52 muscles, nu=127 total); `base.py`'s
  `action_size` is just `mjx_model.nu` with no group filtering, so using the
  source file as-is would let the policy drive direct joint torques *and*
  muscles simultaneously. This copy strips position/velocity/motor,
  keeping only the 52 muscle actuators (nu=52). Eric later hand-edited the
  `<default><muscle force="...">` from `1` to `.5` directly in this file to
  calm down the trained policy's motion.
- `vnl_playground/tasks/mouse/imitation_v24.py` — `default_config()` (plain
  `MouseImitation`, single root `Armature`, `end_effector="N_L_C_right"`,
  `tracked_bodies` mirroring the 17 STAC keypoints) and
  `default_config_v25()`, which originally raised `NotImplementedError`
  (v25 was blocked on a v24-with-joystick STAC fit that didn't exist yet —
  see below, this got superseded by Eric's own v25 work in
  `imitation_arm_hand.py` instead).
- `vnl_playground/train_mouse_janelia_v24.py` — training script mirroring
  `train_mouse_janelia_arm_hand.py`'s structure, `--num-envs` default 2048
  (not v22's 4096, per Eric: smaller for faster smoke-test iteration on an
  unvalidated model).
- `scripts/smoke_test_v24.py` — lightweight rollout/NaN check, no PPO
  training, for fast iteration.

## Bugs found and fixed this session

1. **`reference_clips.py` stacked before filtering.** `keep_clips_idx`
   was applied *after* `np.stack`-ing every clip, so a data directory with
   mixed-length clips (v24's STAC fit was still running: some trials at
   the full 126 frames, others 15-frame in-progress stubs) raised a
   shape-mismatch error before filtering ever got a chance to drop the
   incomplete ones. Fixed by filtering the file list before loading.
   Doesn't change v22/v22x/v23 behavior (their directories always had
   uniform-length clips).
2. **Integrator/solver silently overridden during attach**, same category
   of bug that made v22 unstable before it was fixed there. v24's own
   `<option integrator="implicitfast">` was losing to the arena's `RK4`
   during the attach-conflict resolution (`base.py` only re-applies
   `cfg.integrator`/`cfg.solver` when explicitly set, and neither was set
   for v24 initially). Produced NaN on the very first random-action step.
   Fixed: `imitation_v24.py`'s `default_config()` now explicitly sets
   `cfg.integrator = "euler"`, `cfg.solver = "newton"` (same combo as v22).
3. **`train_mouse_janelia_arm_hand.py`'s `--eval-every` never applied.**
   `argparse` defined the flag but no code ever wrote it to
   `ppo_params.eval_every` — every run all session (including the earlier
   v22 1B-step run) silently used the hardcoded default of 100,000,000
   regardless of what `--eval-every` was passed. This is why that run's
   first checkpoint took far longer to appear than expected. Fixed with a
   one-line override, same pattern already present (correctly) in
   `train_mouse_janelia_v24.py`.
4. **(My own bug, not pre-existing)** `scripts/smoke_test_v24.py` initially
   called `env.reset`/`env.step` without `jax.jit(...)`, unlike every other
   script in this codebase. Every one of 300 python-loop iterations
   re-traced and recompiled the entire environment from scratch — burned
   2.5 wall-hours and 53GB RAM at 0% GPU utilization before being caught
   and killed. Fixed by jitting both calls, matching the rest of the
   codebase; the actual training script (`train_mouse_janelia_v24.py`) was
   never affected, since it already jits correctly (copied from
   `train_mouse_janelia_arm_hand.py`).

## Camera

The first inherited (v22-derived) camera in `train_mouse_janelia_v24.py`
was wrong for v24 in two ways Eric caught immediately from the rendered
eval video: framed the wrong lateral side (couldn't see the arm at all),
and the centroid/azimuth/elevation were v22-specific magic numbers that
don't transfer to a structurally different model. Root cause of "can't see
the arm at all": v24 has giant group-1 ellipsoid collision proxies for the
torso/skull (`T13_col`, `Skull_col`, etc., each several centimeters across
— bigger than the entire arm) that fully occlude it when rendered with
default visualization settings.

Fix, derived empirically (render, look, adjust — not guessed):
- Hide group-1 (collision) geoms via `MjvOption.geomgroup[1] = 0`, showing
  only the real anatomical meshes (group 0).
- Camera centroid/span computed over arm+hand bodies only (not the whole
  skeleton) — `humerus_right`, `ulna_right`, `radius_right`, `N_L_C_right`,
  all `Metacarpal_hand_*`/`Phalanx_hand_*`.
- Azimuth swept through 0/45/.../315 and visually compared; `azimuth=0`
  (with v24's own `mjv_defaultFreeCamera`-derived `elevation=-45`) is the
  one that actually faces the right arm with hand/fingers and muscle
  tendons in frame.
- `distance = span * 2.2` — empirically chosen to fit the whole arm+hand
  without clipping while still filling most of the frame.
- Wireframe rendering enabled (`renderer.scene.flags[mjRND_WIREFRAME] = 1`)
  so muscle/tendon paths read clearly against the bone meshes, per Eric's
  request.

Confirmed end-to-end against the actual training script's own
`build_render_model()` output (not just a standalone test) before trusting
it for the full run.

## v25 (built by Eric directly, not by me)

While I was fixing bug #2 above, Eric built out v25 himself in
`imitation_arm_hand.py`/`consts.py`:
`mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml` — v24's arm+hand
chain plus a new capsule-shaft joystick, physically present but untargeted
(no STAC-fit joystick trajectory to imitate yet, so no joystick tracking
reward). `default_config_v25()` in `imitation_arm_hand.py` (multi-root
`("Armature", "joystick_base")`, same `integrator="euler"`/`solver="newton"`
combo, `ctrl_dt=0.02`/`sim_dt=0.001`). Smoke-tested clean: 0 NaN over 300
random-action steps, `ncon` steady at 123 (persistent light contact, not
fluctuating instability). One thing flagged but not yet fixed: this file's
`<option iterations="30" ls_iterations="30">` (a deliberate margin bump)
is silently overridden back to the arena's `iterations="6"` during attach,
same "parent wins" conflict as the integrator bug — didn't matter since
the smoke test was already clean, but if NaN ever reappears here, know that
bumping iterations in the walker XML alone won't take effect without also
patching it through post-attach the way `solver`/`integrator` already are.

## Muscle force reduction

Eric reduced `<default><muscle force="1">` to `force=".5"` directly in
`mouse_forelimb_right_janelia_v24_muscle_only.xml` after watching the
untrained (checkpoint-0) policy move very forcefully/chaotically —
scales down all 52 muscles' max force uniformly. Verified via
`model.actuator_gainprm` that the change took effect (0.75 = default MuJoCo
muscle gain scaling at force=0.5, consistent). Confirmed via a fresh smoke
test that motion looks calmer at checkpoint 0 with the reduced force.

## Current full run

```
python vnl_playground/train_mouse_janelia_v24.py \
  --tag v24-full-run --no-wandb --num-timesteps 1000000000 --eval-every 50000000
```
Checkpoint dir: `checkpoints/janelia-v24-arm-hand-20260717-103129-v24-full-run`
Confirmed healthy shortly after launch: process alive, GPU at 99%
utilization, memory stable (~4GB, not the 53GB runaway pattern from bug
#4 above). `--no-wandb`, so progress is only visible via the checkpoint
directory's eval videos/weights, not a live dashboard.

## Not done / open items

- v25 has no real joystick-tracking reward yet (blocked on a real
  v24-with-joystick STAC fit — see consts.py's comment above
  `MOUSE_REFERENCE_DATA_JANELIA_V25_PATH`).
- v25's `iterations`/`ls_iterations` bump not actually taking effect (see
  above) — not urgent since it's stable regardless, but worth patching the
  same way integrator/solver are if NaN ever appears.
- No full training run has been launched for v25 yet, only the lightweight
  smoke test.
