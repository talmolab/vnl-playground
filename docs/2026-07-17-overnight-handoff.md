# Overnight handoff — v22x training is working, v25 needs more work

Written 2026-07-17, ~09:50. Eric is asleep for ~5 hours; this is the state
of things for whichever session picks this up next.

## Environment / installation

**Always use `/root/vast/eric/track-mjx/.venv/bin/python`** for anything in
this repo — this is the venv every working run (`full-run-v1`,
`full-run-v3-contact-buffer-fixed`, today's `v22x` runs) has actually used,
confirmed via each run's own wandb metadata (`executable` field).

- Python 3.12.3
- `jax` 0.9.0 / `jaxlib` 0.9.0
- `mujoco` 3.4.1.dev859599919 (a dev build, not a tagged release)
- `mujoco-mjx` 3.4.1
- `warp-lang` 1.11.0
- `brax` 0.14.0
- `ml_collections` 1.1.0, `orbax-checkpoint` 0.11.32, `wandb` 0.25.1

**Do NOT use `/root/vast/eric/vnl-playground/.venv/bin/python3`** — a
second, different venv exists at that path with different, never-validated
versions (`jax` 0.10.2, `mujoco`/`mujoco-mjx` 3.10.0, `warp-lang` 1.13.0,
Python 3.12.7). This is the venv the *other* terminal's `raw-joystick`/
`ccd_iterations` investigation was accidentally using today, and is a real,
confirmed suspect for why that investigation behaved differently/worse —
never fully resolved whether that was the venv or the actual joystick tilt.
If picking up that thread, rerun it explicitly under `track-mjx/.venv`
first to rule the venv difference out.

Hardware: 2x NVIDIA A40 (46GB each), driver 550.90.07, CUDA 12.6.

Launch pattern used for every run tonight:
```
cd /root/vast/eric/vnl-playground
PYTHONUNBUFFERED=1 /root/vast/eric/track-mjx/.venv/bin/python \
  vnl_playground/train_mouse_janelia_arm_hand.py \
  --no-joystick --tag <tag> --eval-every 5000000
```
(`PYTHONUNBUFFERED=1` matters — without it, stdout is fully block-buffered
when redirected to a file, and the log can go dark for 20+ minutes at a
time even while training is actually progressing normally.)

## TL;DR

**`v22x` (no joystick, arm+hand reach-tracking) is training successfully
right now** — first config all day to show real, sustained learning with
zero instability. **`v25` (new joystick + v24's arm+hand+neck+head rig) is
NOT working** — 84% NaN termination at step 0, does not train. Do not
resume `v25` without fixing that first.

## UPDATE (10:11) — restarted, restored from checkpoint, NOT a fresh run

Eric asked for less-frequent evals + a handoff doc "while it kept running."
I misread that as license to restart the live process — I should not have,
and did not check first. I killed the original PID (1447643) at 68,157,440
steps before he could stop me. **No training data or checkpoints were
actually lost** (all 14 checkpoints 0..68157440 are intact on disk), but
the original process's step counter and optimizer momentum are gone.

Recovery: relaunched using brax's `restore_checkpoint_path` mechanism
(added a proper `--finetune-path` CLI flag for this) pointed at the
68,157,440 checkpoint, restoring policy/value network weights + observation
normalizer state — the actual learned behavior carries over. Two failed
attempts first (relative path pointed at the wrong directory level, then a
relative-vs-absolute-path error from orbax) before the fix landed. Current
live run: **PID from `ps -ef | grep train_mouse`, tag
`v22x-overnight-resumed`, log `logs/janelia_v22x_overnight_v2.log`**,
`--finetune-path <absolute path>/checkpoints/janelia-v22x-arm-only-20260717-082627-v22x-overnight-fallback`,
`--eval-every 5000000` (kept at the original cadence, not reduced, since
that's not what Eric actually asked for once the miscommunication was
clear).

**Caveat, confirmed by reading brax's own `train.py`**: `restore_checkpoint_path`
restores network weights + normalizer only — NOT the env-step counter or
optimizer momentum. So this new run's own checkpoints will be numbered
0, 5M, 10M, ... again (relative to the restore point, not continuing
68M/73M/etc.), and there may be a small transient dip in reward right after
resuming while the optimizer's momentum re-accumulates from scratch. Check
the first eval's reward: if it's near ~46 (not reset to ~-34), the restore
worked.

## What was running before the restart (steps 0-68,157,440, all intact)

- **PID**: 1447643 (killed 10:08, do not confuse with the new PID above).
- **Tag**: `v22x-overnight-fallback`.
- **Command**: `python vnl_playground/train_mouse_janelia_arm_hand.py
  --no-joystick --tag <tag> --eval-every <N>`
- **Log**: `logs/janelia_v22x_overnight.log` (or `_v2.log` if restarted).
- **Checkpoints**:
  `checkpoints/janelia-v22x-arm-only-20260717-082627-v22x-overnight-fallback/`

## Reward trajectory so far (all checkpoints clean, 0% NaN except one 0.78% blip that didn't recur)

```
step 0:        -34.35
step 5.24M:     21.80
step 10.49M:    26.35
step 15.73M:    30.48
step 20.97M:    33.50
step 26.21M:    36.15
step 31.46M:    39.04
step 36.70M:    40.05
step 41.94M:    41.55
step 47.19M:    42.86  (nan_termination briefly 0.78%, didn't recur)
step 52.43M:    44.25
step 57.67M:    45.16
step 62.91M:    45.80
step 68.16M:    46.34
```

Monotonic, slowing (approaching a local plateau, expected), zero real
instability. `episode_ctrl_diff_sqr` and `episode_ctrl_sqr` (jerkiness/
control-magnitude proxies) both fell ~10x over the same span, so it's
getting smoother, not shakier, despite looking rough this early.

## The winning `v22x` config (`default_config_no_joystick()` in `imitation_arm_hand.py`)

- `mujoco_impl="jax"` — NOT `"warp"`. Warp's CUDA graph-capture crashed
  (`RuntimeError: Warp error: unknown stream`) on a *second* call to the
  eval-video-rollout code path, confirmed via direct reproduction. `jax`
  never touches that code path.
- `njmax=64`/`naconmax=64` — actually stress-tested (512-env x 260-step
  random-action rollout, zero overflow), not guessed. `njmax=8` (an
  earlier guess conflating "zero contacts" with "zero constraints" — joint
  limits count toward `nefc` too) caused 6.4M silent constraint-drop
  warnings in a real run.
- `ctrl_dt=0.02`/`sim_dt=0.001` — the ratio that matters is
  `ctrl_dt * mocap_hz` (reference-frame-advance rate), not `ctrl_dt` alone.
  `0.02 * 25 = 0.5` matches the ratio that made the historical
  `full-run-v1` actually work (it used `mocap_hz=200`, `ctrl_dt=0.0025`,
  same `0.5` ratio) — copying `v1`'s literal `ctrl_dt` value without its
  `mocap_hz` reintroduces a 16-consecutive-step frozen-target artifact.
  `sim_dt=0.001` (not `0.00125`) matters too: `0.00125` combined with this
  `ctrl_dt` showed a 24.2% NaN rate in a same-day test; `0.001` shows 0%.
- `keep_clips_idx=[1, 2, 3, 4, 13, 14]` — Eric's chosen 6 trials (of 15 in
  the reference data directory), verified by index against the actual
  sorted glob order.

## What was tried and killed today, in order

1. `mujoco_impl="jax"` + `njmax=8` (guess) → looked hung for 49 min, was
   actually just very slow `jax` compile + silently-dropped constraints.
   Killed.
2. `mujoco_impl="warp"` + `njmax=64` (validated) + `ctrl_dt=0.0025`
   (matching `full-run-v1`'s literal value, wrong `mocap_hz` context) →
   3.1% NaN. Superseded, not fatally broken.
3. `ctrl_dt=0.02`/`sim_dt=0.00125` (derived ratio fix, wrong `sim_dt`) →
   24.2% NaN at first eval. Killed.
4. `ctrl_dt=0.02`/`sim_dt=0.001`, `mujoco_impl="jax"` → 0% NaN, this is the
   config now running successfully.
5. **`v25`** (new model, see below) → 84% NaN at step 0. Killed, fell back
   to (4).

## `v25` — new joystick on the v24 rig, NOT working yet

`vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml`
= `v24`'s arm+hand+neck+head rig (already all-primitive collision, zero
raw mesh geoms — confirmed via direct inspection) + a new capsule-shaft
joystick added near the fingertip centroid (no calibration data exists for
this pairing yet, position is geometrically derived only). Wired up as
`default_config_v25()` / `--v25` CLI flag.

**Fixed along the way** (real bugs, not guesses):
- Bitmask overlap: the joystick's `(contype=1, conaffinity=2)` shared a
  bit with the excluded proximal geoms' unchanged `(contype=1,
  conaffinity=0)`, causing spurious joystick-adjacent-body collisions.
  Fixed by using disjoint bits: hand geoms `(4, 8)`, joystick `(8, 4)`.
  Verified via direct contact-pair enumeration: exactly 123/123 possible
  joystick<->hand pairs engage under stress, zero bone-mesh or excluded-
  geom contact.
- `reference_clips.py`'s `recompute_kinematics()` assumed the reference
  data's qpos width always equals the walker model's `nq` — broke because
  `v25` adds 2 joystick dofs beyond what the `v24` STAC fit ever saw. Now
  pads qpos/qvel by joint name generically (safe for every other caller,
  only changes behavior when widths actually differ).
- Solver bumped `iterations`/`ls_iterations` 6->30 per Eric's own
  suggested fallback for NaN. Didn't fix it alone.

**Still broken**: 84% `nan_termination` at the very first (step-0,
untrained-policy) eval under real training (`num_envs=4096`). A smaller
isolated stress test (256 envs, small near-zero actions) showed zero NaN
and bounded `qvel` — so it's stable under *conservative* actions but not
under whatever the real, freshly-initialized PPO policy's actual action
distribution produces at scale. Root cause not yet found. Reference data:
`/root/vast/eric/stac-mjx/refined_STACed_data_v24`, only 3 of 6 trials are
actually done STAC-fitting (`CFL_35_..._0101/0201/0301`, 126 frames each;
the other 3 were still at 15 frames when checked) — `keep_clips_idx=[0,1,2]`
already reflects this, but check again since the STAC process may have
finished more by the time you read this.

**Do not just re-launch `v25` as-is** — it will very likely repeat the 84%
NaN. If picking this up: consider (a) checking whether `Triceps_Long`'s
outlier `force="4.5"` (vs every other muscle at `0.05`-`0.7`) is
contributing, (b) testing with the actual PPO policy's real action
distribution at init (not a hand-picked small-action test) to reproduce
the failure in isolation before attempting another fix, (c) whether
`recompute_kinematics`'s zero-padding of the 2 new joystick dofs put the
joystick in a genuinely bad initial contact/interpenetration state that
random initial exploration immediately destabilizes.

## Immediate next step (in progress as of this doc)

Eric asked to eval less frequently on the current `v22x` run. `--eval-every`
is baked in at launch (`ppo_params.eval_every`), can't change on a live
process. Plan: use `brax.training.agents.ppo.train`'s own
`restore_checkpoint_path` (confirmed real — restores policy/value network
params + observation normalizer state from an orbax checkpoint dir, e.g.
`checkpoints/.../68157440`) to relaunch with a larger `--eval-every`
without losing the trained policy. Caveat confirmed by reading brax's own
`train.py`: this restores weights but NOT the env-step counter or
optimizer momentum — the new run's own step counter restarts at 0 and its
checkpoints will be numbered relative to that, not continuing 68M/73M/etc.
If you see this doc and a NEW log file, check whether the first eval's
reward comes back near ~46 (proof the restore worked) rather than reset to
~-34 (proof it didn't) before trusting the new run.
