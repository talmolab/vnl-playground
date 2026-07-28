# v25 training fixes (2026-07-19/20) and next task: joystick-contact reward

Written 2026-07-20. Covers a debugging session that took v25
(`train_mouse_janelia_arm_hand.py --v25`) from "NaNs at step 0" to a stable,
correctly-tracking overnight run, plus a scoped next task for whoever picks
this up next.

## TL;DR

v25 (v24's arm+hand+neck+head rig + a joystick, see
`docs/2026-07-17-v24-v25-buildout-and-full-run.md`) NaN'd on its first real
training attempt. Five separate bugs were found and fixed, in this order:

1. Solver `iterations`/`ls_iterations` silently reset to 6 instead of the
   XML's intended 30.
2. Muscles at full force (never got the same reduction v24 already needed).
3. Ghost/IK video overlay rendered wrong (camera + hidden-geom-group setup
   never ported from `train_mouse_janelia_v24.py`).
4. `joystick_base`'s anchor position was a ~16.7mm-off placeholder, not
   stac-mjx's real calibrated value.
5. The joystick's passive spring rest point (`springref`, unset -> 0) fought
   against where the hand actually needs to hold it, and grip friction was
   at MuJoCo's low default.

All five are fixed. The run training right now is stable
(`nan_termination: 0.0` every eval) and the ghost/IK overlay now visibly
matches the real sim.

**Update 2026-07-20 (later): the joystick-contact reward term is now
implemented** (approach A, hard-gated) — see the new section near the
bottom, "Joystick-contact reward: implemented", which replaces the old
"Next task" placeholder below. A new run with this reward enabled has been
launched (see "Current run").

## Current run

**Superseded 2026-07-20 (later)** by a new run with the joystick-contact
reward enabled:

- Experiment: `janelia-v25-arm-hand-joystick-20260720-052243-v25-joystick-contact-reward`
- Command: `PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python vnl_playground/train_mouse_janelia_arm_hand.py --v25 --tag v25-joystick-contact-reward --eval-every 5000000`
- Wandb: project `new-janelia`, same run name
- Checkpoint dir: `checkpoints/janelia-v25-arm-hand-joystick-20260720-052243-v25-joystick-contact-reward`
- Log: `logs/janelia_v25_joystick_contact_reward.log`
- No CLI flag needed for the new reward -- it's baked into
  `default_config_v25()`'s `reward_terms`, same way `bodies_pos`/`wrist_pos`
  are.

The prior run this superseded (kept for reference):

- Experiment: `janelia-v25-arm-hand-joystick-20260720-021037-v25-force-half-overnight`
- Command: `python vnl_playground/train_mouse_janelia_arm_hand.py --v25 --tag v25-force-half-overnight --eval-every 5000000`
  (launched with `CUDA_VISIBLE_DEVICES=0` — see gotcha below)
- Wandb: project `new-janelia`, same run name
- Checkpoint dir: `checkpoints/janelia-v25-arm-hand-joystick-20260720-021037-v25-force-half-overnight`
- Status as of superseding: training cleanly, `nan_termination: 0.0` at
  every eval so far. **Slow**: ~30-36 min per 5M-step eval interval (vs.
  ~4 min for the old v22x baseline) — inherent to v25's contact complexity
  (30 active contact pairs, `iterations=30` Newton), with the friction/
  `condim=4` bump from fix #5 adding a further ~20-30% on top. A full
  500M-step run at this rate is multiple days, not overnight. Worth
  profiling/trimming if that matters before committing to a long run.
  Same slowness applies to the new run above -- the contact-distance reward
  computation itself is cheap (13 geom lookups + a 10x3 pairwise distance
  matrix per step) next to the physics step cost that already dominates.

## The five bugs, in the order they were found and fixed

### 1. `iterations`/`ls_iterations` passthrough (imitation_arm_hand.py)

The XML has `<option iterations="30" ls_iterations="30"/>` as a deliberate
convergence-margin bump for the joystick/fingertip contacts, but
`MouseBaseEnv.compile()` in `base.py` unconditionally re-applies
`self._config.iterations`/`ls_iterations` after attach (base default: 6/6) —
same "parent wins" bug class that already hit v22's integrator/solver.
`default_config_v25()` never set these fields, so they silently fell back
to 6/6. Fixed by adding `cfg.iterations = 30; cfg.ls_iterations = 30` to
`default_config_v25()`, mirroring how `solver`/`integrator` are already set
there explicitly.

**General lesson**: any new `base.py` config field needs an explicit
per-variant override in each `default_config_*()` if the walker XML's own
`<option>` value should actually take effect — it is never picked up from
the XML automatically.

This alone took step-0 `nan_termination` from 84% down to ~54% — better,
not fixed. Diagnosis of the *remaining* NaN (see next item) required
comparing three regimes: independent per-step `Uniform(-1,1)` random
actions over full 252-step episodes (never NaN'd, even at the tightest
hand-joystick configurations); a real untrained (tanh-normal) PPO policy
(NaN'd ~50-54%); and `iterations=50` instead of 30 with force still at 1
(still ~52% at step 0, only dropping under 1% *after* 1M steps of real
training). Conclusion: never a solver-convergence problem — an untrained
policy's sustained, correlated, near-saturating actions on full-strength
muscles, which independent random noise doesn't reproduce.

### 2. Muscle force (XML `<default><muscle force="...">`)

v25 was still at `force="1"` (full strength) — v24 already needed
`force=".5"` after Eric observed "very forceful/chaotic" untrained-policy
motion on the same underlying muscle model, but that fix was never ported
to v25's own XML copy. Changed to `force=".5"`. This is what actually took
`nan_termination` to 0.0 (confirmed via `--force-scale 0.5` CLI test before
committing to the XML edit).

### 3. Ghost/IK video rendering (train_mouse_janelia_arm_hand.py)

The eval video's ghost (reference) joystick appeared to render inside the
torso. **Not a kinematics-data bug** — Eric was right that the raw STAC
qpos is fine (checked directly: `x_slide`/`y_slide` bounded within the
mechanical ±6mm limit across all 6 clips) and it is **not** fixed by
`recompute_kinematics` (that flag only touches `MouseReferenceClips`'s
*stored* `xpos`/`xquat` array — the ghost render path in
`train_mouse_janelia_arm_hand.py`'s `policy_params_fn` always does its own
fresh `mujoco.mj_forward()` from raw `ref.qpos`, completely bypassing that
flag). Still set `recompute_kinematics = False` per Eric's read that the
raw data needs no correction — harmless either way for the ghost render,
and "joystick" isn't in `_TRACKED_BODIES_V25` so it doesn't affect reward
either.

The real cause: `build_render_model`'s `__main__` block in
`train_mouse_janelia_arm_hand.py` never got the fixes
`train_mouse_janelia_v24.py` already has for this same shared v24-derived
skeleton — it didn't hide group-1 collision-proxy geoms (giant
`T13_col`/`Skull_col`-style ellipsoids, several cm across, fully occlude
the arm/hand/joystick), and its no-named-camera fallback (v25 has no
named camera in its XML, unlike v22/v23/raw-joystick) still used the old
pre-v24-fix derivation (azimuth=130/elevation=-25/all-body centroid),
aiming through the spine. Fixed: ported `MjvOption.geomgroup[1] = 0` into
the scene option (passed into `renderer.update_scene()`), and rederived the
camera from arm+hand+joystick body centroid/span only, `azimuth=90`
(not v24's 0 — v25's joystick sits laterally off the hand; swept and
checked visually, this is the angle that shows real-vs-ghost joystick side
by side instead of one behind the other), `elevation=-45`.

### 4. `joystick_base` anchor (~16.7mm off) — the real positional bug

Confirmed by Eric via `docs/2026-07-19-v25-joystick-base-anchor-mismatch.md`
(that file has the full derivation): stac-mjx's
`compute_joystick_geometry_v25.py` computes `joystick_base`'s anchor from a
real static-calibration measurement of the physical joystick rig and bakes
it into the model it actually fits reference clips against
(`mouse_forelimb_right_janelia_arm_hand_v25_stac.xml`). vnl-playground's own
copy (`mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml`) had a
placeholder anchor Eric built by hand on 2026-07-17, before that
calibration script existed, never synced afterward. The two anchors were
~16.7mm apart — nearly 3x the joystick's own ±6mm travel range, so the
reference data was asking the sim joystick to be somewhere it could
physically never reach.

Fixed: `joystick_base` `pos` changed from `0.00029639 -0.02926438 0.05621514`
to the calibrated `-0.01114158 -0.02948650 0.04407920`. This is what
actually fixed the ghost/real joystick misalignment fix #3's camera work
had merely made *visible* rather than causing.

Verified after: contact-distance scan shows up to ~2mm of interpenetration
at some clip/frames (expected — the placeholder anchor had been keeping the
joystick artificially far from the hand, so real contact never actually
engaged before), but a 252-step random-action rollout at every
worst-overlap clip/frame still shows zero NaN, and a fresh render shows
real and ghost joystick nearly coincident.

### 5. Spring rest point and grip friction

Once the anchor was correct, Eric noticed the joystick's passive
equilibrium point (`x_slide`/`y_slide` had no `springref`, defaulting to 0 —
the center of the ±6mm range) was fighting the grasp: computed directly
from all 756 reference frames across the 6 clips, `x_slide`/`y_slide` sit
almost entirely between +0.0018 and the +0.006 limit (mean
`0.004997`/`0.004773`), never anywhere near 0. The spring was constantly
pulling the joystick away from where the hand actually needs to hold it.
Fixed: added `springref="0.004997"` / `springref="0.004773"` to the two
joints.

Also bumped grip friction: every joystick/hand contact geom (3 joystick
geoms, 10 hand grip spheres) was at MuJoCo's bare default
(`friction="1 0.005 0.0001"`, `condim=3` — no override anywhere in the XML).
Changed to `friction="2 0.01 0.01" condim="4"` on all 13 geoms, for a
firmer, torsion-resistant hold.

Verified no NaN regression at the worst-overlap clip/frames after both
changes, then relaunched the run described above.

## Files touched this session

- `vnl_playground/tasks/mouse/imitation_arm_hand.py` — `default_config_v25()`:
  added `iterations`/`ls_iterations`, flipped `recompute_kinematics` to
  `False`.
- `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml` —
  muscle `force` `1` -> `.5`; `joystick_base` `pos` corrected; `springref`
  added to `x_slide`/`y_slide`; `friction`/`condim` bumped on the 13
  joystick/hand grip geoms.
- `vnl_playground/train_mouse_janelia_arm_hand.py` — added
  `--iterations`/`--ls-iterations` CLI overrides (mirrors the existing
  `--force-scale` pattern); rewrote the no-named-camera render fallback
  (group-1 hide, arm+hand+joystick centroid, azimuth=90); passes
  `scene_option` into `renderer.update_scene()` (previously omitted
  entirely).

## Infra gotchas hit this session (not new, but bit us again)

- **Pin `CUDA_VISIBLE_DEVICES` for concurrent jobs.** Launching two
  brax/JAX training processes at once without pinning caused an XLA
  all-reduce rendezvous crash (one un-pinned process tried to grab both
  GPUs while the other's pinned process had the second GPU saturated).
- **Don't import `train_mouse_janelia_arm_hand` as a module** to reuse
  `build_render_model` in a quick test script — importing it executes ALL
  of its top-level code, including `wandb.init()`. Spawned several
  throwaway `janelia-v22-arm-hand-*` runs in the `new-janelia` wandb
  project as a side effect during this session's debugging (harmless — no
  real training happens, the process exits right after the test script
  finishes — but clutters wandb). Copy the needed logic into the test
  script instead.
- Real training runs need `/root/vast/eric/track-mjx/.venv/bin/python`, not
  this repo's own `.venv` (incompatible jax/brax versions there — see
  `docs/... v22 training bring-up` context / memory for the full story).

## Joystick-contact reward: implemented (2026-07-20, later)

Implemented as approach A below (distance-proxy), **hard-gated** per Eric's
call on the open design question at the bottom of this section. Lives in
`imitation_arm_hand.py`, registered as `"joystick_contact"` and enabled only
in `default_config_v25()`'s `reward_terms` (v22/v22x/v23 don't get it — v22x
in particular has no joystick body at all).

**Geometry**: the 30 real contact pairs are 10 hand grip geoms (5
metacarpal + 5 digit-tip spheres, `_GRIP_BODIES_V25`) x 3 joystick geoms
(`circular_base`, `joystick_geom` shaft, `joystick_ball`,
`_JOYSTICK_GEOMS_V25`) — same set `default_config_v25()`'s own docstring
already identifies as the only colliding pairs.

**Reference side** (the gate): `reference_clips` only stores body-level
`xpos`/`xquat`, not per-geom, so each geom's reference world position is
reconstructed as `body_xpos + brax.math.rotate(local_geom_offset,
body_xquat)` — `brax.math.rotate`/`inv_rotate` are already used this way in
rodent/fruitfly/celegans/stick `imitation.py`, just newly imported here.
This matters more than it sounds: the joystick geoms sit 7-16mm off their
body's own origin (shaft/ball offset along local z), so a naive
body-xpos-to-body-xpos distance (skipping the geom offset) never gets
anywhere near a real contact distance — confirmed empirically, raw
body-to-body distance stayed at 13-20mm across every reference frame in a
test clip, nowhere near "touching."

**Sim side**: read directly from `data.geom_xpos` (already computed by
`mj_forward`/`mjx_env.step` every physics step) — no manual reconstruction
needed, only the reference side needs it.

**Distance metric**: surface-to-surface clearance (center-to-center
distance minus both geoms' radii — `geom_size[gid][0]`, treating the
capsule shaft as a sphere of its own radius, a coarse approximation judged
fine for a proxy reward), minimum over all 30 pairs. Center-to-center alone
would be off by the same 7-16mm offset problem above.

**Calibration** (2026-07-20, computed directly from all 756 reference
frames across the 6 v25 clips, not the "known geom sizes" estimate the
original ask suggested — see why below): surface clearance is already <=0
(interpenetrating — STAC-fit hand mesh actually overlapping the joystick
mesh) on 66% of frames and <=1mm on 82%. This means the real mouse is
gripping the joystick almost continuously in this dataset, not just at
sparse contact events — worth knowing before reading the reward as sparse.
Picked `contact_threshold=0.001` (1mm) for the hard gate, `exp_scale=0.002`,
`weight=0.1` (matching `bodies_pos`/`wrist_pos`'s starting weight, per the
original ask's own recommendation — no more principled starting point
exists yet, retune once training shows whether the policy is ignoring it).

**Design question resolved**: hard gate (reward is exactly 0 outside the
reference-contact window), not a soft `exp(-dist/scale)` weighting — Eric's
call, 2026-07-20.

**Verified** (before launching the run in "Current run" above): instantiated
`MouseImitationArmHand(default_config_v25())` directly, ran a 252-step
random-action rollout (`jax.jit`-compiled) on all 6 clips — zero
`nan_termination`, gate fires 66-93% of frames per clip (matches the
per-frame calibration stat above, not degenerately always-on or always-off),
reward correctly saturates toward the 0.1 weight cap when sim clearance is
small under lucky random actions.

### Where this plugs in

Reward terms are registered via the decorator pattern in
`vnl_playground/tasks/reward_registry.py` (`RewardRegistry.reward(name)`),
implemented as methods on the env class, and enabled via
`cfg.reward_terms[name] = {...params...}`. See `imitation.py`'s
`_bodies_pos_reward`/`_wrist_pos_reward` (lines ~400-427) for the exact
pattern: pull `target = self._get_current_target(data, info)` (or, for a
non-tracked body like the joystick, index `self.reference_clips.at(...)`
directly the way the ghost-render code does), compute a distance, log it
to `metrics`, return `weight * exp(-(distance/exp_scale)**2 / 2)`.
`imitation_arm_hand.py`'s own `_registry` (built by copying the parent's
`.rewards`/`.terminations` dicts then overriding, see the bottom of that
file) is where a v25-specific reward should live, since no other model
variant has a joystick contact to reward.

### Two implementation approaches to weigh

**A. Distance-proxy (matches existing style closely, cheap, robust).**
Reward is gated by whether the *reference* data says contact should be
happening (e.g., threshold on the distance between the reference
joystick's `xpos` and the reference hand's grip-point `xpos` — calibrate
the threshold from the known geom sizes: joystick shaft radius 1mm, ball
radius 2mm, hand grip spheres ~0.3-0.9mm, so "should be touching" is
roughly sum-of-radii plus a small margin), multiplied by
`exp(-(sim_distance/exp_scale)**2/2)` using the corresponding *simulated*
distance. This is exactly how `wrist_pos`/`bodies_pos` already work, just
gated by a reference-contact indicator instead of applied unconditionally.

**B. Literal contact-based (closer to what "making contact" means, more
involved).** The model has a fixed, already-verified set of exactly 30
possible contact pairs (10 hand grip spheres × 3 joystick geoms — every
other geom pair is either non-colliding or excluded via the
`contype`/`conaffinity` bitmask, see `default_config_v25()`'s docstring).
Reward could sum measured penetration/contact depth
(`jp.clip(-data.contact.dist, 0, None)`, gated by the reference-contact
indicator from approach A) restricted to those known geom-id pairs, read
out of `data.contact` (`data._impl.contact` in current mujoco — the API
emitted a `DeprecationWarning` on direct `.contact` access during this
session's debugging, worth checking which is correct at implementation
time). **Open question, verify before relying on this**: whether mjx's
per-step contact array orders/assigns these 30 pairs to stable indices
you can gather by known geom-id, or whether the active set/ordering can
shift step-to-step in a way that needs handling (e.g. via `geom1`/`geom2`
masks recomputed every step rather than assumed-fixed slot indices).

### Recommendation

Start with A — it's a strict continuation of the existing reward-function
style in this file, doesn't touch the mjx contact array at all, and is
easy to reason about/debug. Only reach for B if A's proxy turns out to be
too loose (e.g., rewards near-misses the same as real holds).

**Resolution (2026-07-20, later): went with A**, implemented as described
in "Joystick-contact reward: implemented" above. B's open question (mjx
per-step contact ordering stability) was never investigated since A turned
out sufficient — still worth investigating if A's proxy proves too loose in
practice.

## Also worth doing: log real progress to wandb

`train_mouse_janelia_arm_hand.py`'s `wandb_progress(num_steps, metrics)`
(around line 369) receives `num_steps` — brax's cumulative env-timestep
count — but only uses it for the console `pprint(f"Step {num_steps}")`; it
is never added to the dict passed to `wandb.log(metrics)`. That call bumps
wandb's own internal step counter by 1 per call (i.e. per eval), not by
real env timesteps, so the wandb dashboard's x-axis currently shows "eval
number", not actual training progress or an ETA against
`ppo_params.num_timesteps` (500M by default). Fix: add
`metrics["timesteps"] = num_steps` and
`metrics["total_timesteps"] = ppo_params.num_timesteps` (or equivalent
keys) before the `wandb.log(metrics)` call, so the dashboard can show
real progress (`timesteps / total_timesteps`) instead of just eval count.
Small, low-risk change — didn't apply it to the currently-running overnight
job to avoid a 4th relaunch tonight; pick it up on the next restart.

### Also worth deciding

- Should "should be in contact" be a hard gate (0/1, reward only applies
  when true) or a soft weighting (e.g., itself an `exp(-dist/scale)`
  term multiplying the sim-side reward)? A hard gate is simpler to reason
  about; a soft one avoids a discontinuity right at the threshold.
  **Decided 2026-07-20: hard gate** (Eric's call) — see "Joystick-contact
  reward: implemented" above.
- Reward weight/`exp_scale` will need empirical tuning the same way every
  other term in `cfg.reward_terms` was (see `default_config_v25()`) — no
  principled starting point exists yet, start small relative to
  `bodies_pos`/`wrist_pos` and increase if the policy ignores it.
  **Started at weight=0.1, exp_scale=0.002** — still unretuned, watch the
  run in "Current run" and adjust if the policy ignores this term.

### Still outstanding

The wandb real-progress-logging fix described above ("Also worth doing")
was not applied this session either — still pick it up on the next restart,
it's independent of the joystick-contact reward work.
