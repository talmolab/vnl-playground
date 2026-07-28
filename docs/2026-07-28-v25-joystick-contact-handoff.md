# v25 hand–joystick contact: handoff (2026-07-28)

Handoff for a collaborator picking up the **v25 arm+hand+joystick imitation**
task. Focus is the hand↔joystick contact problem and the physics fixes made
this session. (Earlier v22–v24 history is in the other `docs/*.md` files;
you don't need it to work on v25.)

## What v25 is

A right mouse forelimb (arm + hand + neck/head) driven by ~50 muscles,
imitating STAC-fit reference clips while its hand grips and moves a
small 2-DOF sliding **joystick**. The policy is trained with brax PPO to
track the reference kinematics *and* physically manipulate the joystick.

## Paths you need

| What | Path |
|---|---|
| Repo | `/root/vast/eric/vnl-playground` (branch **`eric/new-janelia`**) |
| v25 walker XML | `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_janelia_v25_arm_hand_joystick.xml` |
| v25 env + config | `vnl_playground/tasks/mouse/imitation_arm_hand.py` (`MouseImitationArmHand`, `default_config_v25()`) |
| Training entrypoint | `vnl_playground/train_mouse_janelia_arm_hand.py` (`--v25`) |
| v25 STAC reference data | `/root/vast/eric/stac-mjx/refined_STACed_data_v25/` (6 `*_ik.h5` clips, 126 frames each) |
| Upstream model authoring (Blender → MuJoCo) | `/root/vast/eric/janelia_model/` (`.blend` files, `PROTOCOL_blender_to_mujoco_v2.md`) |
| Python env (use THIS, not repo `.venv`) | `/root/vast/eric/track-mjx/.venv/bin/python` |
| Wandb project | `new-janelia` |
| Checkpoints | `checkpoints/<exp_name>/` |

## Run training

```bash
cd /root/vast/eric/vnl-playground
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  /root/vast/eric/track-mjx/.venv/bin/python \
  vnl_playground/train_mouse_janelia_arm_hand.py --v25 --tag <your-tag> \
  --eval-every 5000000
```

Always pin `CUDA_VISIBLE_DEVICES` when >1 job runs (unpinned concurrent
brax/JAX jobs crash on XLA all-reduce rendezvous). v25 is slow (~30 min per
5M-step eval) — contact-heavy, `iterations=30` Newton. A full 500M run is
multiple days. Useful override flags: `--force-scale`, `--iterations`,
`--ls-iterations`, `--joint-stiffness`, `--episode-length` (see `-h`).

## The contact problem and the current strategy

**Symptom:** hand contact geoms penetrate the joystick and the trained policy
touches but never moves the joystick to the reference position (`joystick_pos`
reward stayed flat across a 94M-step run).

**Root cause (measured, all 6 clips replayed offline):**
1. Penetration is **baked into the reference itself** — replaying the STAC-fit
   qpos puts the hand inside the joystick on **69% of frames, up to 2.56mm**.
   It is **entirely against `joystick_ball`** (the top sphere); the shaft and
   base have healthy clearance.
2. The joystick was **~4× heavier than the whole moving forelimb**
   (`density=40000` → 2.0g vs forelimb 0.53g), so the hand physically could
   not move it.
3. The reference grip is **loose/one-sided** — even at the exact reference
   pose the hand touched the ball only ~50% of frames, so contact (and the
   push it enables) was intermittent.

**Chosen direction (Eric):** emergent physical push — the hand should genuinely
drive the joystick via contact — with the **reference treated as fixed** (no
STAC re-fit). So the fix reconciles the collision model + joystick dynamics
with the fixed, slightly-penetrating reference.

**Fixes applied this session (in the v25 XML):**
- `joystick_geom` `density` **40000 → 4000** (joystick mass **2.0g → 0.23g**;
  now movable by the arm).
- The 5 hand grip geoms enlarged **1.5×** (`Metacarpal_hand_3` 1.8→2.7mm, the
  4 fingertips scaled to match) → contact is now **consistent** (offline
  `ncon` rose from 0–0.7 to 0.5–2.3) so the policy always has a handle.
- `gap="0.001"` (1mm) added to all 3 joystick geoms → the reference grip sits
  at ~zero contact force (no fighting the tracking reward); only firmer presses
  generate force to drive the joystick. Also damps twitchiness.
- Kept high friction (`4 0.05 0.05`, `condim=4`) — needed so the grip can
  *drag* the joystick laterally, not just push radially.

**Validated:** the edited XML loads in the real env and runs **0 reward-NaN
over 1512 harsh random-action steps** (6 clips × 252). Joystick DOFs stable.

## Honest limitations / open items

- **The reference isn't a physically-consistent push** (STAC fit hand markers
  and joystick pose independently) and the grip is only ~half-contact by
  nature, so precise emergent tracking has a real ceiling. The physics now
  *permits* the push; whether the closed-loop policy learns the servoing is
  the open question — **the real validation is a short training probe** (watch
  whether `joystick_pos` finally moves off flat). If it still won't move,
  the next lever is a hybrid weak position-actuator assist on `x_slide`/
  `y_slide` (Eric declined pure-kinematic, but a light assist is a middle
  ground).
- **`contact_threshold` needs recalibration.** With the 1.5× larger grip geoms
  the reference-side clearance shifted, so the `joystick_contact` reward's hard
  gate (`contact_threshold=0.001` in `default_config_v25()`) now fires ~96% of
  frames instead of the ~66% it was calibrated for — effectively almost
  ungated. Either accept it (near-continuous grip is expected now) or lower the
  threshold to restore discrimination.
- **`joystick_ball` is still 2.0mm.** We enlarged the fingers instead of
  shrinking the ball; the ball size is a further knob if penetration looks bad
  in renders.
- **wandb x-axis** still shows eval-count, not env timesteps — small logging
  fix noted in `docs/2026-07-20-...-handoff.md` ("log real progress to wandb"),
  never applied.

## Reproduce the diagnosis

Offline analysis scripts live in the session scratchpad (not committed);
the method: load the XML with `mujoco`, map each clip's `*_ik.h5` `qpos` onto
the model joints, `mj_forward` per frame, and compute surface clearance
(`geom_xpos` distance − summed radii) for the 5 grip × 3 joystick geom pairs.
A kinematic "push harness" drives the hand along the reference while leaving
the joystick fully dynamic to test whether contact moves it (note: open-loop,
so it can't prove trackability — a trained policy closes the loop).
