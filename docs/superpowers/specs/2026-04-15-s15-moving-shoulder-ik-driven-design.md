# s15: IK-Driven Moving-Shoulder Imitation (Design Spec)

**Date:** 2026-04-15
**Branch:** `eric/janelia`
**Author:** Eric + Claude
**Status:** Draft for user review

## Motivation

In the current fixed-shoulder janelia model (`mouse_forelimb_right.xml`), the shoulder base is rigidly pinned at the origin. The biological shoulder translates up to ~4 mm over a reach. Freezing it at the far end of that range means the hand starts farther from the target, and the triceps burst must start **earlier** (and rise more steeply) to cover the extra angular distance in the same time window. That time-shift corrupts the EMG-vs-model comparison — a shift we think accounts for the triceps onset mismatch we've been chasing.

Rather than adding three more actuators (and three more things for the muscle policy to co-solve), we **drive the shoulder translation kinematically from the STAC-mjx v16 IK solution** and let the policy learn only the muscle activations. This gives the arm the correct base position at every tick, so the triceps (and every other muscle) sees the same geometry the mouse did.

## Success Criteria

1. Training runs to convergence with the moving-shoulder XML and v16 IK reference clips.
2. `sh_tx/ty/tz` inside the rollout match the IK reference exactly at every control step (bit-exact up to float32).
3. `sh_tx/ty/tz` do **not** appear in any reward term.
4. Existing muscle-only reward signal (`joints`, `joints_vel`, `wrist_pos`, `bodies_pos`, `control_cost`, `saturation_cost`) continues to function, computed only over the four muscle-controlled DOFs and the non-shoulder-base body positions.
5. Triceps EMG timing comparison against bio EMG shifts to the correct onset (validation, not a training gate).

## Architecture

### Data flow

```
stac-mjx v16 IK h5 files  ──►  reference_data_moving_shoulder/  ──►  MouseReferenceClips (7-dim qpos)
                                                                                │
                                                                                ▼
                                                              MouseImitationMovingShoulder.step()
                                                                                │
                                                        ┌───────────────────────┤
                                                        │                       │
                                       mjx_env.step (muscles + arm)   override qpos[:3], qvel[:3]
                                                        │                       │
                                                        └──────────► next state ◄
```

The env exposes the 7-dim `data.qpos` to the policy via `proprioception`, so the network sees the instantaneous base position. Rewards are masked so the snapped DOFs never enter the optimization.

### Components

#### 1. New XML: `mouse_forelimb_right_moving_shoulder_ik.xml`

Location: `vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_moving_shoulder_ik.xml`

Source: Copy of `stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml` with these edits:

- **Delete** the three position actuators on lines 357–359:
  ```xml
  <position name="sh_tx_act" joint="sh_tx" kp="10" ctrlrange="-0.01 0.01"/>
  <position name="sh_ty_act" joint="sh_ty" kp="10" ctrlrange="-0.01 0.01"/>
  <position name="sh_tz_act" joint="sh_tz" kp="10" ctrlrange="-0.01 0.01"/>
  ```
- **Zero passive forces** on the three slide joints (lines 168–170) so the override is the only thing that moves them:
  - `damping="0"`, `stiffness="0"` (drop `springref` since it's only meaningful with nonzero stiffness).
  - Leave the `range="-0.01 0.01"` in place as a sanity limit (IK values stay well inside).
- Everything else (muscles, elbow actuator, bodies, sites, cameras) stays identical.

Result: 4 muscle + 1 elbow = same control dimension as existing sigmoid_normal runs. `mj_model.nu` unchanged. `mj_model.nq` = 7 (vs 4 before).

#### 2. New constants in `consts.py`

```python
JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH = MOUSE_PATH / "xmls" / "mouse_forelimb_right_moving_shoulder_ik.xml"
MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH = MOUSE_PATH / "reference_data_moving_shoulder"
```

#### 3. New reference-data directory

Location: `vnl_playground/tasks/mouse/reference_data_moving_shoulder/`

Contents: Symlinks to every `A36-1_*_trial*_ik.h5` inside `/root/vast/eric/stac-mjx/refined_STACed_data_janelia_moving_shoulder_v2/` (46 trials).

Schema (already verified):
- `qpos`: `(100, 7)` float32, names `[sh_tx, sh_ty, sh_tz, sh_rotation, sh_extension, sh_elv, elbow]`
- `qvel`: `(100, 7)` float32
- `xpos`: `(100, 8, 3)`, names `[world, ground, shoulder_base, clavicle, scapula, humerus, ulna, wrist]`
- `xquat`: `(100, 8, 4)`

This schema drops directly into `MouseReferenceClips` — the loader is schema-agnostic on `qpos.shape[1]`.

#### 4. New env class: `MouseImitationMovingShoulder`

Location: `vnl_playground/tasks/mouse/imitation_moving_shoulder.py` (new file, subclasses `MouseImitation`).

Overrides:

**`default_config()`** — same as parent, but:
- `walker_xml_path = JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH`
- `reference_data_path = MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH`
- `recompute_kinematics = False` (reference was generated with the same kinematic chain)
- `reward_terms["joints"]["ik_driven_dims"] = 3` — new config key: number of leading qpos dims to exclude from the joints/joints_vel reward (the IK-driven ones).

**`reset()`** — inherited. The reference `qpos` has all 7 dims, so the initial pose is correct without any override.

**`step()`** — new implementation:

```python
def step(self, state, action):
    n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
    data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

    # Compute the frame we are now in (next control tick)
    info = state.info
    # --- IK override of shoulder translation DOFs ---
    cur_frame = self._get_cur_frame(data, info)
    # Clamp cur_frame to valid range so indexing is safe at episode end
    last_valid = self._clip_length() - 1
    cur_frame_clamped = jp.minimum(cur_frame, last_valid)
    ref = self.reference_clips.at(clip=info["reference_clip"], frame=cur_frame_clamped)
    n_ik = self._config.ik_driven_dims  # 3
    data = data.replace(
        qpos=data.qpos.at[:n_ik].set(ref.qpos[:n_ik]),
        qvel=data.qvel.at[:n_ik].set(ref.qvel[:n_ik]),
    )
    # Re-run forward kinematics so xpos/xquat reflect the snapped base pose.
    data = mjx.forward(self.mjx_model, data)

    # ... rest identical to parent: truncated, prev_action, obs, reward, done ...
```

`ik_driven_dims` becomes a first-class config field on the env (default `3`). This makes the override dimension explicit and lets future variants set it to 0 (fall back to free shoulder) without subclassing again.

**`_joints_reward` / `_joints_vel_reward`** — override to mask leading `ik_driven_dims`:

```python
@_registry.reward("joints")
def _joints_reward(self, data, info, metrics, weight, exp_scale):
    target = self._get_current_target(data, info)
    n = self._config.ik_driven_dims
    distance = jp.linalg.norm(target.joints[n:] - data.qpos[n:])
    metrics["joint_l2_error"] = distance
    reward = weight * jp.exp(-((distance / exp_scale) ** 2) / 2)
    metrics["rewards/joints"] = reward
    return reward
```

Same pattern for `_joints_vel_reward`. `wrist_pos`, `bodies_pos`, `control_cost`, `saturation_cost` need no change (they operate on bodies or on action, not on qpos dims).

**Termination `pose_error`** — also needs masking to avoid spurious early termination from the snapped DOFs' velocities propagating to non-shoulder bodies:

```python
@_registry.termination("pose_error")
def _bad_pose(self, data, info, max_l2_error):
    target = self._get_current_target(data, info)
    n = self._config.ik_driven_dims
    pose_error = jp.linalg.norm(target.joints[n:] - data.qpos[n:])
    return pose_error > max_l2_error
```

(The base error should actually always be 0 post-snap, so masking is mostly cosmetic — but it keeps the termination threshold comparable to sigmoid_normal.)

#### 5. New training entry point: `train_mouse_janelia_sigmoid_moving_shoulder.py`

Location: `/root/vast/eric/vnl-playground/train_mouse_janelia_sigmoid_moving_shoulder.py`

Created by copying `train_mouse_janelia_sigmoid_normal.py` and changing:

1. Imports:
   ```python
   from vnl_playground.tasks.mouse.imitation_moving_shoulder import MouseImitationMovingShoulder
   from vnl_playground.tasks.mouse.consts import (
       JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH,
       MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH,
   )
   ```
2. Env factory: replace `MouseImitation(...)` with `MouseImitationMovingShoulder(...)`.
3. `env_cfg.walker_xml_path = JANELIA_MOUSE_MOVING_SHOULDER_IK_XML_PATH`
4. `env_cfg.reference_data_path = MOUSE_REFERENCE_DATA_MOVING_SHOULDER_PATH`
5. Default `--run-name` / wandb `group` bumped to something like `s15_moving_shoulder`.
6. Everything else — distribution, encoder/decoder, PPO hyperparams, EMG comparison, checkpointing, render — identical. The EMG comparison logic is unaffected: we still pull the muscle activations from the action vector, still plot vs. triceps/biceps.

### Rewards, observations, and masking — summary table

| Component | Shape in sigmoid_normal | Shape in s15 | Masked? |
|---|---|---|---|
| `data.qpos` | (4,) | (7,) | First 3 snapped; full vector exposed in `proprioception` |
| `data.qvel` | (4,) | (7,) | First 3 snapped; full vector exposed in `proprioception` |
| `joints` reward L2 | 4-dim | 4-dim (dims 3:7) | Yes — IK dims excluded |
| `joints_vel` reward L2 | 4-dim | 4-dim (dims 3:7) | Yes |
| `pose_error` termination | 4-dim | 4-dim (dims 3:7) | Yes |
| `wrist_pos`, `bodies_pos` | body xpos | body xpos | No change |
| Action (muscle + elbow) | same dim | same dim | No change |
| `task_obs` (joint targets delta) | 4 × ref_len | 7 × ref_len | No — policy sees the IK target deltas for shoulder too (always ≈ 0 since we're snapped to it), which is fine |

### Error handling

- **Clamping frame index**: `cur_frame_clamped = min(cur_frame, last_valid_frame)` so the override at episode end reads a valid frame even when the env is about to terminate on truncation.
- **NaN guard**: existing `nan_termination` still applies; if IK qpos has a NaN it would propagate, but the h5 files are verified clean.
- **Range sanity**: slide-joint `range="-0.01 0.01"` in the XML is wider than observed IK excursion (~±0.004). If an IK value ever exceeds it, `mjx.forward` will clamp and trigger a small discrepancy — add a one-time assertion in env `__init__` that checks `max(abs(reference.qpos[:, :3])) < joint_range`.

### Testing plan

Not test-driven (no pytest infra in this repo for envs), but we'll gate progression on:

1. **Sanity script** (`scripts/sanity_check_s15.py`): instantiate env, run 50 steps with zero action, assert `data.qpos[:3]` matches `ref.qpos[:3]` bit-exact each step.
2. **Short training run** (100k steps): confirm reward curve increases, `body_errors/wrist_body` drops, no NaNs.
3. **Full training** (same budget as sigmoid_normal): compare triceps EMG onset vs. biology.

## Non-goals / YAGNI

- No new reward terms. The design is a drop-in replacement for `MouseImitation` with a masked reward and a post-step override.
- No changes to policy architecture, distribution, or PPO hyperparams.
- No retraining of the encoder on the shoulder-translation dims (the policy sees them in proprioception; nothing prevents it from using them as context).
- No support for switching `ik_driven_dims` between clips in one run — it's a static env config.
- No auto-generation of the reference_data_moving_shoulder directory inside the Python package; we symlink once as a setup step.

## Open questions (none blocking)

- Long-term: if v16 IK gets re-run (v17+), we'll want the reference dir to point at a versioned snapshot rather than a live symlink. For now, symlinks are fine since v16 is frozen.
