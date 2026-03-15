# Janelia Mouse Forelimb Parameter Sweep Plan

## Context

We have a working imitation learning setup for a 4-DOF mouse forelimb (12 muscle actuators) tracking IK reference clips via PPO. Each 1B-timestep run takes ~50 min. We want to systematically find better physics and training parameters. Two independent sweeps: XML physics parameters and training/RL parameters.

## Implementation: Sweep Infrastructure

### Step 1: Add physics override support to `base.py`

**File:** `vnl_playground/tasks/mouse/base.py`

Add optional config fields and a post-compile hook in `MouseBaseEnv.compile()` that modifies `mj_model` fields before `mjx.put_model()`:
- `joint_damping` (float) -> sets `mj_model.dof_damping[:]`
- `joint_armature` (float) -> sets `mj_model.dof_armature[:]`
- `joint_stiffness` (float) -> sets `mj_model.jnt_stiffness[:]`
- `force_scale` (float) -> multiplies `mj_model.actuator_gainprm[:, 0]`

### Step 2: Add argparse to training script

**File:** `vnl_playground/train_mouse_janelia_imitation.py`

Add CLI args for all sweep parameters (physics + training), sweep tag, and wandb group. Use these to override `env_cfg` and `ppo_params` before creating envs.

### Step 3: Create sweep shell scripts

**New files:** `sweep_physics.sh` and `sweep_training.sh`

Each script launches runs sequentially with different parameter combinations, using the argparse interface.

---

## Sweep 1: XML Physics Parameters (13 runs, ~11 hours)

Current defaults: `damping=1e-5, armature=4e-8, stiffness=1e-12`, muscle forces 0.2-0.6N.

### Phase 1A: Damping Sweep (6 runs) -- All below current 1e-5

| Run | Tag | Damping | Force Scale | Notes |
|-----|-----|---------|-------------|-------|
| S1-00 | `baseline` | 1e-5 | 1.0x | Comparison anchor (current) |
| S1-01 | `damp-5e-6` | 5e-6 | 1.0x | 2x less damping |
| S1-02 | `damp-1e-6` | 1e-6 | 1.0x | 10x less damping |
| S1-03 | `damp-5e-7` | 5e-7 | 1.0x | 20x less damping |
| S1-04 | `damp-1e-7` | 1e-7 | 1.0x | 100x less damping |
| S1-05 | `damp-1e-8` | 1e-8 | 1.0x | 1000x less damping (near zero) |

### Phase 1B: Armature (3 runs)

| Run | Tag | Damping | Armature | Force Scale | Notes |
|-----|-----|---------|----------|-------------|-------|
| S1-06 | `high-arm` | 1e-5 | 4e-6 | 1.0x | 100x more armature |
| S1-07 | `low-arm` | 1e-5 | 4e-10 | 1.0x | 100x less armature |
| S1-08 | `best-damp-high-arm` | *best from 1A* | 4e-6 | 1.0x | Best damping + higher armature |

### Phase 1C: Stiffness + Combined (3 runs)

| Run | Tag | Damping | Stiffness | Notes |
|-----|-----|---------|-----------|-------|
| S1-09 | `med-stiff` | *best from 1A* | 1e-6 | Add passive centering |
| S1-10 | `high-stiff` | *best from 1A* | 1e-3 | Strong passive springs |
| S1-15 | `best-combo` | *best* | *best* | Best combination of 1A-1C + 1D |

### Phase 1D: Force Distribution (4 runs) -- Target: all muscles ~0.1N

Current forces: most 0.2N, AD=0.4N, Supra/Subscap=0.6N, Infra=0.5N.

| Run | Tag | Forces | Notes |
|-----|-----|--------|-------|
| S1-11 | `half-force` | 0.5x scale (0.1-0.3N) | Half all forces |
| S1-12 | `all-0.1N` | All muscles = 0.1N | Uniform low force |
| S1-13 | `all-0.2N` | All muscles = 0.2N | Uniform medium force |
| S1-14 | `all-0.1N-low-damp` | All muscles = 0.1N, damping=best from 1A | Low force + best damping |

---

## Sweep 2: Training/RL Parameters (11 runs, ~9 hours)

Uses best physics from Sweep 1 (or baseline if Sweep 1 is inconclusive). Reward weights/exp_scales are held CONSTANT across all runs to keep reward magnitude comparable.

### Phase 2A: Reference Clip Structure (3 runs) -- Likely biggest impact

| Run | Tag | reference_length | episode_length | Notes |
|-----|-----|-----------------|----------------|-------|
| S2-01 | `ref-5` | 5 | 50 | 25ms lookahead (vs current 5ms) |
| S2-02 | `ref-10` | 10 | 50 | 50ms lookahead |
| S2-03 | `long-ep` | 5 | 90 | Cover 45 of 50 clip frames |

### Phase 2B: Entropy Cost (3 runs)

| Run | Tag | entropy_cost | Notes |
|-----|-----|-------------|-------|
| S2-04 | `ent-1e3` | 1e-3 | 10x less (faster convergence) |
| S2-05 | `ent-5e3` | 5e-3 | 2x less |
| S2-06 | `ent-1e1` | 1e-1 | 10x more (more exploration) |

### Phase 2C: Control Cost (2 runs)

| Run | Tag | control_cost | control_diff_cost | Notes |
|-----|-----|-------------|-------------------|-------|
| S2-07 | `no-ctrl` | 0.0 | 0.0 | Remove control penalty entirely |
| S2-08 | `ctrl-smooth` | 0.01 | 0.01 | Add smoothness penalty |

### Phase 2D: PPO Hyperparameters (3 runs)

| Run | Tag | Change | Notes |
|-----|-----|--------|-------|
| S2-09 | `lr-3e4` | lr=3e-4 | 3x lower learning rate |
| S2-10 | `disc-99` | discount=0.99 | Longer credit horizon |
| S2-11 | `big-batch` | batch=2048, minibatches=32 | 2x bigger batch |

---

## wandb Organization

- **Group**: `sweep1-physics` or `sweep2-training`
- **Tags**: `["janelia", "sweep1", tag]` or `["janelia", "sweep2", tag]`
- **Name**: `S1-00-baseline`, `S2-01-ref-5`, etc.
- All sweep params logged to wandb config for filtering

## Priority Order (if time-limited)

1. S1-00 (baseline) -- mandatory
2. S1-01 + S1-02 (core damping-force test)
3. S2-01 + S2-02 (reference length -- likely biggest training win)
4. S2-07 (zero control cost -- quick check)
5. S1-03, S1-04, S1-11 (more force/damping points)
6. Everything else

## Decision Criteria

Compare runs by: `joint_l2_error` (primary), `wrist_pos_error`, learning curve slope in first 200M steps, NaN/termination rate, `ctrl_sqr` (effort), and rollout video quality.

## Verification

After implementing the sweep infrastructure, verify by running baseline (S1-00) and confirming it reproduces current training behavior (same reward curve, same metrics).
