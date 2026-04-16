# S10-MS Moving-Shoulder Sweep Design

**Date:** 2026-04-16
**Branch:** `eric/janelia`
**Entry point:** `train_mouse_janelia_sigmoid_moving_shoulder.py`
**Walker XML:** `mouse_forelimb_right_moving_shoulder_ik.xml` (script default — no `--walker-xml` needed)

## Motivation

The s9-moving-shoulder sweep (`s9runs_moving_shoulder.csv`, 20 runs) had 10 failures that died ~21 min in because of a transient code regression. The 10 runs that finished never explored damping below 5e-7, force-scale outside 0.3–1.0, muscle-tau at all, or the control-cost / control-diff response surface. This sweep does all of the above, plus reruns the dead configs, plus grounds at three baselines (B / C / D).

## Baselines

| Name | fs | damp | cc | cdc | arm | Notes |
|---|---|---|---|---|---|---|
| **B** | 0.3 | 9e-7 | 0.05 | 0.1 | 4e-10 | S11/S13/S14 baseline, moving-shoulder XML |
| **C** | 0.6 | 5e-7 | 0.05 | 0.1 | 4e-10 | Mid-range s9-ms finished config |
| **D** | 0.3 | 1e-7 | 0.05 | 0.1 | 4e-10 | "Try 1e-7 like we were doing before" |

Fixed across every cell: `ctrl-dt=0.0025, sim-dt=0.00125, episode-length=100, qvel-init=zeros, joint-armature=4e-10, joints-weight=5.0, joints-vel-weight=0.5, wrist-pos-weight=0.1, bodies-pos-weight=0.1, num-timesteps=800000000, num-evals=8`.

## Script layout

6 shell scripts, matching the **"2 × 2-GPU jobs + 2 × 1-GPU jobs"** layout (same pattern as S10_LAUNCH.md).

| Script | GPU role | Cells | Sweep dimension |
|---|---|---:|---|
| `sweep_s10_ms_1.sh` | 2-GPU job 1 / GPU0 | 10 | Damping (at fs=0.3, cc=0.05, cdc=0.1) |
| `sweep_s10_ms_2.sh` | 2-GPU job 1 / GPU1 | 10 | Force scale (at damp=9e-7, cc=0.05, cdc=0.1) |
| `sweep_s10_ms_3.sh` | 2-GPU job 2 / GPU0 | 9 | Control cost (at baseline B) |
| `sweep_s10_ms_4.sh` | 2-GPU job 2 / GPU1 | 9 | Control diff (at baseline B) |
| `sweep_s10_ms_5.sh` | 1-GPU job 1 | 10 | Failed s9-ms reruns (exact configs) |
| `sweep_s10_ms_6.sh` | 1-GPU job 2 | 11 | Muscle-tau + baseline-C + baseline-D grounding |
| **Total** | | **59** | |

## Cell contents

### `sweep_s10_ms_1.sh` — damping sweep (10 cells)

Baseline B except for damping. All cells: `fs=0.3, cc=0.05, cdc=0.1, seed=1`.

| # | tag | damping |
|---|---|---|
| 1 | `d1em6-fs0p3` | 1e-6 |
| 2 | `d9em7-fs0p3` | 9e-7 |
| 3 | `d8em7-fs0p3` | 8e-7 |
| 4 | `d7em7-fs0p3` | 7e-7 |
| 5 | `d6em7-fs0p3` | 6e-7 |
| 6 | `d5em7-fs0p3` | 5e-7 |
| 7 | `d4em7-fs0p3` | 4e-7 |
| 8 | `d3em7-fs0p3` | 3e-7 |
| 9 | `d2em7-fs0p3` | 2e-7 |
| 10 | `d1em7-fs0p3` | 1e-7 |

### `sweep_s10_ms_2.sh` — force-scale sweep (10 cells)

Baseline B except force scale. All cells: `damp=9e-7, cc=0.05, cdc=0.1, seed=1`.

| # | tag | fs |
|---|---|---|
| 1 | `d9em7-fs0p1` | 0.1 |
| 2 | `d9em7-fs0p2` | 0.2 |
| 3 | `d9em7-fs0p3` | 0.3 |
| 4 | `d9em7-fs0p4` | 0.4 |
| 5 | `d9em7-fs0p5` | 0.5 |
| 6 | `d9em7-fs0p6` | 0.6 |
| 7 | `d9em7-fs0p7` | 0.7 |
| 8 | `d9em7-fs0p8` | 0.8 |
| 9 | `d9em7-fs0p9` | 0.9 |
| 10 | `d9em7-fs1p0` | 1.0 |

### `sweep_s10_ms_3.sh` — control-cost sweep (9 cells)

Baseline B except cc. All cells: `fs=0.3, damp=9e-7, cdc=0.1, seed=1`.

| # | tag | control_cost |
|---|---|---|
| 1 | `cc0p00` | 0.0 |
| 2 | `cc0p01` | 0.01 |
| 3 | `cc0p025` | 0.025 |
| 4 | `cc0p05` | 0.05 |
| 5 | `cc0p06` | 0.06 |
| 6 | `cc0p07` | 0.07 |
| 7 | `cc0p08` | 0.08 |
| 8 | `cc0p09` | 0.09 |
| 9 | `cc0p10` | 0.10 |

### `sweep_s10_ms_4.sh` — control-diff sweep (9 cells)

Baseline B except cdc. All cells: `fs=0.3, damp=9e-7, cc=0.05, seed=1`.

| # | tag | control_diff_cost |
|---|---|---|
| 1 | `cdc0p00` | 0.0 |
| 2 | `cdc0p01` | 0.01 |
| 3 | `cdc0p025` | 0.025 |
| 4 | `cdc0p05` | 0.05 |
| 5 | `cdc0p06` | 0.06 |
| 6 | `cdc0p07` | 0.07 |
| 7 | `cdc0p08` | 0.08 |
| 8 | `cdc0p09` | 0.09 |
| 9 | `cdc0p10` | 0.10 |

### `sweep_s10_ms_5.sh` — failed-s9-ms reruns (10 cells)

Exact configs of the 10 dead s9-ms runs, with `cc=0.05, cdc=0.1, seed=1`.

| # | tag | damp | fs |
|---|---|---|---|
| 1 | `d1em7-fs0p05` | 1e-7 | 0.05 |
| 2 | `d5em7-fs0p3` | 5e-7 | 0.3 |
| 3 | `d5em7-fs0p6` | 5e-7 | 0.6 |
| 4 | `d5em7-fs0p7` | 5e-7 | 0.7 |
| 5 | `d8em7-fs0p4` | 8e-7 | 0.4 |
| 6 | `d8em7-fs0p5` | 8e-7 | 0.5 |
| 7 | `d8em7-fs0p7` | 8e-7 | 0.7 |
| 8 | `d8em7-fs1p0` | 8e-7 | 1.0 |
| 9 | `d9em7-fs0p4` | 9e-7 | 0.4 |
| 10 | `d9em7-fs0p7` | 9e-7 | 0.7 |

### `sweep_s10_ms_6.sh` — muscle-tau + multi-baseline grounding (11 cells)

The training script exposes `--muscle-tau-act` and `--muscle-tau-deact` as **global** overrides (applied to all muscles). Cells 1–6 sweep these globally. Cells 7–11 ground two alternate baselines.

Muscle-tau sweep at baseline B (`fs=0.3, damp=9e-7, cc=0.05, cdc=0.1, seed=1`):

| # | tag | tau_act (s) | tau_deact (s) | Notes |
|---|---|---|---|---|
| 1 | `tau-a01-d03` | 0.010 | 0.030 | fast-twitch lower end |
| 2 | `tau-a01-d04` | 0.010 | 0.040 | |
| 3 | `tau-a02-d04` | 0.020 | 0.040 | |
| 4 | `tau-a03-d04` | 0.030 | 0.040 | |
| 5 | `tau-a04-d06` | 0.040 | 0.060 | mixed human-like |
| 6 | `tau-mouse-mean` | 0.009 | 0.028 | fiber-weighted mouse-forelimb mean (f_slow~0.1, tau_fast=5/20 ms, tau_slow=40/90 ms) |

Baseline-C grounding (`fs=0.6, damp=5e-7, cc=0.05, cdc=0.1`, 3 seeds):

| # | tag | seed |
|---|---|---|
| 7 | `baselineC-s1` | 1 |
| 8 | `baselineC-s2` | 2 |
| 9 | `baselineC-s3` | 3 |

Baseline-D grounding (`fs=0.3, damp=1e-7, cc=0.05, cdc=0.1`, 2 seeds):

| # | tag | seed |
|---|---|---|
| 10 | `baselineD-s1` | 1 |
| 11 | `baselineD-s2` | 2 |

## Run-name convention

All runs: `s10-ms-<tag>-<YYYYMMDD-HHMMSS>`. Same as s11–s14. Each cell captures a `/tmp/sweep_s10_ms_<tag>.log`.

## Wandb

Each script writes to a dedicated `--wandb-group`:
- s10_ms_1 → `s10-ms-damping`
- s10_ms_2 → `s10-ms-force-scale`
- s10_ms_3 → `s10-ms-control-cost`
- s10_ms_4 → `s10-ms-control-diff`
- s10_ms_5 → `s10-ms-failed-rerun`
- s10_ms_6 → `s10-ms-tau-and-baselines`

Cross-cutting wandb tags on every run: `s10-ms moving-shoulder`. Each cell adds its own tags (e.g., `damping d5em7 fs0p3`).

## Out of scope for s10_ms

1. **Muscle-length proprioception.** The env's `_get_proprioception` returns `concat(qpos, qvel)` (imitation.py:258). Replacing that with `concat(actuator_length, actuator_velocity)` needs an env code change (new `--proprio-type {joint, muscle}` flag). Will be handled as a separate brainstorm → plan → code change → follow-up sweep (likely `sweep_s10_ms_7.sh` or `sweep_s11_ms_1.sh`), running best-known defaults × 3 seeds × 2 proprio modes = 6 cells.
2. **Per-muscle muscle-tau from fiber composition.** Requires XML variants (bake per-muscle `timeconst` into XML) or a new per-muscle CLI flag. Cell 6 in s10_ms_6 approximates this with the fiber-weighted global mean only.

## Crash policy

- Any single cell crashes → log, continue. Same as s9_ms.
- >3 crashes on one GPU → pause, investigate.
- If the "failed-rerun" script (s10_ms_5) has ≥5 crashes → pause; the underlying regression the user flagged isn't fixed.

## Success criteria

- All 59 cells complete without the s9_ms ~21-min regression.
- Damping sweep (s10_ms_1) produces a monotone or clear-optimum curve on reward & EMG.
- Force-scale sweep (s10_ms_2) confirms or refutes the s11-era "high fs = bad EMG" pattern on the moving-shoulder XML.
- Control-cost & control-diff sweeps (s10_ms_3/4) bound the reward–smoothness frontier for the moving shoulder.
- Muscle-tau sweep (s10_ms_6) shows whether global tau under 30 ms changes EMG alignment.
- Baseline-C and baseline-D grounding (s10_ms_6) give seeded variance on alternate operating points for later comparison.
