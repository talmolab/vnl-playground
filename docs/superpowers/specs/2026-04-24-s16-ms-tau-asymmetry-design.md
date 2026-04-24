# s16-ms Design: Per-muscle Tau Asymmetry + Novel-Regime Probes

**Date:** 2026-04-24
**Supersedes / succeeds:** s15 (per-muscle-tau-agnostic broad recon)
**Primary hypothesis:** Phase-lag asymmetry (biceps ~45 ms, triceps ~27 ms) is driven by muscle activation dynamics being too fast relative to bio — fixable with per-muscle `tau_act` slowdown tuned to each muscle's measured lag.

## Problem

s15 closed shape (top `lagged_corr_max` = 0.89 biceps / 0.93 triceps at S3) but failed all ship gates on every one of its 30 runs. Two dominant failure modes:

1. **Systematic phase lag: sim leads bio by 30–50 ms.**
   - Median `eval/emg_biceps_phase_lag_ms = 45.0` (75th pct = 49.4; edge-saturated in 8/30 runs → true lag > 50 ms).
   - Median `eval/emg_triceps_phase_lag_ms = 27.5`.
   - Asymmetric: biceps lag ≈ 1.7× triceps lag. Global `tau_act` cannot fix both simultaneously.
2. **Biceps MAE blown (triceps fine).**
   - Top-5 s15 runs: biceps MAE 0.18–0.29 (gate = 0.15); triceps MAE 0.07–0.11.
   - Biceps amplitude envelope is too hot; triceps envelope fits.

s15 never swept `muscle_tau_act`, `muscle_tau_deact`, per-muscle force, `joint_armature`, or alternative XMLs. s16 attacks these axes.

## Strategy: seven groups, staged launch

### Group D — Pre-flight diagnostics (0 training, ~2 h GPU-eval)

Rule out the cheapest non-muscle-dynamics explanations **before** launching 102 training runs.

- **D1 — Reference-shift audit.** Add `--bio-shift-ms` flag to `scripts/emg_comparison.py` (≈30 min). Re-eval existing s15 S3, R2, A3 checkpoints at biceps shift ∈ {0, -25, -45, -65} ms × triceps shift ∈ {0, -20, -35} ms. If any `(bshift, tshift)` gives `mean_corr ≥ 0.80` on both muscles at `--emg-norm-percentile 98`, **the problem is reference alignment, not muscle dynamics.** Abort training-sweep launch, open a PR to re-filter the bio reference with `scipy.signal.filtfilt`, and re-run s15 analysis with the fixed reference.
- **D2 — Convergence check.** Plot `eval/emg_biceps_phase_lag_ms` and `eval/emg_triceps_phase_lag_ms` training histories from the top-5 s15 wandb runs. If either metric is still monotonically decreasing at step 800M → increase s16 `--num-timesteps` to 1.2 B. If plateaued by step 500M → keep 800M.
- **D3 — Filter-pipeline inspection.** Document the bio-EMG preprocessing filter (order, cutoff, causal vs zero-phase) by reading the reference-loading code in `train_mouse_janelia_sigmoid_moving_shoulder.py` around line 143. Used to interpret D1's result.

**Gating:** If D1 clears gates on ≥ 1 checkpoint, stop here. Otherwise proceed. D2/D3 feed annotations into the spec's run parameters but don't gate launch.

### Group T — Tau characterization at S3 anchor (18 cells)

Dense tau grid at a single config (S3: `fs=1.1, joint_damping=1.5e-6, shoulder_damping=6e-7, control_cost=0.025, control_diff_cost=0.025`). Separates the shape-vs-timing tradeoff that a broad factorial cannot.

| ID | tau_act config | tau_deact | Purpose |
|---|---|---|---|
| T1a | global 10 | 40 | MuJoCo default — reproduce s15 baseline |
| T1b | global 15 | 40 | Near-default |
| T1c | global 20 | 40 | Mild slowdown |
| T1d | global 25 | 40 | **Primary hypothesis midpoint** |
| T1e | global 30 | 40 | User's upper suggestion |
| T1f | global 40 | 40 | Heavy slowdown |
| T1g | global 55 | 40 | Extreme — shape-cap test |
| T2a | global 25 | 30 | Fast decay |
| T2b | global 25 | 60 | Long decay |
| T2c | global 25 | 100 | Very long decay |
| T3a | biceps 25 / others 15 | 40 | Biceps-only slowdown, mild |
| T3b | biceps 35 / others 15 | 40 | Biceps-only, mid |
| T3c | biceps 45 / others 15 | 40 | Biceps-only, matching measured lag |
| T3d | biceps 55 / others 15 | 40 | Biceps-only, over-shoot |
| T3e | biceps 70 / others 15 | 40 | Extreme biceps slowdown |
| T4a | b=35, br=25, tl=20, tla=20 | 40 | = τ-asym-mild profile (shared with B/C/N/S/X) |
| T4b | b=45, br=30, tl=15, tla=15 | 40 | Aggressive biceps, slightly faster triceps |
| T4c | b=55, br=40, tl=25, tla=25 | 40 | All-slow asymmetric |

### Group B — Breadth × 3 tau profiles (54 cells)

**3 tau profiles** (τ-sym15 control is covered by T1b; dropped from B to save cells):

| Profile | biceps | brachialis | triceps_long | triceps_lat |
|---|---|---|---|---|
| **τ-sym25** | 25 | 25 | 25 | 25 |
| **τ-asym-mild** | 30 | 25 | 20 | 20 |
| **τ-asym-aggr** | 45 | 30 | 20 | 20 |

`tau_deact=40` fixed in Group B. 18 B-configs:

| ID | fs | joint_damp | shoulder_damp | cc | cdc | Origin / hypothesis |
|---|---|---|---|---|---|---|
| B01 | 1.1 | 9e-7 | 9e-7 | 0.025 | 0.025 | s13 anchor-A / s15 A1 — historical leader |
| B02 | 1.2 | 1e-6 | 1e-6 | 0.035 | 0.0 | s13 anchor-C |
| B03 | 1.1 | 1.5e-6 | 6e-7 | 0.025 | 0.025 | **s15 S3 (current leader)** |
| B04 | 1.1 | 9e-7 | 9e-7 | 0.05 | 0.0 | s15 R2 smoothOnly |
| B05 | 0.9 | 6e-7 | 6e-7 | 0.025 | 0.025 | s15 F4 — slow-soft |
| B06 | 1.0 | 5e-7 | 5e-7 | 0.05 | 0.1 | s11 / s15 A3 goldilocks |
| B07 | 1.3 | 9e-7 | 4e-7 | 0.025 | 0.025 | high fs + weak shoulder |
| B08 | 0.9 | 6e-7 | 6e-7 | 0.0 | 0.05 | slow-soft + bursty |
| B09 | 1.2 | 1.2e-6 | 5e-7 | 0.05 | 0.0 | high fs + asym damp + smoothOnly |
| B10 | 1.1 | 1.5e-6 | 1.5e-6 | 0.025 | 0.025 | strong symmetric damping |
| B11 | 1.0 | 6e-7 | 3e-7 | 0.025 | 0.025 | weakest-shoulder probe |
| B12 | 1.2 | 1.5e-6 | 5e-7 | 0.025 | 0.025 | S3-damping at fs=1.2 |
| B13 | 1.1 | 1.2e-6 | 5e-7 | 0.05 | 0.0 | mid-stiff + weak shoulder + smoothOnly |
| B14 | 1.1 | 1.5e-6 | 6e-7 | 0.0 | 0.05 | S3 damping + bursty |
| B15 | 1.0 | 5e-7 | 5e-7 | 0.0 | 0.05 | low damping + bursty |
| B16 | 1.3 | 1.2e-6 | 1.2e-6 | 0.0 | 0.05 | high fs + strong damp + bursty |
| B17 | 1.2 | 1.5e-6 | 4e-7 | 0.025 | 0.025 | S3-style at fs=1.2 with weaker shoulder |
| B18 | 1.4 | 1.2e-6 | 1.2e-6 | 0.025 | 0.025 | fs=1.4 + stiffer damp |

18 configs × 3 profiles = **54 cells**.

### Group C — Balanced hunches (12 cells, τ-asym-mild unless noted)

Rebalanced to cover biceps, triceps, and reward hypotheses symmetrically.

| ID | Base | Twist | Hypothesis |
|---|---|---|---|
| C01 | B03 (S3) | `--biceps-force 0.7×` | Biceps amplitude too high |
| C02 | B03 | `--biceps-force 0.5×` | Strong biceps amplitude cut |
| C03 | B03 | `--triceps-long-force 0.8× --triceps-lat-force 0.8×` | Symmetric control: triceps amplitude too high |
| C04 | B03 | `--biceps-force 0.7× --biceps-tau-act 55` | Stack biceps amplitude + timing |
| C05 | B03 | `--saturation-cost 0.02` | Penalize saturated activation |
| C06 | B03 | `--joints-vel-weight 0.2` | Loosen velocity tracking |
| C07 | B03 | `--joints-vel-weight 0.0` | Drop velocity tracking entirely |
| C08 | B03 | `--muscle-tau-deact 100` | Long-decay probe |
| C09 | B03 + τ-asym-aggr | `--biceps-force 0.7×` | Stack aggressive tau + force |
| C10 | B05 (F4) + τ-asym-aggr | `--biceps-force 0.6×` | Biceps-bad cell + aggressive fix |
| C11 | B04 (R2) + τ-asym-aggr | `--saturation-cost 0.02` | Smooth reward + aggressive tau + sat penalty |
| C12 | B01 (s13 anchor-A) | `--biceps-force 0.8×` | Historical leader + mild biceps cut |

### Group N — New-corner physics probes (5 cells, τ-asym-mild, S3 base)

Never-touched corners of the established parameter space.

| ID | Change |
|---|---|
| N1 | `--force-scale 0.7` (low-force corner, never tested) |
| N2 | `--force-scale 1.5` (high-force corner, never tested) |
| N3 | `--joint-armature 1e-10` (lower inertia) |
| N4 | `--joint-armature 1e-9` (higher inertia) |
| N5 | `--joint-armature 4e-9` (much higher inertia) |

### Group V — Seed-variance control (3 cells)

Same config as `B03 + τ-asym-mild`, seeds 1, 2, 3. Measures noise floor for interpreting single-seed comparisons across the remaining 99 cells. ("V" avoids collision with the s15 cell label "S3".)

| ID | Config | Seed |
|---|---|---|
| V1 | B03 + τ-asym-mild | 1 |
| V2 | B03 + τ-asym-mild | 2 |
| V3 | B03 + τ-asym-mild | 3 |

### Group X — Novel exploratory probes (10 cells, τ-asym-mild, S3 base)

Single-cell probes on genuinely untouched levers. Not trying to win the sweep — trying to reveal a new axis for s17.

| ID | Change | Hypothesis |
|---|---|---|
| X1 | `--qvel-init reference` | Remove "catch-up burst" at episode t=0 |
| X2 | `--sim-dt 0.000625` (2× finer) | Tests if muscle activation ODE is aliased |
| X3 | `--discounting 0.995` | Longer credit-assignment horizon — delays firing |
| X4 | `--body-diaginertia 5e-6` | Heavier limb → natural firing delay |
| X5 | `--body-diaginertia 2e-7` | Lower inertia — sensitivity probe |
| X6 | `--joint-stiffness 1e-5` | Passive spring restoring force |
| X7 | `--joints-weight 0.5 --joints-vel-weight 0.05 --wrist-pos-weight 0.02 --bodies-pos-weight 0.02` | Break kinematic-tracking dictatorship |
| X8 | `--saturation-cost 0.1 --saturation-margin 0.8` | Strongly penalize > 0.8 activation |
| X9 | `--walker-xml vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_loose.xml` | Structurally different baseline |
| X10 | `--walker-xml vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_ratios.xml` | Built-in muscle force ratios |

**Deferred to s17 (require code work):**
- EMG-tracking reward term (rejected for s16).
- Train-time `--bio-reference-shift-ms` (revisit if D1 shift audit is informative).

## Totals

| Group | Cells | Purpose |
|---|---:|---|
| D | 0 training + 3 eval | Pre-flight diagnostics |
| T | 18 | Dense tau characterization at S3 |
| B | 54 | Breadth × 3 tau profiles |
| C | 12 | Balanced hunches |
| N | 5 | New-corner physics probes |
| V | 3 | Seed-variance anchor |
| X | 10 | Novel exploratory probes |
| **Total** | **102** training | |

Single-seed throughout except V1–V3 and B03+τ-asym-mild (which is part of both Group V and Group B — treat as seed 0 of the 3-seed pack; cell counts here count the 3 V cells, B03+τ-asym-mild is the shared seed-0 run).

## Fixed base arguments (all cells)

```
--ctrl-dt 0.0025 --sim-dt 0.00125
--joint-armature 4e-10
--episode-length 100
--num-timesteps 800000000 --num-evals 8
--joints-weight 5.0 --joints-vel-weight 0.5
--wrist-pos-weight 0.1 --bodies-pos-weight 0.1
--qvel-init zeros
--emg-norm-percentile 98
```

XML: `mouse_forelimb_right_moving_shoulder_ik.xml` (moving-shoulder default).

Overrides per group: Group T varies tau. Group B varies fs/damping/reward + tau profile. Group C, N, X, S listed in their tables. Group X9/X10 override `--walker-xml`.

## Success criteria

**Primary ranking metric:** `min(biceps_lagged_corr_max, triceps_lagged_corr_max)` — weakest-muscle-wins.

**Secondary ranking (s16-specific):** `min(biceps_mean_corr, triceps_mean_corr)` — zero-lag shape. The whole point is to close the `mean_corr` ↔ `lagged_corr_max` gap. A run where both are nearly equal has actually eliminated the lag, not just matched shape.

**Ship gates (applied per run; all must pass):**

- `eval/episode_reward ≥ 400`
- `eval/emg_biceps_lagged_corr_max ≥ 0.85` (↑ from s15's 0.80)
- `eval/emg_triceps_lagged_corr_max ≥ 0.85`
- `eval/emg_biceps_mean_corr ≥ 0.75` (**new headline — proves lag is gone**)
- `eval/emg_triceps_mean_corr ≥ 0.75`
- `|eval/emg_biceps_phase_lag_ms| ≤ 15` (↓ from s15's 20)
- `|eval/emg_triceps_phase_lag_ms| ≤ 15`
- `eval/emg_biceps_mae ≤ 0.15`
- `eval/emg_triceps_mae ≤ 0.15`
- `eval/emg_biceps_trial_corr_mean ≥ 0.5`
- `eval/emg_triceps_trial_corr_mean ≥ 0.5`
- `eval/emg_biceps_lagged_corr_edge_saturated == 0`
- `eval/emg_triceps_lagged_corr_edge_saturated == 0`

**Tie-breakers** (if > 1 cell passes):

1. Lower `|biceps_phase_lag_ms| + |triceps_phase_lag_ms|`
2. Lower `biceps_mae + triceps_mae`
3. Higher `min(mean_corr)` across muscles

## Post-sweep decision tree

1. **≥ 1 cell clears all gates** → 3-seed replication of the winner (+3 cells). Done.
2. **No cell clears all gates**, partitioned by failure mode:
   - **MAE blown only** (lag + shape OK) → per-muscle force is the missing knob. Launch follow-up: `biceps_force` ∈ {0.5, 0.6, 0.7, 0.8, 0.9} × `triceps_force` ∈ {0.8, 1.0} at top-tau config. ~10 cells.
   - **`phase_lag_ms` blown only** (MAE + shape OK) → reference-side issue. Accept D1's shift offset, report shape-only metrics. No follow-up training.
   - **Shape blown (`lagged_corr_max < 0.85`) but lag + MAE OK** → muscle dynamics tradeoff biting — finer tau resolution at best T-cell. ~6 cells.
   - **Everything blown** → consider s17 with EMG-in-loss reward term (currently rejected).

## Script partition (6 scripts, 2× 2-GPU jobs + 2× 1-GPU jobs)

17 cells per script × 6 scripts = 102. Priority-1 (first 3 cells of each script) hit the primary hypothesis from multiple angles.

| Job | GPU | Script | Cells | Priority-1 |
|---|---|---|---:|---|
| Job1 ericmmimic2 | GPU0 | `sweep_s16_1.sh` | 17 | T1d (tau_act=25), B03-asym-aggr, X1 (qvel_init=reference) |
| Job1 ericmmimic2 | GPU1 | `sweep_s16_2.sh` | 17 | T3c (biceps-only 45), B03-asym-mild, X7 (broken-tracking reward) |
| Job2 vastlrn | GPU0 | `sweep_s16_3.sh` | 17 | T4b (b=45 tl=15), C04 (stack), X8 (sat-cost 0.1) |
| Job2 vastlrn | GPU1 | `sweep_s16_4.sh` | 17 | T2b (tau_deact=60), B01-asym-aggr, X4 (high inertia) |
| Job3 | GPU0 | `sweep_s16_5.sh` | 17 | C01 (bforce 0.7×), X9 (loose XML), V1 (seed 1) |
| Job4 | GPU0 | `sweep_s16_6.sh` | 17 | N1 (fs=0.7), X6 (joint stiffness), V2 (seed 2) |

Priority principle: within each script, cells are ordered so the first 3 are "must-run" (critical hypotheses); 4th through ~12th are "should-run" (B/C/N); last 2–5 are "budget-permitting" (probes we can lose without killing signal).

Each script uses the same `BUDGET_HOURS=12` / `ESTIMATED_RUN_SECONDS=12600` guard as s15.

## Budget

Per-run wall-clock at 800M steps with the 17-metric EMG pipeline: ~3.0–3.5 h (same as s15).
Per-GPU budget per window: **12 h** → ~3 cells per GPU per window.
Total: 102 cells / 18 per window ≈ **5 budget windows** → **~60 h wall-clock** across 4 job slots.

Worst case with a doubled `--num-timesteps` from D2 convergence check (1.2 B): ~7.5 windows → ~90 h.

## Risks

- **R1.** Per-muscle tau plumbing sets `actuator_dynprm[:, 0]` globally first, then overrides 4 named muscles. Any shoulder muscles in the XML get the global `--muscle-tau-act`. All T/B/C/N/S/X cells include an explicit `--muscle-tau-act` value so this is deterministic; the `τ-sym25` profile on Group B uses only `--muscle-tau-act 25` (no per-muscle overrides); asym profiles set the global to the minimum of the per-muscle values and override the 4 named muscles explicitly.
- **R2.** Shape may degrade as `tau_act` increases — a slower activation ODE flattens bursts. Group T's dense ladder separates "tau fixes lag" from "tau breaks shape." If T1's `lagged_corr_max` drops below 0.80 at tau_act ≥ 30 ms, Group B's asym-aggr profile is at risk; we'd fall back to asym-mild as primary.
- **R3.** Single-seed noise. Group S measures seed std at one cell; any claimed win within S's measured std × 1.5 requires 3-seed verification before shipping.
- **R4.** D1 could show the lag is reference-processing artifact, invalidating the entire training plan. This is a feature — we want to know.
- **R5.** Alternative XMLs (X9, X10) may fail to load or have different actuator layouts that break per-muscle tau overrides. Mitigation: launch X9/X10 cells last in their scripts so they can be skipped without losing core signal.
- **R6.** Budget overrun. 102 cells × 3.5 h / 6 GPUs = 60 h wall-clock, which exceeds any single 12 h launch window. Multiple sequential launches required.

## Rejected alternatives

- **EMG reward term.** User-rejected. Defer to s17 if s16 fails to ship.
- **Train-time `--bio-reference-shift-ms`.** Deferred pending D1 result. Revisit in s17 if D1 shows eval-side shift helps.
- **Full factorial over per-muscle tau + per-muscle force.** Combinatorial explosion (4 muscles × ~3 levels each = 81 combos per cell). Instead, Group T explores tau on the dominant muscle (biceps); Group C explores force on biceps and triceps separately.
- **Another single-seed broad recon without tau.** Falsified by s15 — no configuration in the s13-s15 parameter space clears lag gates without tau intervention.
- **Changing `--ctrl-dt`.** User-rejected.

## Out of scope

- Changes to the reference XML/skeleton (beyond pointing `--walker-xml` at existing variants).
- Changes to the PPO trainer core or observation space.
- New observation-history / frame-stacking flags (flagged as s17 candidate).
- Video-diagnostic overhaul.
- New muscles or joints.
- Bio-EMG preprocessing changes (separate PR if D1 warrants).
