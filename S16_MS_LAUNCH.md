# S16-MS Launch Commands

**102 runs** across **6 scripts**, **6 GPUs**, organized as 2× 2-GPU jobs + 2× 1-GPU jobs. Each script = 17 cells at 800 M steps, single-seed (except Group V seed-variance controls).

All cells use the moving-shoulder XML (`mouse_forelimb_right_moving_shoulder_ik.xml`) except X9/X10 which override via `--walker-xml`. EMG reference normalized as `arr / p98(arr)` then clipped to `[0, 1]` (`--emg-norm-percentile 98`, unchanged from s15). Full 17-metric EMG set logged every eval cycle.

**Design:** 102 cells across 7 groups testing per-muscle tau asymmetry as the primary lag-fix lever, with balanced hunch layer, new-corner physics probes, seed-variance anchor, and novel-regime exploratory cells. Priority-ordered within each script so first 3 cells are must-run (primary hypothesis + biggest novel probes). Full rationale in `docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md`.

## Pre-launch checklist

All infra is already on `eric/janelia`:

- [x] `--muscle-tau-act`, `--muscle-tau-deact`, per-muscle tau overrides (`--biceps-tau-act`, etc.) — `train_mouse_janelia_sigmoid_moving_shoulder.py:393-405`.
- [x] `--biceps-force`, `--triceps-long-force`, `--triceps-lat-force`, `--brachialis-force` — absolute pre-fs actuator gainprm.
- [x] `--emg-norm-percentile 98` with clip-to-[0,1] (default since commit `b41db34`).
- [x] `--body-diaginertia`, `--joint-stiffness`, `--joint-armature`, `--saturation-cost`, `--saturation-margin` — existing CLI.
- [x] `--qvel-init {zeros,reference}` — existing choice.
- [x] Full 17-metric EMG logging in trainer eval loop.
- [x] Alternative XML files present: `mouse_forelimb_right_loose.xml`, `mouse_forelimb_right_ratios.xml` (untracked, loaded via `--walker-xml` path).

No code changes required.

## Optional pre-flight: Group D diagnostics (~2 h, no training)

Recommended but not gating. Before launching the 102 training runs, run the reference-shift audit and convergence check from the spec. If D1's reference-shift audit clears mean_corr ≥ 0.80 on both muscles for any existing s15 checkpoint, **stop and re-filter the bio reference instead of launching s16**.

D1 requires a ~30-minute code change to `scripts/emg_comparison.py` to add `--bio-shift-ms`. See spec § Group D for details.

## Interactive Job 1 — 2 GPUs ericmmimic2

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s16_ms_1.sh > /tmp/sweep_s16_ms_1_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s16_ms_2.sh > /tmp/sweep_s16_ms_2_master.log 2>&1 &
```

## Interactive Job 2 — 2 GPUs vastlrn

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s16_ms_3.sh > /tmp/sweep_s16_ms_3_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s16_ms_4.sh > /tmp/sweep_s16_ms_4_master.log 2>&1 &
```

## Interactive Job 3 — 1 GPU

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s16_ms_5.sh > /tmp/sweep_s16_ms_5_master.log 2>&1 &
```

## Interactive Job 4 — 1 GPU

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s16_ms_6.sh > /tmp/sweep_s16_ms_6_master.log 2>&1 &
```

## Script summary

| GPU | Script | Cells | Priority-1 cells (first 3 — must-run) |
|---|---|---:|---|
| Job1 GPU0 | `sweep_s16_ms_1.sh` | 17 | **T1d** tau=25, **B03-aggr** S3+asym-aggr, **X1** qvel_init=reference |
| Job1 GPU1 | `sweep_s16_ms_2.sh` | 17 | **T3c** biceps-only 45, **B03-mild** S3+asym-mild, **X7** broken kinematic-tracking |
| Job2 GPU0 | `sweep_s16_ms_3.sh` | 17 | **T4b** per-muscle (45/30/15/15), **C04** biceps-force 0.07 + btau 55, **X8** sat-cost 0.1 |
| Job2 GPU1 | `sweep_s16_ms_4.sh` | 17 | **T2b** tau_deact=60, **B01-aggr** anchor-A+asym-aggr, **X4** body-diaginertia 5e-6 |
| Job3 GPU0 | `sweep_s16_ms_5.sh` | 17 | **C01** biceps-force 0.07, **X9** loose-XML, **V1** seed 2 |
| Job4 GPU0 | `sweep_s16_ms_6.sh` | 17 | **N1** fs=0.7, **X6** joint-stiffness 1e-5, **V2** seed 3 |
| **Total** |  | **102** | 18 must-run (first 3 × 6 GPUs) |

## Cell group distribution

| Group | Count | Purpose | Script distribution |
|---|---:|---|---|
| T — tau characterization | 18 | Dense tau grid at S3 anchor | 1(4), 2(4), 3(3), 4(3), 5(3), 6(1) |
| B — breadth × tau profile | 54 | 18 configs × {sym25, mild, aggr} | 1(9), 2(8), 3(8), 4(8), 5(9), 6(12) |
| C — balanced hunches | 12 | biceps/triceps/reward probes | 1(1), 2(2), 3(3), 4(2), 5(2), 6(2) |
| N — new-corner physics | 5 | force-scale 0.7/1.5 + armature | 1(1), 2(1), 3(1), 4(1), 5(0), 6(1) |
| V — seed-variance | 3 | B03-mild at seeds 2/3/4 | 5(1), 6(2) |
| X — novel exploratory | 10 | qvel/sim-dt/discounting/inertia/stiffness/reward/XML | 1(2), 2(2), 3(2), 4(2), 5(1), 6(1) |

## Wall-clock estimates

Per-run wall-clock at 800 M steps with full EMG metric logging: **~3.0–3.5 h**.
Per-GPU budget per window: **12 h** (override via `BUDGET_HOURS=<n>`).
Per-run estimate for the budget-check: **12600 s = 3.5 h** (override via `ESTIMATED_RUN_SECONDS=<s>`).

Expected completion per GPU per 12 h window: **3 cells**. Each script has 17 cells, so **~5 budget windows per GPU** to run everything. Total wall-clock across 6 GPUs running in parallel: **~60 h** (~2.5 days).

Relaunch the same script in a new window to resume — the budget guard cleanly stops before starting a run it can't complete, so cells run in priority order across multiple launch windows.

## Success criteria (quick reference)

**Primary ranking metric:** `min(biceps_lagged_corr_max, triceps_lagged_corr_max)`.

**Secondary ranking (s16-specific):** `min(biceps_mean_corr, triceps_mean_corr)` — zero-lag shape. Measures whether the lag is actually gone, not just compensable.

**Ship gates (per run; all must pass):**
- `eval/episode_reward ≥ 400`
- `eval/emg_{m}_lagged_corr_max ≥ 0.85` both muscles (↑ from s15's 0.80)
- `eval/emg_{m}_mean_corr ≥ 0.75` both muscles (**new headline — proves lag is gone**)
- `|eval/emg_{m}_phase_lag_ms| ≤ 15` both muscles (↓ from s15's 20)
- `eval/emg_{m}_mae ≤ 0.15` both muscles
- `eval/emg_{m}_trial_corr_mean ≥ 0.5` both muscles
- `eval/emg_{m}_lagged_corr_edge_saturated == 0` both muscles

**Tie-breakers** (if > 1 cell passes):
1. Lower `|biceps_phase_lag_ms| + |triceps_phase_lag_ms|`
2. Lower `biceps_mae + triceps_mae`
3. Higher `min(mean_corr)` across muscles

**If ≥ 1 cell passes**, replicate the winner at 3 seeds (~3 cells, ~10 h GPU) to confirm.

## Watch flags during training (wandb live)

- `eval/emg_{biceps,triceps}_phase_lag_ms` — the headline. Should drop from s15's 45/27 ms toward ≤ 15 ms.
- `eval/emg_{biceps,triceps}_mean_corr` — should climb toward `lagged_corr_max` as lag closes.
- `eval/emg_{biceps,triceps}_lagged_corr_edge_saturated` — if `1`, true lag is outside ±50 ms window and metrics underestimate.
- `eval/emg_{biceps,triceps}_per_trial_phase_lag_std_ms` — trial-to-trial timing jitter. Group V cells let us estimate how much of this is seed noise.

## Crash policy

- Each `sweep_s16_ms_N.sh` uses `set -o pipefail` and logs both stdout and stderr per run. A crashed training step records the run to `CRASHED` and the loop continues to the next cell.
- To re-run only skipped/crashed cells after a batch completes, copy the relevant `run_cell ...` line from the appropriate script and invoke it on a fresh GPU.

## Anchor summary

Most cells sit on one of these bases (fs, joint_damping, shoulder_damping, control_cost, control_diff_cost) with a specific tau profile and/or probe twist:

| Base | fs | joint_damp | shoulder_damp | cc | cdc | Origin |
|---|---|---|---|---|---|---|
| **S3 (B03)** | 1.1 | 1.5e-6 | 6e-7 | 0.025 | 0.025 | s15 leader — most-probed |
| anchor-A (B01) | 1.1 | 9e-7 | 9e-7 | 0.025 | 0.025 | s13 historical leader |
| anchor-C (B02) | 1.2 | 1e-6 | 1e-6 | 0.035 | 0.0 | s13 high-R branch |
| R2 (B04) | 1.1 | 9e-7 | 9e-7 | 0.05 | 0.0 | s15 smoothOnly reward |
| F4 (B05) | 0.9 | 6e-7 | 6e-7 | 0.025 | 0.025 | s15 slow-soft surprise |
| s11-goldi (B06) | 1.0 | 5e-7 | 5e-7 | 0.05 | 0.1 | s11 low-regime leader |

## Tau profiles (applied via per-muscle tau overrides)

| Profile | biceps | brachialis | triceps_long | triceps_lat | When used |
|---|---|---|---|---|---|
| **τ-sym25** | 25 | 25 | 25 | 25 | `--muscle-tau-act 0.025` |
| **τ-asym-mild** | 30 | 25 | 20 | 20 | `--muscle-tau-act 0.020 --biceps-tau-act 0.030 --brachialis-tau-act 0.025` |
| **τ-asym-aggr** | 45 | 30 | 20 | 20 | `--muscle-tau-act 0.020 --biceps-tau-act 0.045 --brachialis-tau-act 0.030` |

Triceps muscles take the global `--muscle-tau-act` (20 ms in both asym profiles) because no per-muscle override is applied. Shoulder abductor muscles also take the global value.

## Post-sweep decision tree

1. **≥ 1 cell clears all gates** → 3-seed replication of the winner, done.
2. **No cell clears all gates**, by failure mode:
   - **MAE blown only** → per-muscle force sweep follow-up (~10 cells).
   - **phase_lag_ms blown only** → reference-side issue; accept D1 shift offset.
   - **Shape blown only (`lagged_corr_max < 0.85`)** → finer tau resolution at best T-cell (~6 cells).
   - **Everything blown** → s17 candidates include EMG-in-loss reward term.

## Related docs

- `docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md` — full design spec
- `docs/superpowers/specs/2026-04-23-s15-ms-design.md` — s15 predecessor (metric infra baseline)
- `S15_MS_LAUNCH.md` — s15 launch doc (same partition template)
