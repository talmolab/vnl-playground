# S15-MS Launch Commands

**30 runs** across **6 scripts**, **6 GPUs**, organized as 2× 2-GPU jobs + 2× 1-GPU jobs.
All cells use the moving-shoulder XML (`mouse_forelimb_right_moving_shoulder_ik.xml`, trainer default) and the new EMG metric pipeline: **reference normalized as `arr / p98(arr)` then clipped to [0, 1]** so the bio reference saturates at the same 1.0 ceiling as MuJoCo's Hill-model activation (`--emg-norm-percentile 98`). Full 17-metric EMG set logged every eval cycle (`lagged_corr_max`, `phase_lag_ms`, `lagged_corr_edge_saturated`, `trial_corr_mean`, per-trial lag mean/std, etc.).

**Design:** 30 hypothesis-driven parameter combos across 5 groups (anchors, shoulder-decoupling, reward-shaping, fs×damping fills, interactions). Each GPU runs 5 candidates single-seed; priority-ordered so the first 3 complete at ~10.5 h and the 4th/5th run budget-permitting (default `BUDGET_HOURS=12`, `ESTIMATED_RUN_SECONDS=12600`). Full rationale in `docs/superpowers/specs/2026-04-23-s15-ms-design.md` and `docs/superpowers/specs/2026-04-23-s15-ms-thinking.md`.

## Pre-launch checklist

All Stage 1 infra is already merged on `eric/janelia` (commits `5cc0553` through `8cd97de`):

- [x] `vnl_playground/eval_metrics/emg.py` — shared EMG metrics module (lagged corr, per-trial corr, edge-saturation) with 13 unit tests passing.
- [x] `train_mouse_janelia_sigmoid_moving_shoulder.py` — uses shared module, logs 17 EMG metrics per muscle every eval cycle, exposes `--emg-norm-percentile` (default 100).
- [x] `scripts/emg_comparison.py` — same metric parity + `--output-json` for Stage 2.
- [x] `scripts/s15_stage2_eval.sh` — 8-checkpoint re-evaluation driver (~3 h).

No additional prerequisites.

## Interactive Job 1 — 2 GPUs ericmmimic2

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_1.sh > /tmp/sweep_s15_ms_1_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s15_ms_2.sh > /tmp/sweep_s15_ms_2_master.log 2>&1 &
```

## Interactive Job 2 — 2 GPUs vastlrn

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_3.sh > /tmp/sweep_s15_ms_3_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s15_ms_4.sh > /tmp/sweep_s15_ms_4_master.log 2>&1 &
```

## Interactive Job 3 — 1 GPU

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_5.sh > /tmp/sweep_s15_ms_5_master.log 2>&1 &
```

## Interactive Job 4 — 1 GPU

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s15_ms_6.sh > /tmp/sweep_s15_ms_6_master.log 2>&1 &
```

## Script summary

| GPU | Script | Cells | Priority-ordered cells (first 3 must-run, last 2 budget-permitting) |
|---|---|---:|---|
| Job1 GPU0 | sweep_s15_ms_1.sh | 5 | **A1** baselineA, **R1** bursty, **F5** fs1p05_d8e7, S5 shWeak_fs1p2, A5 anchorCstrong |
| Job1 GPU1 | sweep_s15_ms_2.sh | 5 | **S1** shWeak3e7, **F2** fs1p2_d9e7, **R2** smoothOnly, I3 fs1p3_weakSh, F1 fs1p0_d1p2 |
| Job2 GPU0 | sweep_s15_ms_3.sh | 5 | **A2** anchorCfs1p2, **F6** fs1p15_d9e7, **R5** lightPenalty, S2 shMid6e7, F4 fs0p9_d6e7 |
| Job2 GPU1 | sweep_s15_ms_4.sh | 5 | **I1** weakSh_bursty, **F3** fs1p3_d1p2, **S4** shStrong, R6 mildBursty, A3 s11goldilocks |
| Job3 GPU0 | sweep_s15_ms_5.sh | 5 | **A4** anchorAmid, **I2** interpAC_asym, **S6** shWeak_fs1p3, F7 fs1p25_d1p1, R3 s11style |
| Job4 GPU0 | sweep_s15_ms_6.sh | 5 | **S3** elbowStrong, **F8** fs1p4_d1p2, **I4** slow_weakSh, R4 noPenalty, I5 weakSh_mildBurst |
| **Total** |  | **30** | 18 must-run (first 3 × 6 GPUs) + 12 budget-permitting (last 2 × 6 GPUs) |

## Candidate table (all 30)

| ID | Label | fs | joint_damp | shoulder_damp | cc | cdc | Group |
|---|---|---|---|---|---|---|---|
| A1 | baselineA | 1.1 | 9e-7 | 9e-7 | 0.025 | 0.025 | anchor — s13 single-seed leader |
| A2 | anchorCfs1p2 | 1.2 | 1e-6 | 1e-6 | 0.035 | 0.0 | anchor — s13 anchor-C fs=1.2 |
| A3 | s11goldilocks | 1.0 | 5e-7 | 5e-7 | 0.05 | 0.1 | anchor — s11 d5em7 fs=1.0 |
| A4 | anchorAmid | 1.1 | 7e-7 | 7e-7 | 0.025 | 0.025 | anchor — mid-damping at leader fs |
| A5 | anchorCstrong | 1.2 | 1.2e-6 | 1.2e-6 | 0.025 | 0.025 | anchor — anchor-C + stronger damping |
| S1 | shWeak3e7 | 1.1 | 9e-7 | **3e-7** | 0.025 | 0.025 | shoulder — weak shoulder at leader |
| S2 | shMid6e7 | 1.1 | 9e-7 | **6e-7** | 0.025 | 0.025 | shoulder — mid shoulder |
| S3 | elbowStrong | 1.1 | **1.5e-6** | 6e-7 | 0.025 | 0.025 | shoulder — stiff elbow, weak shoulder |
| S4 | shStrong | 1.1 | 6e-7 | **1.5e-6** | 0.025 | 0.025 | shoulder — reverse: stiff shoulder, weak elbow |
| S5 | shWeak_fs1p2 | 1.2 | 1e-6 | **4e-7** | 0.025 | 0.025 | shoulder — weak shoulder at fs=1.2 |
| S6 | shWeak_fs1p3 | 1.3 | 1e-6 | **5e-7** | 0.025 | 0.025 | shoulder — weak shoulder at fs=1.3 |
| R1 | bursty | 1.1 | 9e-7 | 9e-7 | **0.0** | **0.05** | reward — sharp bursts allowed |
| R2 | smoothOnly | 1.1 | 9e-7 | 9e-7 | **0.05** | **0.0** | reward — magnitude penalty only |
| R3 | s11style | 1.1 | 9e-7 | 9e-7 | **0.05** | **0.1** | reward — s11 reward shaping |
| R4 | noPenalty | 1.1 | 9e-7 | 9e-7 | **0.0** | **0.0** | reward — no action penalty at all |
| R5 | lightPenalty | 1.1 | 9e-7 | 9e-7 | 0.01 | 0.02 | reward — light penalty |
| R6 | mildBursty | 1.1 | 9e-7 | 9e-7 | 0.015 | 0.035 | reward — mild bursty |
| F1 | fs1p0_d1p2 | **1.0** | 1.2e-6 | 1.2e-6 | 0.025 | 0.025 | fs×damp — slow-arm baseline |
| F2 | fs1p2_d9e7 | **1.2** | 9e-7 | 9e-7 | 0.025 | 0.025 | fs×damp — anchor-A damping at fs=1.2 |
| F3 | fs1p3_d1p2 | **1.3** | 1.2e-6 | 1.2e-6 | 0.025 | 0.025 | fs×damp — high fs + high damping |
| F4 | fs0p9_d6e7 | **0.9** | 6e-7 | 6e-7 | 0.025 | 0.025 | fs×damp — low fs + low damping |
| F5 | fs1p05_d8e7 | **1.05** | 8e-7 | 8e-7 | 0.025 | 0.025 | fs×damp — dense mid-point |
| F6 | fs1p15_d9e7 | **1.15** | 9e-7 | 9e-7 | 0.025 | 0.025 | fs×damp — dense leader sibling |
| F7 | fs1p25_d1p1 | **1.25** | 1.1e-6 | 1.1e-6 | 0.025 | 0.025 | fs×damp — fs=1.25 probe |
| F8 | fs1p4_d1p2 | **1.4** | 1.2e-6 | 1.2e-6 | 0.025 | 0.025 | fs×damp — high-reward fs + stiffer damp |
| I1 | weakSh_bursty | 1.1 | 9e-7 | **3e-7** | **0.0** | **0.05** | interaction — weak shoulder + bursty |
| I2 | interpAC_asym | 1.2 | 1e-6 | **5e-7** | 0.03 | 0.0125 | interaction — A/C interp + asym shoulder |
| I3 | fs1p3_weakSh | **1.3** | 9e-7 | **4e-7** | 0.025 | 0.025 | interaction — high fs + weak shoulder |
| I4 | slow_weakSh | 1.0 | 1.2e-6 | **5e-7** | 0.025 | 0.025 | interaction — slow arm + weak shoulder |
| I5 | weakSh_mildBurst | 1.1 | 9e-7 | **5e-7** | **0.0** | 0.025 | interaction — weak shoulder + mild bursty |

## Wallclock estimates

Per-run wall-clock at 800M steps with the new EMG metric overhead: ~3.0–3.5 h.
Per-GPU budget: **12 h** (override via `BUDGET_HOURS=<n>`).
Per-run estimate for the budget-check: **12600 s = 3.5 h** (override via `ESTIMATED_RUN_SECONDS=<s>`).

Expected completion per GPU: **3–4 cells**. The loop stops when the remaining budget is less than one estimated run — no dangling half-runs.

Total across 6 GPUs: **18–24 completed runs** out of the 30 scheduled.

## Anchor rationale (quick reference)

Starting-point baseline for every candidate is `--joint-armature 4e-10 --control-cost 0.025 --control-diff-cost 0.025` (s13 anchor-A). Every candidate varies 1–3 of (`force_scale`, `joint_damping`, `shoulder_damping`, `control_cost`, `control_diff_cost`) from this baseline with a specific hypothesis:

| Group | Hypothesis |
|---|---|
| **A1–A5** | Reproduce + slightly extend known frontier. A1 is the current single-seed leader (R=411, bcorr=0.70, tcorr=0.58 under old metrics). |
| **S1–S6** | Triceps phase might improve if shoulder damping is decoupled from elbow damping (s14 C7's shoulder_fs<elbow_fs hinted at this). |
| **R1–R6** | Reward shaping controls how sharp the policy's bursts can be. cc=0 lets peaks go high; cdc constrains rate-of-change. |
| **F1–F8** | Dense goldilocks fills over fs ∈ [0.9, 1.4] × damping ∈ [6e-7, 1.5e-6] — plug the gaps between anchors A and C. |
| **I1–I5** | Combine promising levers: shoulder-decoupling × reward-shaping × fs extremes. |

## Success criteria (quick reference)

**Primary ranking metric:** `min(biceps_lagged_corr_max, triceps_lagged_corr_max)` — weakest muscle wins.

**Ship gates (applied per single run):**
- `eval/episode_reward ≥ 400`
- `eval/emg_biceps_lagged_corr_max ≥ 0.80`
- `eval/emg_triceps_lagged_corr_max ≥ 0.80`
- `eval/emg_biceps_trial_corr_mean ≥ 0.5` AND `eval/emg_triceps_trial_corr_mean ≥ 0.5`
- `|eval/emg_biceps_phase_lag_ms| ≤ 20` AND `|eval/emg_triceps_phase_lag_ms| ≤ 20`
- `eval/emg_biceps_mae ≤ 0.15` AND `eval/emg_triceps_mae ≤ 0.15`

**If ≥ 1 cell passes**, replicate it with 3 seeds in a follow-up (~18 GPU-h) to confirm.

**Watch flags during training (wandb live):**
- `eval/emg_*_lagged_corr_edge_saturated` — if it's 1, the true lag is outside ±50 ms, and the reported `lagged_corr_max` underestimates.
- `eval/emg_*_per_trial_phase_lag_std_ms` — trial-to-trial timing jitter.

## Crash policy

- Each `sweep_s15_ms_N.sh` uses `set -o pipefail` and logs both stdout and stderr per run. A crashed training step records the run to `CRASHED`; the loop continues to the next cell so a single failure doesn't take down the remaining candidates.
- To re-run only the skipped or crashed cells after the batch completes, copy the relevant `run_cell ...` line from the appropriate `sweep_s15_ms_N.sh` and invoke it on a fresh GPU.

## Optional pre-sweep: Stage 2 re-evaluation (~3 h, no training)

If you want to see how the existing s13/s14 leaders rank under the new metrics before committing to the sweep:

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash scripts/s15_stage2_eval.sh > /tmp/s15_stage2.log 2>&1 &
```

Outputs `plots/2026-04-23-s15-stage2/eval_matrix.csv` with 32 rows (8 checkpoints × 2 muscles × 2 percentiles). Can be run on one of the idle GPUs before the main sweep or in parallel on a different machine. Pure eval, no training.
