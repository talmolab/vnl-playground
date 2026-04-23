# S15-MS Launch Instructions

**Goal:** sweep over the s13 goldilocks region under the new EMG metric pipeline and confirm a cell that clears `lagged_corr_max ≥ 0.80` on both muscles with `|phase_lag_ms| ≤ 20`, `R ≥ 400`, and `mae ≤ 0.15`.

Full spec: `docs/superpowers/specs/2026-04-23-s15-ms-design.md`
Implementation plan: `docs/superpowers/plans/2026-04-23-s15-ms-implementation.md`

## What is already done (Stage 1 — code changes, committed)

| commit | what |
|---|---|
| `5cc0553` | scaffold shared EMG metrics module |
| `0b9e36b` | implement compute_lag_metrics |
| `d56891a` | implement compute_per_trial_metrics |
| `2d4ba8e` | add compute_all_emg_metrics unified entry point |
| `8a04327` | factor _lag_scan and add edge-saturation diagnostic |
| `dbee742` | add `--emg-norm-percentile` CLI flag (default 100) + prior uncommitted trainer edits* |
| `dd1b851` | wire trainer eval loop to shared EMG metrics module |
| `8cd97de` | wire emg_comparison.py + add `--emg-norm-percentile` and `--output-json` |
| `5079e8e` | add Stage 2 driver script `scripts/s15_stage2_eval.sh` |

\* `dbee742` also absorbed the per-muscle tau CLI args, `clip_start_frame` threading, and S9-redo defaults (d=8e-7, fs=1.0) that were uncommitted on the branch when Stage 1 started. Code is correct; if the commit history needs splitting, do it before merging.

The trainer now logs 17 EMG metrics per muscle per eval cycle, including `lagged_corr_max`, `phase_lag_ms`, `phase_lag_steps`, `lagged_corr_edge_saturated`, `trial_corr_mean/median`, `per_trial_lagged_corr_mean/median`, `per_trial_phase_lag_mean_ms/std_ms`.

## Recommended order

### 1. Stage 2 first (optional but high-value, ~3 h GPU, no training)

Re-evaluate 8 frontier checkpoints (s13 / s14 / s12 / s11 / s10) with new metrics at both p98 (legacy) and p100 (new). Tells us which branch of Stage 3 is correct and whether the current leader already clears gates.

```bash
cd /root/vast/eric/vnl-playground
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/s15_stage2_eval.sh > /tmp/s15_stage2.log 2>&1 &
```

Outputs: `plots/2026-04-23-s15-stage2/eval_matrix.csv` (32 rows = 8 ckpts × 2 muscles × 2 percentiles) and 16 JSON + 16 log files.

Inspect with:
```bash
cd /root/vast/eric/vnl-playground
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('plots/2026-04-23-s15-stage2/eval_matrix.csv')
df['min_lagged'] = df.groupby(['checkpoint','norm_pct'])['lagged_corr_max'].transform('min')
df['max_mae']    = df.groupby(['checkpoint','norm_pct'])['mean_mae'].transform('max')
df['max_lag_ms'] = df.groupby(['checkpoint','norm_pct'])['phase_lag_ms'].transform(lambda s: s.abs().max())
print(df.drop_duplicates(['checkpoint','norm_pct'])
        [['checkpoint','norm_pct','min_lagged','max_mae','max_lag_ms']]
        .sort_values(['checkpoint','norm_pct'])
        .to_string(index=False))
"
```

Decision tree:
- **any row at p100 with min_lagged ≥ 0.80, max_mae ≤ 0.15, max_lag_ms ≤ 20** → Branch 1, replicate that cell. Edit `sweep_s15_ms_branch1.sh` if the winner isn't anchor-A fs=1.1.
- **min_lagged ≥ 0.80 but max_lag_ms > 20** → Branch 2, no retraining needed; lag is a fixed reference offset. Document in a Stage 2 report.
- **p100 doesn't raise min_lagged to 0.80** → Branch 3, wider grid (add shoulder-damping scan — see below).
- **per-trial normalization probe improves p100 min_lagged by ≥ 0.10** → Branch 4, implement `--emg-norm-mode per_trial` and retrain.

If you choose to skip Stage 2 and go straight to Branch 1 (training), the sweep below is still the right default: it gives 5 seeds on the current single-run leader plus bracketing cells to absorb seed variance.

### 2. Stage 3 Branch 1 launch (default, 15 runs on your 6-GPU / 4-job topology)

3 cells × 5 seeds = 15 runs. Each run ~3–3.5 h at 800M steps (the expanded EMG metric computation adds CPU-side work during each eval cycle, so slower than s13's ~1 h).

**Your layout: 2× 2-GPU jobs + 2× 1-GPU jobs = 6 GPUs across 4 job slots.**
Partition: each 2-GPU job owns one fs cell and runs 5 seeds across its 2 GPUs (3 seeds on GPU0, 2 seeds on GPU1). The 2 single-GPU jobs split the third cell's 5 seeds (3 + 2). Max per-GPU wall-clock = **3 × ~3.5 h ≈ 10–11 h**, fitting the 12 h budget.

| Job | GPUs | Cell (fs) | Seeds on GPU0 | Seeds on GPU1 | Per-GPU runs |
|---|---|---|---|---|---|
| Job 1 (2-GPU) | 0, 1 | **fs=1.0** | 1, 2, 3 | 4, 5 | 3 / 2 |
| Job 2 (2-GPU) | 0, 1 | **fs=1.1** | 1, 2, 3 | 4, 5 | 3 / 2 |
| Job 3 (1-GPU) | 0 | **fs=1.2** | 1, 2, 3 | — | 3 |
| Job 4 (1-GPU) | 0 | **fs=1.2** | 4, 5 | — | 2 |

**Launch commands (run each pair in its own shell on the appropriate job).**

Job 1 — 2-GPU machine #1, fs=1.0:
```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 CELLS_TO_RUN="1.0" SEEDS_TO_RUN="1 2 3" nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p0_gpu0.log 2>&1 &
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 CELLS_TO_RUN="1.0" SEEDS_TO_RUN="4 5"   nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p0_gpu1.log 2>&1 &
```

Job 2 — 2-GPU machine #2, fs=1.1:
```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 CELLS_TO_RUN="1.1" SEEDS_TO_RUN="1 2 3" nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p1_gpu0.log 2>&1 &
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 CELLS_TO_RUN="1.1" SEEDS_TO_RUN="4 5"   nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p1_gpu1.log 2>&1 &
```

Job 3 — 1-GPU machine #1, fs=1.2 seeds 1–3:
```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 CELLS_TO_RUN="1.2" SEEDS_TO_RUN="1 2 3" nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p2_a.log 2>&1 &
```

Job 4 — 1-GPU machine #2, fs=1.2 seeds 4–5:
```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 CELLS_TO_RUN="1.2" SEEDS_TO_RUN="4 5"   nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p2_b.log 2>&1 &
```

Monitor: `tail -f /tmp/s15_ms_branch1_*.log` on each machine.

**Fallback — fewer GPUs available tonight.** If only 3 GPUs are free, drop fs=1.2 (it's the bracketing sibling, not the leader) and run 2 cells × 5 seeds = 10 runs on 3 GPUs. With ~3.5 h per run, GPU0-with-3-runs finishes in ~10.5 h; GPU1-with-2-runs in ~7 h; 1-GPU slot with 5 serial runs in ~17.5 h (over budget — trim to 3 seeds there):
```bash
# 2-GPU job: fs=1.1 (leader), 5 seeds split 3/2
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 CELLS_TO_RUN="1.1" SEEDS_TO_RUN="1 2 3" nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p1_gpu0.log 2>&1 &
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 CELLS_TO_RUN="1.1" SEEDS_TO_RUN="4 5"   nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p1_gpu1.log 2>&1 &
# 1-GPU job: fs=1.0 (low sibling), 3 seeds serial (~10.5 h)
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 CELLS_TO_RUN="1.0" SEEDS_TO_RUN="1 2 3" nohup bash sweep_s15_ms_branch1.sh > /tmp/s15_ms_branch1_fs1p0.log 2>&1 &
```

**Stage 2 timing caveat.** At ~20 min per eval-replay run × 16 runs, Stage 2 is still ~3 h. If you run it on one of the 1-GPU slots *in parallel* with the sweep, that slot's sweep share (fs=1.2 seeds 1-3) starts after Stage 2 finishes — adding ~3 h to that slot's wall-clock, pushing it to ~13.5 h. If the 12 h ceiling is firm, either (a) run Stage 2 first and defer the sweep, or (b) skip Stage 2 tonight and launch all 4 job slots into the sweep directly (the bracketing fs=1.0/1.1/1.2 covers branch uncertainty).

### 3. Success criteria (ship gate)

Median across ≥ 3 successful seeds on any cell:

- `eval/episode_reward ≥ 400`
- `eval/emg_biceps_lagged_corr_max ≥ 0.80`
- `eval/emg_triceps_lagged_corr_max ≥ 0.80`
- `eval/emg_biceps_trial_corr_mean ≥ 0.5`
- `eval/emg_triceps_trial_corr_mean ≥ 0.5`
- `|eval/emg_biceps_phase_lag_ms| ≤ 20`
- `|eval/emg_triceps_phase_lag_ms| ≤ 20`
- `eval/emg_biceps_mae ≤ 0.15`
- `eval/emg_triceps_mae ≤ 0.15`

Primary ranking metric (if multiple cells pass): `min(biceps_lagged_corr_max, triceps_lagged_corr_max)` — weakest muscle wins.

### 4. If Branch 3 is needed instead (Stage 2 said shape caps)

Regenerate the sweep with a shoulder-damping mini-scan on top of the replication:
- Keep `sweep_s15_ms_branch1.sh` as-is for the 10-run replication portion.
- Ask Claude to generate `sweep_s15_ms_branch3b_shoulder.sh` (4 shoulder-damping levels × 3 seeds = 12 runs at `--force-scale 1.1 --shoulder-damping {3e-7, 6e-7, 9e-7, 1.2e-6}`).
- Total 22 runs. On 4 GPUs, ~11 h. Fits budget.

## Monitoring during training

Each eval cycle logs 17 EMG metrics per muscle to wandb under `eval/emg_<muscle>_*`. The ones to watch live (wandb panels you may want to add):

- `eval/emg_biceps_lagged_corr_max` and `eval/emg_triceps_lagged_corr_max` — headline metric.
- `eval/emg_biceps_phase_lag_ms` and `eval/emg_triceps_phase_lag_ms` — should converge near 0.
- `eval/emg_biceps_lagged_corr_edge_saturated` and `eval/emg_triceps_lagged_corr_edge_saturated` — if these are ever 1, the true lag is outside ±50 ms and the reported `lagged_corr_max` is an underestimate.
- `eval/emg_biceps_per_trial_phase_lag_std_ms` — trial-to-trial timing jitter. High (> 10 ms) at convergence means the policy is inconsistent across trials.

## Known limitations / caveats

- Seed variance in s13 was large (median bcorr std = 0.20). 5-seed median is the right rollup; 3-seed is borderline.
- `lagged_corr_max` absorbs phase error, so shape and timing are disentangled. If `lagged_corr_max ≥ 0.80` but `mean_corr ≤ 0.5`, the policy has the right shape at the wrong time — that's still a win on shape, and Branch 2 calls it.
- The pre-existing tau overrides + S9-redo defaults in `dbee742` are now baseline — any comparison against historical s13 runs using the same `--joint-damping 9e-7 --force-scale 1.1` args will still reproduce the leader cell.

## After training completes

Task 13 in the plan — pull s15 runs from wandb, apply gates, write `docs/superpowers/specs/2026-04-23-s15-ms-results.md`. I can run that once the sweep finishes.
