# S19-MS Launch Commands — Bayesian framework + multi-seed σ² anchor + γ regime revisits

**9 cells (10 with bonus)** across **6 scripts**, **6 GPUs**, partitioned as 2× 2-GPU jobs + 2× 1-GPU jobs. **Time-box: ~10 h wall** (BUDGET_HOURS=10 default).

**Tri-track week:**
- **Track 1 (CPU/code, parallel):** Build Bayesian framework MVP M3 in `vnl_playground/bayesian_emg/` from existing s17+s18 cache.
- **Track 2a (4 GPU runs):** Multi-seed σ² anchor at C1 (cc=0, cdc=0, fs=1.1, p98_per_muscle) — seeds 1, 2, 3, 4. Combined with seed 0 from s18, gives 5 seeds at C1.
- **Track 2b (5 GPU runs):** γ revisits — re-train s10/s11/s13/s15/s16 historical leaders under cohort + p98_per_muscle norm. Each cell is a known qualitatively-distinct motor mode.
- **Bonus (1 GPU run):** A1.s5 (5th anchor seed) — strengthens σ² estimator from 5 to 6 seeds. Optional.

Spec & rationale: `docs/superpowers/specs/2026-05-02-s19-ms-bayesian-population-design.md`.

## Pre-launch checklist

- [x] s17 + s18 results in CSV form (`s18.csv`, `s17_s18.csv`); winner under (AD, biceps) min identified as C1.
- [x] Trainer infra unchanged from s17/s18: `--reference-data-path`, `--train-animals` (default = all 5), `--emg-animals`, `--emg-norm-method`, `--emg-norm-percentile`, `--muscle-tau-act` already in `train_mouse_janelia_sigmoid_moving_shoulder.py`.
- [x] Reference clip dir exists at `vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/` (278 symlinks, 5 animals).
- [x] s18 C1 cell already converged at seed 0 (eval/episode_reward=442.7) — confirms the anchor cell trains.

No pre-flight needed for anchor seeds (C1 already known to converge from s18). γ cells run an automatic 50M-step preflight gate inside the sweep script (`PREFLIGHT_REWARD_FLOOR=250` default; override via env var).

## Interactive Job 1 — 2 GPUs (machine #1) ericmimic2

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s19_ms_1.sh > /tmp/sweep_s19_ms_1_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s19_ms_2.sh > /tmp/sweep_s19_ms_2_master.log 2>&1 &
```

## Interactive Job 2 — 2 GPUs (machine #2) vastlrn

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s19_ms_3.sh > /tmp/sweep_s19_ms_3_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s19_ms_4.sh > /tmp/sweep_s19_ms_4_master.log 2>&1 &
```

## Interactive Job 3 — 1 GPU (machine #3) ericmimic

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s19_ms_5.sh > /tmp/sweep_s19_ms_5_master.log 2>&1 &
```

## Interactive Job 4 — 1 GPU (machine #4) ericemgdata (BONUS — optional)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s19_ms_6.sh > /tmp/sweep_s19_ms_6_master.log 2>&1 &
```

## Script summary

| GPU | Script | Cells | Cells (priority order) |
|---|---|---:|---|
| Job1 GPU0 | `sweep_s19_ms_1.sh` | 2 | **A1.s1** anchor seed 1 at C1, **γ1** s10 fs=0.5 (preflight) |
| Job1 GPU1 | `sweep_s19_ms_2.sh` | 2 | **A1.s2** anchor seed 2 at C1, **γ2** s11-goldi (preflight) |
| Job2 GPU0 | `sweep_s19_ms_3.sh` | 2 | **A1.s3** anchor seed 3 at C1, **γ3** s13-anchorA (preflight) |
| Job2 GPU1 | `sweep_s19_ms_4.sh` | 2 | **A1.s4** anchor seed 4 at C1, **γ4** s15-F1 (preflight) |
| Job3 GPU0 | `sweep_s19_ms_5.sh` | 1 | **γ5** s16-T1f tau=40ms (preflight) |
| Job4 GPU0 | `sweep_s19_ms_6.sh` | 1 | **A1.s5** bonus seed (optional) |
| **Total** |  | **9 + 1 bonus** |  |

## Cell table

### Track 2a — multi-seed σ² anchor (4 cells, no preflight)

All cells: cc=0, cdc=0, fs=1.1, jd=1.5e-6, sd=6e-7, percentile=98, p98_per_muscle norm.

| ID | Seed | Notes |
|---|---|---|
| **A1.s1** | 1 | new |
| **A1.s2** | 2 | new |
| **A1.s3** | 3 | new |
| **A1.s4** | 4 | new |
| (A1.s5) | 5 | bonus, script 6 |
| (A1.s0) | 0 | already exists from s18 — `s18-ms-C1-cc0-cdc0` |

### Track 2b — γ regime revisits (5 cells, all with preflight)

All cells: cohort training, p98_per_muscle norm, percentile=98.

| ID | Origin | fs | jd | sd | cc | cdc | tau (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| **γ1** | s10 d9em7-fs0p5 | **0.5** | 9e-7 | 9e-7 | 0.05 | 0.10 | 25 (default) | very low fs / low effort |
| **γ2** | s11-goldi (B06) | 1.0 | **5e-7** | **5e-7** | 0.05 | 0.10 | 25 (default) | low damping symmetric |
| **γ3** | s13-anchorA | 1.1 | 9e-7 | 9e-7 | 0.025 | 0.025 | 25 (default) | mid damping symmetric |
| **γ4** | s15-F1 | 1.0 | 1.2e-6 | **1.2e-6** | 0.025 | 0.025 | 25 (default) | coupled-equal damping |
| **γ5** | s16-T1f | 1.1 | 1.5e-6 | 6e-7 | 0.025 | 0.025 | **40** (override) | extended tau |

## Wallclock estimates

Per-run wall-clock at 800M steps with 5-animal EMG eval overhead: **~3.5–4.0 h**. Per-cell budget estimate: **14400 s = 4.0 h** (override via `ESTIMATED_RUN_SECONDS=<s>`). Preflight: 50M steps ≈ 0.5 h; budget guard reserves `PREFLIGHT_SECONDS=2400` (40 min) on top.

| Job type | Cells | Per-GPU budget | Expected complete |
|---|---:|---:|---|
| 2-cell scripts (s19_1–s19_4) | 2 | 10 h | 1–2 cells per GPU |
| 1-cell scripts (s19_5, s19_6) | 1 | 10 h | 1 cell per GPU |

Override: `BUDGET_HOURS=20 bash sweep_s19_ms_N.sh` to extend.

## Preflight gate

γ cells trigger an automatic 50M-step pilot at the cell's hyperparameters (no wandb). Final `eval/episode_reward` is parsed from the log; if `< 250` the full 800M run is skipped and the cell is recorded to `PREFLIGHT_FAILED`. Override via `PREFLIGHT_REWARD_FLOOR=<x>` env var. If the metric can't be parsed (regex fails), the script proceeds to the full run with a warning.

The anchor seed cells (A1.s1–s5) skip preflight — C1 already converged at seed 0 in s18 (reward 442.7).

## Success criteria (quick reference)

**Headline metric:** `min(eval/emg_cohort_AD_corr, eval/emg_cohort_biceps_corr)` (cohort-mean over 5 animals; **triceps deferred** — anatomical AD + biceps drive the reach).

**Per-cell acceptance gates:**
- `eval/episode_reward ≥ 350`
- `eval/emg_cohort_biceps_corr ≥ 0.4`
- `eval/emg_cohort_AD_corr ≥ 0.3` (s18 C1 baseline = 0.65)
- `eval/emg_cohort_<muscle>_trial_mae ≤ 0.3` for AD, biceps

**Anchor seed σ² gate (Track 2a output):**
- All 4 new seeds at C1 must complete; combined with s18 seed-0, framework computes per-(mouse, muscle) σ² across 5 seeds.
- σ² is then pinned in `configs/bayesian_emg/preregistration.yaml` (cache-frozen before final report).

**Watch flags during training (wandb live):**
- `eval/emg_cohort_<muscle>_corr` time-series — rises smoothly; sharp drops mean policy collapse.
- `latent_kl_*` — same intention-network diagnostics as s15/s16/s17/s18.

## Crash policy

- Each `sweep_s19_ms_N.sh` uses `set -o pipefail` and logs both stdout and stderr per run. A crashed training step records the run to `CRASHED`; the loop continues to the next cell.
- A failed preflight records the cell to `PREFLIGHT_FAILED` and skips the full run; the loop continues.
- To re-run a failed cell, copy the relevant `run_cell` or `preflight_then_full` line from the appropriate script and invoke it on a fresh GPU.

## Post-sweep deliverables

1. **σ² report (Track 2a)** — per-(mouse, muscle) across-seed scatter at C1 from 5 seeds (s18 seed 0 + s19 seeds 1–4). 5 mice × 2 muscles (AD, biceps) = 10 σ² estimates.
2. **γ-cell ranking (Track 2b)** — 5 γ cells ranked by `cohort_min(AD, biceps)_corr`. Confirms or refutes the prediction that γ1 (fs=0.5) recovers single-animal-era performance.
3. **Bayesian framework Phase 1 report** — cross-mouse 5×5 discrimination matrix + permutation p-value + within-mouse coverage curve + UCM alignment heatmap, on s17+s18+s19 cache.
4. **Pre-registration YAML** — `configs/bayesian_emg/preregistration.yaml` with cache hash committed.

## Out of scope (deferred to s20+)

- Multi-seed at γ cells (s20 if a γ cell wins).
- Triceps debugging (deferred per task definition).
- ABC likelihood (Bayesian framework Phase 2).
- Gaussian envelope likelihood with DTW (Phase 3).
- Bayes factors for sweep-design comparisons (post Phase 1).
- Per-animal specialists, animal-conditioned policies.
- EMG in the reward.
