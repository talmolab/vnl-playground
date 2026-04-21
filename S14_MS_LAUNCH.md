# S14-MS Launch Commands

**96 runs** across **6 scripts**, **6 GPUs**, organized as 2× 2-GPU jobs + 2× 1-GPU jobs.
All cells use the moving-shoulder XML (`mouse_forelimb_right_moving_shoulder_ik.xml`, trainer default).

Design: per-muscle biceps-vs-triceps force-scale ratio ladder + coupled diagonal + lagged cross-correlation EMG metric, at 2 well-performing damping anchors. Full rationale in
`docs/superpowers/specs/2026-04-21-s14-ms-per-muscle-fs-ratio-design.md`.

## Pre-launch checklist (prerequisites P1–P3)

Do NOT launch the sweeps until these are complete:

- [ ] **P1.** `compute_emg_metrics()` in `train_mouse_janelia_sigmoid_moving_shoulder.py` adds `compute_lagged_corr()` and logs `eval/emg_{muscle}_lagged_corr`, `eval/emg_{muscle}_phase_lag_ms`, and per-trial medians (`..._trial_lagged_corr_median`, `..._trial_phase_lag_ms_median`). Existing `eval/emg_{muscle}_corr` unchanged.
- [ ] **P2.** Unit test (`tests/test_lagged_corr.py`) asserts identity, anti-correlation, and a known-shift fixture; sign convention for `phase_lag_steps` documented as "positive = sim leads EMG".
- [ ] **P3.** Backfill s13 (30 finished runs) with the updated eval pipeline → `s13_with_lagged.csv`. Confirms `phase_lag_ms` distribution fits within ±50 ms (max_lag_steps=20 @ ctrl_dt=2.5 ms); if >5% saturate, widen window and rerun backfill first.

## Per-muscle force overrides — quick reference

Every cell sets `--force-scale 1.3` (shoulder fs) and pushes per-muscle biceps/triceps via absolute overrides computed in-script:

```
--biceps-force     = 0.1 * b_eff / 1.3
--brachialis-force = 0.1 * b_eff / 1.3
--triceps-long-force = 0.1 * t_eff / 1.3
--triceps-lat-force  = 0.1 * t_eff / 1.3
```

Scripts precompute these via `python3 -c "print(0.1 * $x / 1.3)"` per call and echo the resolved values into the per-cell log, so anything that ends up on-disk has the exact args used.

## Interactive Job 1 — 2 GPUs (anchor A, coupled + low-t zones)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s14_ms_1.sh > /tmp/sweep_s14_ms_1_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s14_ms_2.sh > /tmp/sweep_s14_ms_2_master.log 2>&1 &
```

## Interactive Job 2 — 2 GPUs (anchor A core + anchor C coupled)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s14_ms_3.sh > /tmp/sweep_s14_ms_3_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_s14_ms_4.sh > /tmp/sweep_s14_ms_4_master.log 2>&1 &
```

## Interactive Job 3 — 1 GPU (anchor C low-t)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s14_ms_5.sh > /tmp/sweep_s14_ms_5_master.log 2>&1 &
```

## Interactive Job 4 — 1 GPU (anchor C core)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_s14_ms_6.sh > /tmp/sweep_s14_ms_6_master.log 2>&1 &
```

## Script summary

| GPU | Script | Anchor | Cells | Runs | Zone |
|---|---|---|---|---:|---|
| Job1 GPU0 | `sweep_s14_ms_1.sh` | A (d=9e-7) | C1–C8 × 2 seeds | 16 | Coupled diagonal |
| Job1 GPU1 | `sweep_s14_ms_2.sh` | A (d=9e-7) | E1–E6, L0, F1 × 2 seeds | 16 | Low-t asym + sym + falsifier |
| Job2 GPU0 | `sweep_s14_ms_3.sh` | A (d=9e-7) | L1–L8 × 2 seeds | 16 | Asymmetric core |
| Job2 GPU1 | `sweep_s14_ms_4.sh` | C (d=1e-6) | C1–C8 × 2 seeds | 16 | Coupled diagonal |
| Job3 GPU0 | `sweep_s14_ms_5.sh` | C (d=1e-6) | E1–E6, L0, F1 × 2 seeds | 16 | Low-t asym + sym + falsifier |
| Job4 GPU0 | `sweep_s14_ms_6.sh` | C (d=1e-6) | L1–L8 × 2 seeds | 16 | Asymmetric core |
| **Total** | | | | **96** | |

## Wallclock estimates

16 cells × ~1 hr / stream at 800M steps. All 6 streams parallel → **~16 h total wallclock**. Overnight-and-a-bit.

## Ladder quick reference (identical across both anchors)

| Zone | Cells | `(t_eff, b_eff)` |
|---|---|---|
| Coupled diagonal | C1–C8 | (0.7,0.7), (0.8,0.8), (0.9,0.9), (1.1,1.1), (1.2,1.2), (1.3,1.3), (1.4,1.4), (1.5,1.5) |
| Symmetric baseline | L0 | (1.0, 1.0) |
| Low-t asymmetric | E1–E6 | (0.7,1.0), (0.7,1.3), (0.8,1.0), (0.8,1.3), (0.9,1.0), (0.9,1.3) |
| Asymmetric core | L1–L8 | (1.0,1.1), (1.0,1.2), (1.0,1.3), (1.0,1.4), (1.1,1.3), (1.1,1.4), (1.2,1.4), (1.2,1.5) |
| Reverse falsifier | F1 | (1.3, 1.0) |

**Cross-sweep anchor:** C6 = (1.3, 1.3) is mathematically identical to a coupled fs=1.3 run (override=0.1 = XML default, force_scale=1.3). Expect its R/bcorr/tcorr to sit inside the s13 anchor-X fs=1.3 seed range.

## Anchor rationale (quick reference)

| anchor | damp | cc | cdc | s12/s13 role |
|---|---|---|---|---|
| A | 9e-7 | 0.025 | 0.025 | s12 near-R-bar regime (R=397 at fs=1.0); s13 R=415–438 at fs=1.3 |
| C | 1e-6 | 0.035 | 0.0   | s12 surrogate-predicted composite optimum (d=1e-6 family); s13 R=413–431 at fs=1.3 |

Anchor B (d=3e-7) **dropped** — low-damping family was not on the composite frontier in s12/s13.

## Success criteria (quick reference)

**Winner (any single seed):**
- `eval/episode_reward ≥ 400`
- `eval/emg_biceps_corr ≥ 0.70`
- `eval/emg_triceps_corr ≥ 0.60`
- `eval/emg_biceps_mae ≤ 0.15` AND `eval/emg_triceps_mae ≤ 0.15`
- `|eval/emg_biceps_phase_lag_ms| ≤ 10` AND `|eval/emg_triceps_phase_lag_ms| ≤ 10`

Winner replicates in s15 with ≥ 3 seeds.

**Decisive test for the ratio hypothesis:** any asymmetric cell's composite (reward, `min(bcorr, tcorr)`, `max(|lag|)`) strictly beats **both** of its coupled neighbors — e.g., L3 = (1.0, 1.3) beats both L0 = (1.0, 1.0) and C6 = (1.3, 1.3). If the entire coupled diagonal is on the Pareto front, the hypothesis is rejected.

## Crash policy

- Single cell crash → log, continue.
- ≥ 2 consecutive crashes in one script → pause that GPU, investigate (most likely actuator saturation at high coupled fs, or NaN reward).
- **Highest risk:** coupled (1.5, 1.5) (C8) at anchor A — largest cumulative gain; watch for NaN reward or early termination.
- **Shoulder-operating-point regression:** if seed-1 L0 at any anchor drops below R=400, that anchor's `fs_shoulder=1.3` is wrong — pause that anchor's remaining scripts and pick a different shoulder fs.

## Monitoring

Per-cell logs at `/tmp/sweep_s14_ms_{tag}.log`. Master script logs at `/tmp/sweep_s14_ms_{1..6}_master.log`.
Wandb groups: `s14-ms-part{1..6}`. Cross-cutting tag: `s14-ms`.

Wandb filter URL (after first run appears):
`https://wandb.ai/<user>/<project>?workspace=tag:s14-ms`

Useful sub-filters:
- `tag:s14-ms AND tag:anchorA` — all 48 anchor-A runs
- `tag:s14-ms AND tag:coupled` — coupled diagonal (32 runs) for the no-ratio Pareto plot
- `tag:s14-ms AND tag:asymmetric` — asymmetric cells only (32 runs)
- `tag:s14-ms AND tag:L3` — central-prediction cell, across both anchors + both seeds (4 runs)

## Follow-up (s15)

Based on scan results:
- **Winner cell found.** Replicate with 3 seeds at that (anchor, t_eff, b_eff).
- **Asymmetric cell beats coupled everywhere.** Densify 2D `(t_eff, b_eff)` grid around the winning ratio at the better anchor.
- **Coupled diagonal is Pareto-optimal.** Ratio hypothesis rejected; return to mechanism search (muscle tau, phase-aligned EMG reward, longer training).
- **Low-t zone (E1–E6, C1–C3) outperforms high-t.** s14 initial expectation was wrong; s15 densifies t ∈ [0.7, 1.0].
- **Big phase-lag reduction but no bar crossed.** Add an EMG phase-alignment reward term; rerun at best anchor.
