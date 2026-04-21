---
name: s13-ms-force-scale-above-one-design
description: Design for s13_ms — test force_scale>1.0 at three s12-derived anchors, add per-trial EMG correlation metric, and backfill s10/s12 for threshold calibration
type: project
---

# s13-ms: force_scale > 1.0 sweep with per-trial EMG correlation

**Date:** 2026-04-21
**Status:** design

## Goal

Find an s13_ms configuration that simultaneously crosses `eval/episode_reward ≥ 400` AND meaningful EMG shape fidelity measured by **per-trial** correlation (not mean-trace correlation). s12 capped at R=398 (`armA-d9em7-fs1p0-cc0p025-cdc0p00`); nothing in s12 crossed R=400.

## Hypotheses

1. **Force-scale hypothesis.** s12's R=400 ceiling was reward-limited, not shape-limited. Extrapolating `force_scale` past 1.0 should push reward over 400 at high-damp anchors without collapsing mean-trace EMG correlation. The `fs × damp` landscape plot (`plots_s12/09_fs_damp_landscape.png`) has everything at `fs > 1.0` empty — this sweep fills it.

2. **Measurement hypothesis.** The `eval/emg_*_corr` metric that s10/s11/s12 were ranked on is **mean-trace** correlation — `np.corrcoef(sim_muscle.mean(axis=0), emg_mean_trace)`. It rewards phase-lock of the *trial-averaged* traces and ignores whether individual trials actually match their individual EMG. Per-trial correlation (Pearson r computed within each trial, then aggregated) is scientifically more meaningful. s13 promotes per-trial trial_corr to primary success metric.

3. **Backfill hypothesis.** s13's trial_corr threshold can only be calibrated against a baseline of s10+s12 runs evaluated on the same metric. Without backfill, any s13 bar would be pulled from thin air.

## Prerequisites (pre-launch, gating)

**P1. Metric change.** In `train_mouse_janelia_sigmoid_moving_shoulder.py`, extend `compute_emg_metrics()` (around line 159-171) to add:

```python
# inside compute_emg_metrics, after existing mean_corr computation
if bio_traces is not None:
    per_trial_corrs = [
        np.corrcoef(sim_muscle[i], bio_traces[i])[0, 1]
        for i in range(sim_muscle.shape[0])
    ]
    result["trial_corr_mean"]   = float(np.nanmean(per_trial_corrs))
    result["trial_corr_median"] = float(np.nanmedian(per_trial_corrs))
```

Log in the eval loop (around line 2020-2034):

```python
wandb_log[f"eval/emg_{muscle_name.lower()}_trial_corr"]        = m["trial_corr_mean"]
wandb_log[f"eval/emg_{muscle_name.lower()}_trial_corr_median"] = m["trial_corr_median"]
```

**Keep** existing `eval/emg_{muscle}_corr` unchanged for back-compat with s10/s11/s12 CSVs.

**P2. Mirror in eval-replay script.** Apply the same change to `scripts/emg_comparison.py` (or equivalent eval-replay path) so the backfill uses the same computation as live training.

**P3. Backfill.** Re-evaluate 46 s12 checkpoints + ~40 s10 checkpoints from `checkpoints/` using the updated eval. Produces `s12_s10_with_trial_corr.csv`.

**P4. Threshold calibration.** From the backfill CSV, compute s10+s12 distribution of `emg_biceps_trial_corr` and `emg_triceps_trial_corr`. Set s13 success bars at the p75 of the union, rounded to the nearest 0.05.

## Anchors

Chosen from the s10+s11+s12 pool by the criterion "closest to R=400 while strong on mean-trace composite, one per damping regime."

| anchor | source | damp | cc | cdc | R | bcorr | tcorr | bmae | tmae | rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| A | s12 | 9e-7 | 0.025 | 0.025 | 397 | 0.53 | 0.53 | 0.14 | 0.12 | best-balanced composite at R≈400 boundary |
| B | s11 | 3e-7 | 0.025 | 0.05  | 410 | 0.56 | 0.45 | 0.16 | 0.14 | top composite among R≥400 runs |
| C | s12 | 1e-6 | 0.035 | 0.0   | 377 | 0.65 | 0.72 | 0.13 | 0.11 | d=1e-6 regime (bcorr-frontier family); closest to R=400 in that regime |

Each anchor fixes `(damp, cc, cdc)` and varies only `force_scale` along the ladder.

## Swept axis

`force_scale ∈ {1.1, 1.2, 1.3, 1.4, 1.5}` — five ticks per anchor. Existing s12/s11 data provides the fs=1.0 baseline.

## Arm structure

**Single arm (M — main fs ladder, qvel=zeros):** 3 anchors × 5 fs = **15 cells, 1 seed each**.

| anchor | fs values | qvel | seed |
|---|---|---|---|
| A (d9e-7, cc0.025, cdc0.025) | 1.1, 1.2, 1.3, 1.4, 1.5 | zeros | 1 |
| B (d3e-7, cc0.025, cdc0.05)  | 1.1, 1.2, 1.3, 1.4, 1.5 | zeros | 1 |
| C (d1e-6, cc0.035, cdc0.0)   | 1.1, 1.2, 1.3, 1.4, 1.5 | zeros | 1 |

## Dropped axes

- **qvel_init=reference.** Originally proposed (s12 Arm D had one surviving qref cell at tcorr=0.76). Dropped from s13 to keep the sweep single-hypothesis; defer to s14.
- **Seed replication.** 1 seed per cell in this wave. If a cell crosses the trial_corr bar on 1 seed, s14 replicates at +2 seeds.
- **Low-damp anchors (d ≤ 2e-7).** Per s11 evidence, reward-weak without shape compensation. Not worth extrapolating into fs>1.0.
- **Training budget changes.** Inherits s12's `--num-timesteps 800000000`.

## Pinned base args (inherited from `sweep_s12_ms_1.sh`)

```
# --walker-xml omitted: trainer defaults to mouse_forelimb_right_moving_shoulder_ik.xml
--ctrl-dt 0.0025
--sim-dt 0.00125
--episode-length 100
--joint-armature 4e-10
--num-timesteps 800000000
--joints-weight 5.0
--joints-vel-weight 0.5
--wrist-pos-weight 0.1
--bodies-pos-weight 0.1
--qvel-init zeros
```

Trainer: `train_mouse_janelia_sigmoid_moving_shoulder.py`.

## Wandb tags

Root: `s13-ms`. Per-cell tags: `armM`, `anchorA`/`anchorB`/`anchorC`, `fs1p1`…`fs1p5`, `qzero`. Per-run `--tag`: e.g. `armM-anchorA-fs1p2`. Run names: `s13-ms-armM-anchorA-fs1p2-YYYYMMDD-HHMMSS`.

## Success criteria

Any cell that crosses all four bars on a single seed is a **winner** (replicate in s14 with +2 seeds):

- `eval/episode_reward ≥ 400`
- `eval/emg_biceps_trial_corr ≥ <P4 calibration>` (filled in after backfill)
- `eval/emg_triceps_trial_corr ≥ <P4 calibration>`
- `eval/emg_biceps_mae ≤ 0.15` AND `eval/emg_triceps_mae ≤ 0.15`

Arm-level partial signals:
- Reward rises monotonically with fs at any anchor → fs extrapolation is valid; fs>1.5 may be worth a sub-sweep.
- Shape correlation (trial_corr) collapses at fs≥1.3 → actuator saturation; retarget to fs∈[1.05, 1.2].
- No cell crosses all four bars but Anchor C gets closest → the d=1e-6 regime is the shape-frontier; s14 densifies around it.

## Execution

**Compute:** 2 × 2-GPU workers + 3 × 1-GPU workers = **5 parallel streams**. Training is ~1 hr/run at 800M timesteps.

**Partition:** 15 cells split 3-3-3-3-3 across five `sweep_s13_ms_N.sh` scripts, one per job. Allocation by anchor for clean tag grouping:

- `sweep_s13_ms_1.sh` (2-GPU): Anchor A fs ∈ {1.1, 1.2, 1.3}
- `sweep_s13_ms_2.sh` (2-GPU): Anchor A fs ∈ {1.4, 1.5}, Anchor B fs=1.1
- `sweep_s13_ms_3.sh` (1-GPU): Anchor B fs ∈ {1.2, 1.3, 1.4}
- `sweep_s13_ms_4.sh` (1-GPU): Anchor B fs=1.5, Anchor C fs ∈ {1.1, 1.2}
- `sweep_s13_ms_5.sh` (1-GPU): Anchor C fs ∈ {1.3, 1.4, 1.5}

Per-script structure inherits the `run_cell` loop from `sweep_s12_ms_1.sh`: OK/CRASHED tallies, per-cell log redirect, timestamped run names, wandb tags.

**Wallclock:** 3 cells × ~1 hr = ~3 hr per worker, all streams in parallel.

**Launch doc:** `S13_MS_LAUNCH.md` with the five shell commands, GPU assignments, screen/tmux session names, wandb filter URL, and the pre-launch checklist (P1–P4 completed).

## Risks

- **fs=1.5 actuator saturation.** No hard clamp exists in the trainer; saturation manifests as NaN reward or early-terminate collapse. Mitigation: Anchor A fs=1.5 crashes → s14 range retargeted; other anchors' fs=1.5 continues.
- **d=3e-7 at fs>1.0 may be unstable.** s12 showed high-fs wants high damp; Anchor B tests the opposite. Cheap to learn — 5 runs.
- **Metric change bug.** If trial_corr computation is wrong, entire sweep ranks on wrong numbers. Mitigation: add a unit test for `compute_emg_metrics` that checks trial_corr against a hand-computed fixture before backfill.
- **Backfill mismatches live eval.** If `scripts/emg_comparison.py` uses a slightly different pipeline than training eval, backfill numbers won't be directly comparable. Mitigation: unit test that computes trial_corr on the same fixture via both paths and asserts agreement.

## Open questions NOT answered by this sweep

- **Does qref lift trial_corr?** s14.
- **Is fs>1.5 productive?** Out of scope; only probed if Arm M shows monotonic rise.
- **Seed variance at the frontier.** 1 seed per cell; variance unknown until s14 replication.
- **Does trial_corr correlate with wandb's existing `*_trial_mae`?** Plotting during backfill will answer this — if trial_mae is a good proxy for trial_corr, we've been closer to the right metric than we thought.
