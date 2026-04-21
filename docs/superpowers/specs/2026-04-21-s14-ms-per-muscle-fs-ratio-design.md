---
name: s14-ms-per-muscle-fs-ratio-design
description: Design for s14_ms — per-muscle (biceps vs triceps) force-scale ratio ladder + coupled diagonal at 2 well-performing damping anchors, plus lagged cross-correlation EMG metric
type: project
---

# s14-ms: per-muscle force-scale ratio sweep with lagged cross-correlation

**Date:** 2026-04-21
**Status:** design

## Goal

Break the reward-vs-shape Pareto frontier that s10→s13 revealed: **no cell in any prior sweep has crossed R≥400 AND bcorr≥0.70 AND tcorr≥0.60 simultaneously**. s10's shape-king region (fs=0.5–0.7, bcorr=0.90, tcorr=0.79) peaks at R=344. s13's high-fs region (fs=1.1–1.8) pushes R to 430–450 but triceps correlation collapses below 0.5. s14 tests whether **asymmetric per-muscle force scaling** — triceps kept in its shape-preserving regime while biceps is pushed into the reward-driving regime — breaks that frontier.

A second, orthogonal goal: **separate timing from shape** in the EMG metrics. The Pearson `emg_*_corr` penalizes both amplitude-mismatch and phase-mismatch indiscriminately. s13 anecdotally observed "the triceps phase of the bump gets earlier and earlier" as fs rises — we suspect a large fraction of the tcorr collapse is pure phase lag. The new `lagged_corr`/`phase_lag_ms` metric isolates those two effects.

## Hypotheses

1. **Per-muscle ratio hypothesis.** Both muscles need effective fs≥1.0 to match reward; triceps shape-corr degrades above ~fs=1.2 because higher forces yield faster angular velocity and force the triceps deceleration-burst to fire earlier than biological EMG. If we hold triceps at `t_eff ≤ 1.1` while boosting biceps to `b_eff ∈ {1.3, 1.4, 1.5}`, we should see tcorr recover toward s10 levels (≥0.7) while R stays ≥400.

2. **Phase-lag hypothesis.** At matched `t_eff`, a non-trivial fraction (>0.1) of the current `tcorr` gap vs s10 is pure phase lag; after lag alignment (`lagged_corr`), the same s13 cells should look substantially better, and the s14 ratio cells should reduce raw lag (not just boost lagged_corr).

3. **Falsification.** The reverse ratio `(t=1.3, b=1.0)` should **not** improve tcorr — if it does, the mechanism is not what we think.

## Prior data this builds on

Pooled from `s12_s11_s10_ms_all.csv` (6715 rows; s10/s11/s12 finished runs) and `s13_ms.csv` (32 rows; s13 through today).

### The shape-king / reward-king tradeoff

| regime | source | fs | R | bcorr | tcorr |
|---|---|---|---|---|---|
| shape-king | s10 | 0.7, d=5e-7 | 344 | **0.90** | **0.79** |
| shape-king | s10 | 0.5, d=9e-7 | 327 | 0.86 | 0.84 |
| s12 composite peak | s12 | 1.0, d=1e-6 | 373 | 0.69 | 0.72 |
| s12 near-R-bar | s12 | 1.0, d=9e-7, cc=0.025, cdc=0.025 | 397 | 0.53 | 0.53 |
| s13 anchor-A best | s13 | 1.2, d=9e-7 | 428 | **0.84** | 0.33 |
| s13 anchor-B best | s13 | 1.4, d=3e-7 | 429 | 0.74 | 0.43 |
| s13 anchor-C best | s13 | 1.2, d=1e-6, s2 | 438 | 0.75 | 0.24 |

Clear pattern: as effective fs rises 0.7 → 1.5, bcorr holds up (or rises), tcorr collapses. No symmetric-fs cell crosses the composite bar.

### s13 per-anchor fs landscape summary

| anchor | best fs for bcorr | best fs for tcorr | best fs for R |
|---|---|---|---|
| A (d9e-7 / 0.025 / 0.025) | 1.2 (s2): 0.84 | 1.1 (s2): 0.58 | 1.6: 450 |
| B (d3e-7 / 0.025 / 0.05)  | 1.4: 0.74 | 1.3: 0.50 | 1.5: 434 |
| C (d1e-6 / 0.035 / 0.0)   | 1.2 (s2): 0.75 | 1.5 (s2): 0.47 | 1.5 (s2): 444 |

Best bcorr at each anchor sits at fs=1.2–1.4; best tcorr at fs=1.1–1.3. The best-bcorr fs and best-tcorr fs differ within each anchor — that's the signal that an asymmetric per-muscle fs should do better than any symmetric fs.

## Mechanism: per-muscle force override via existing CLI

`vnl_playground/tasks/mouse/base.py:250-289` already implements per-muscle absolute force overrides that apply **before** the global `force_scale` multiplier:

```
effective_force_i = override_i × force_scale    # if override set
                  = xml_default_i × force_scale # otherwise
```

XML defaults (from any s12/s13 `xml/muscle_force/*` wandb log):

| muscle | XML default (N) |
|---|---|
| Biceps_Long | 0.1 |
| Brachialis | 0.1 |
| Triceps_Long | 0.1 |
| Triceps_Lateral | 0.1 |
| AD | 0.4 |
| MD | 0.2 |
| PD | 0.2 |
| Pec_C | 0.2 |
| Lat | 0.2 |
| Infraspinatus | 0.5 |
| Supraspinatus | 0.5 |
| Subscapularis | 0.6 |

For each cell with desired effective triceps multiplier `t_eff` and biceps multiplier `b_eff`, relative to per-anchor shoulder fs `fs_s`:

```
--force-scale          = fs_s
--biceps-force         = 0.1 * b_eff / fs_s
--brachialis-force     = 0.1 * b_eff / fs_s
--triceps-long-force   = 0.1 * t_eff / fs_s
--triceps-lat-force    = 0.1 * t_eff / fs_s
```

Giving: effective biceps = `(0.1 * b_eff / fs_s) * fs_s = 0.1 * b_eff`; same for triceps. Shoulder muscles (no override) stay at `xml_default * fs_s`, preserving the anchor's reward regime.

**Muscle grouping.** Biceps_Long and Brachialis scale together (both elbow flexors). Triceps_Long and Triceps_Lateral scale together (both elbow extensors). Otherwise the policy compensates via the un-scaled synergist.

No code change is required for this axis — the launch script computes the 4 absolute overrides from `(t_eff, b_eff, fs_s)`.

## Prerequisites (pre-launch, gating)

### P1. Lagged cross-correlation metric

Extend `compute_emg_metrics()` in `train_mouse_janelia_sigmoid_moving_shoulder.py:159-182`:

```python
def compute_lagged_corr(sim_trace, emg_trace, max_lag_steps=20):
    """Max |Pearson r| over ±max_lag_steps; returns (max_abs_r_signed, best_lag_steps)."""
    n = min(len(sim_trace), len(emg_trace))
    sim_trace = np.asarray(sim_trace[:n])
    emg_trace = np.asarray(emg_trace[:n])
    best_r, best_lag = 0.0, 0
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            s, e = sim_trace[-lag:], emg_trace[:n + lag]
        elif lag > 0:
            s, e = sim_trace[:n - lag], emg_trace[lag:]
        else:
            s, e = sim_trace, emg_trace
        if len(s) < 10 or np.std(s) < 1e-8 or np.std(e) < 1e-8:
            continue
        r = float(np.corrcoef(s, e)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_r, best_lag
```

Add to `compute_emg_metrics`, after the existing `mean_corr` and `trial_mae` block:

```python
# Mean-trace lagged correlation
lag_r, lag_steps = compute_lagged_corr(sim_mean, emg_mean_trace, max_lag_steps=20)
result["lagged_corr"] = lag_r
result["phase_lag_steps"] = lag_steps

if bio_traces is not None:
    per_trial_lagged = []
    per_trial_lag_steps = []
    for i in range(min(sim_muscle.shape[0], bio_traces.shape[0])):
        r_i, lag_i = compute_lagged_corr(
            sim_muscle[i, :T], bio_traces[i, :T], max_lag_steps=20
        )
        per_trial_lagged.append(r_i)
        per_trial_lag_steps.append(lag_i)
    result["trial_lagged_corr_median"] = float(np.nanmedian(per_trial_lagged))
    result["trial_phase_lag_steps_median"] = float(np.nanmedian(per_trial_lag_steps))
```

Log in the eval loop around line 2026-2034:

```python
wandb_log[f"eval/emg_{muscle_name.lower()}_lagged_corr"] = m["lagged_corr"]
wandb_log[f"eval/emg_{muscle_name.lower()}_phase_lag_ms"] = m["phase_lag_steps"] * env_cfg.ctrl_dt * 1000
if "trial_lagged_corr_median" in m:
    wandb_log[f"eval/emg_{muscle_name.lower()}_trial_lagged_corr_median"] = m["trial_lagged_corr_median"]
    wandb_log[f"eval/emg_{muscle_name.lower()}_trial_phase_lag_ms_median"] = (
        m["trial_phase_lag_steps_median"] * env_cfg.ctrl_dt * 1000
    )
```

Keep `eval/emg_{muscle}_corr` unchanged for backward compatibility with s10/s11/s12/s13 analysis.

**Lag window choice:** ±20 steps at `ctrl_dt = 0.0025s` = ±50 ms. EMG bursts in the Janelia reach are ~100–200 ms wide, so 50 ms covers half a bump-width — enough to catch real phase mismatch without letting the metric match unrelated bumps at pathological lags.

### P2. Unit test for compute_lagged_corr

Add `tests/test_lagged_corr.py` with fixtures:
- Identity: `compute_lagged_corr(x, x)` returns `(1.0, 0)` within `1e-9`.
- Shifted: `compute_lagged_corr(x[5:], x[:-5])` returns approximately `(1.0, -5)` (or `+5` depending on sign convention — fix and document).
- Anti-correlated: `compute_lagged_corr(x, -x)` returns `(-1.0, 0)`.

The sign convention for `phase_lag_steps` in this design: **positive lag means sim leads EMG** (sim burst occurs earlier in time). Verify this holds in the unit test.

### P3. Backfill lagged-corr on s13 (and optionally s12) checkpoints

Re-evaluate the 30 finished s13 runs with the updated `compute_emg_metrics` to produce `s13_with_lagged.csv`. This gives us:
- Calibration of what `phase_lag_ms` values currently exist at the s13 frontier (expect triceps to show positive lag — sim leads bio).
- A numerical answer to hypothesis 2: how much of the current tcorr gap closes after lag alignment.

Out of scope for the gating backfill: s10/s11/s12. The s13 backfill is enough to calibrate the s14 lag bar and interpret results; s10-era replays can happen post-hoc if s14 produces a winner.

## Anchors

Two anchors — both at "well-performing" damping regimes. Dropping anchor B from s13 (d=3e-7); s12's surrogate-model analysis and s13's per-anchor peaks both show the composite (reward + shape) frontier sits at d ∈ {9e-7, 1e-6}, not at low damping. Keeping two damping levels rather than one as a robustness check on the ratio hypothesis.

| anchor | damp | cc | cdc | fs_shoulder | s13 @ fs=1.3 reward | s12 composite notes |
|---|---|---|---|---|---|---|
| A | 9e-7 | 0.025 | 0.025 | 1.3 | 415 / 438 (s1/s2) | s12 near-R-bar regime (R=397 at fs=1.0) |
| C | 1e-6 | 0.035 | 0.00 | 1.3 | 431 / 413 (s1/s2) | s12 surrogate optimum; shape frontier |

Uniform `fs_shoulder=1.3` across anchors keeps the shoulder-muscle operating point consistent, so differences between A and C can be attributed to damping rather than shoulder force.

## Ratio ladder

24 cells per anchor, split into four zones:

- **Coupled diagonal (8 cells, C1–C8):** `t_eff = b_eff` sweep, spans `{0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5}`. Serves as the "no asymmetry" baseline at every fs level — so every asymmetric cell has a matched coupled neighbor to compare against. `(1.3, 1.3)` also matches s13 anchor-X fs=1.3 exactly (shoulder = elbow = 1.3), giving a same-eval-pipeline cross-check with s13 per-trial seeds. `(1.0, 1.0)` lives in the core zone as `L0`.
- **Low-t asymmetric zone (6 cells, E1–E6):** `t_eff ∈ {0.7, 0.8, 0.9}` × `b_eff ∈ {1.0, 1.3}`. Tests whether a weakened triceps paired with a strong shoulder (fs_shoulder=1.3) helps tcorr or breaks the reach.
- **Core both-high zone (9 cells, L0–L8):** `t_eff ∈ {1.0, 1.1, 1.2}` × `b_eff ∈ {1.0, …, 1.5}`, all `b_eff ≥ t_eff`. Primary hypothesis region.
- **Falsifier (1 cell, F1):** reverse ratio `t_eff > b_eff`.

| # | t_eff | b_eff | gap | zone / rationale |
|---|---|---|---|---|
| C1 | 0.7 | 0.7 | 0.0 | coupled low — pure fs=0.7 effect at shoulder=1.3 |
| C2 | 0.8 | 0.8 | 0.0 | coupled |
| C3 | 0.9 | 0.9 | 0.0 | coupled |
| L0 | 1.0 | 1.0 | 0.0 | coupled; baseline at fs_shoulder=1.3 |
| C4 | 1.1 | 1.1 | 0.0 | coupled; matches s13 anchor-X fs=1.1 at shoulder=1.3 |
| C5 | 1.2 | 1.2 | 0.0 | coupled |
| C6 | 1.3 | 1.3 | 0.0 | coupled; matches s13 anchor-X fs=1.3 identically |
| C7 | 1.4 | 1.4 | 0.0 | coupled |
| C8 | 1.5 | 1.5 | 0.0 | coupled |
| E1 | 0.7 | 1.0 | 0.3 | low-t asymmetric |
| E2 | 0.7 | 1.3 | 0.6 | low-t asymmetric — widest gap |
| E3 | 0.8 | 1.0 | 0.2 | low-t asymmetric |
| E4 | 0.8 | 1.3 | 0.5 | low-t asymmetric |
| E5 | 0.9 | 1.0 | 0.1 | low-t asymmetric |
| E6 | 0.9 | 1.3 | 0.4 | low-t asymmetric |
| L1 | 1.0 | 1.1 | 0.1 | tiny gap at triceps floor |
| L2 | 1.0 | 1.2 | 0.2 | gap 0.2 at triceps floor |
| L3 | 1.0 | 1.3 | 0.3 | gap 0.3 at triceps floor (central prediction) |
| L4 | 1.0 | 1.4 | 0.4 | gap 0.4 at triceps floor |
| L5 | 1.1 | 1.3 | 0.2 | raised triceps floor, gap 0.2 |
| L6 | 1.1 | 1.4 | 0.3 | raised triceps floor, gap 0.3 |
| L7 | 1.2 | 1.4 | 0.2 | higher floor both, gap 0.2 |
| L8 | 1.2 | 1.5 | 0.3 | top regime, gap 0.3 |
| F1 | 1.3 | 1.0 | −0.3 | **reverse ratio (falsifier)** |

Primary 1D slices for analysis:
- **Coupled diagonal:** C1–C8 plus L0 (pure fs effect at shoulder=1.3; directly comparable to s13 at X=1.1–1.5 and a new regime below that)
- **Asymmetric at b=1.0:** E1, E3, E5, L0 (lower triceps alone, no biceps boost)
- **Asymmetric at b=1.3:** E2, E4, E6, L3, C6 (sweep triceps 0.7→1.3 with biceps fixed at central value)
- **t=1.0 slice:** L0, L1, L2, L3, L4 (hold triceps at symmetric-baseline, sweep biceps)
- **Anti-diagonal asymmetric:** L3, L5, L7, L8 (matched-gap at rising triceps floor)

Ratio effect is real iff asymmetric cells beat their coupled neighbors (lattice-close cells with t=b at midpoint) on composite. F1 falsifies.

**Seeds.** 2 seeds per cell (not 1) because s13 showed ±0.3 bcorr seed-variance at some anchor-C cells; a 1-seed readout is not robust enough to distinguish real effects from seed noise at the ratio scale we're testing.

### Total

2 anchors × 24 cells × 2 seeds = **96 runs**.

## Pinned base args

```
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

Per-anchor: `--joint-damping`, `--control-cost`, `--control-diff-cost`.
Per-cell: `--force-scale 1.3`, `--biceps-force`, `--brachialis-force`, `--triceps-long-force`, `--triceps-lat-force` (computed from `t_eff`, `b_eff` per the formula above).

Trainer: `train_mouse_janelia_sigmoid_moving_shoulder.py`.

## Wandb tags

Root: `s14-ms`. Per-cell tags: `anchorA`/`anchorB`/`anchorC`, `L0…L8`, `F1`, `t{value}b{value}` (e.g., `t1p0b1p3`), `seed1`/`seed2`, `qzero`. Per-run `--tag` prefix: e.g. `anchorC-L4-t1p0b1p3-seed1`. Run names: `s14-ms-anchorC-L4-t1p0b1p3-s1-YYYYMMDD-HHMMSS`.

## Success criteria

**Winner (any single seed satisfies all bars):**

- `eval/episode_reward ≥ 400`
- `eval/emg_biceps_corr ≥ 0.70`
- `eval/emg_triceps_corr ≥ 0.60`
- `eval/emg_biceps_mae ≤ 0.15` AND `eval/emg_triceps_mae ≤ 0.15`
- `|eval/emg_biceps_phase_lag_ms| ≤ 10` AND `|eval/emg_triceps_phase_lag_ms| ≤ 10`

A winner replicates in s15 with ≥3 seeds.

**Arm-level partial signals:**

- **Ratio works (asymmetric beats coupled).** Any asymmetric cell produces composite (reward + min(bcorr,tcorr) + min lag) strictly better than its coupled neighbors (e.g., L3=(1.0,1.3) beats both L0=(1.0,1.0) and C6=(1.3,1.3)) → ratio hypothesis confirmed; s15 drills in finer ratio grid around the optimum.
- **Coupled diagonal is always best.** If C1–C8 sweep out the Pareto front and no asymmetric cell beats its coupled neighbors → ratio hypothesis rejected; ask instead whether we need better damping/cc/cdc or a non-force mechanism.
- **Ratio fails, lag explains.** `tcorr` stays flat across the ladder, but `lagged_corr` is consistently 0.1+ higher than `corr` at all cells → current tcorr gap is mostly phase lag; s15 adds a phase-aligned EMG reward term rather than more ratio.
- **Reverse control F1 matches L4 on shape.** `(t=1.3, b=1.0)` produces tcorr ≥ L4's tcorr → mechanism is not differential-force but something else (maybe total actuator energy or co-contraction); s14 hypothesis falsified.
- **Low-t zone (E1–E6 + C1–C3) outperforms the high-t zone.** If the best cells all have `t_eff ≤ 0.9`, that contradicts my initial argument about shoulder/triceps mismatch and suggests the s10 shape-king mechanism does transfer — s15 focuses the ratio around the low-t frontier.
- **Anchor C dominates A across the board.** If C consistently out-performs A on the composite, s15 focuses on C and adds a damping micro-sweep.

## Execution

**Compute.** 2 × 2-GPU workers + 4 × 1-GPU workers = 6 parallel streams (same as s13 allocation). Training ≈ 1 hr/run at 800M timesteps.

**Partition.** 96 runs split across six `sweep_s14_ms_N.sh` scripts of 16 runs each, one script per GPU stream, assigned so each script stays within a single anchor and a coherent zone:

| Script | GPU | Anchor | Cells | Seeds | Count |
|---|---|---|---|---|---|
| `sweep_s14_ms_1.sh` | 2-GPU / GPU0 | A | C1–C8 (coupled diag) | 1, 2 | 16 |
| `sweep_s14_ms_2.sh` | 2-GPU / GPU1 | A | E1–E6, L0, F1 (low-t + sym + falsifier) | 1, 2 | 16 |
| `sweep_s14_ms_3.sh` | 2-GPU / GPU0 | A | L1–L8 (asymmetric core) | 1, 2 | 16 |
| `sweep_s14_ms_4.sh` | 2-GPU / GPU1 | C | C1–C8 (coupled diag) | 1, 2 | 16 |
| `sweep_s14_ms_5.sh` | 1-GPU | C | E1–E6, L0, F1 (low-t + sym + falsifier) | 1, 2 | 16 |
| `sweep_s14_ms_6.sh` | 1-GPU | C | L1–L8 (asymmetric core) | 1, 2 | 16 |

Concrete run order inside each script is finalized in the implementation plan.

Per-script structure: inherit the `run_cell` loop from `sweep_s13_ms_N.sh` — OK/CRASHED tallies, per-cell log redirect, timestamped run names, wandb tags.

**Wallclock.** 16 cells × ~1 hr / stream = ~16 hr per stream, all six streams in parallel ≈ 16 hr total.

**Launch doc.** `S14_MS_LAUNCH.md` with six shell commands, GPU assignments, screen/tmux session names, wandb filter URL, and the pre-launch checklist (P1–P3 completed).

## Risks

- **Lag window too narrow/wide.** ±50 ms may clip real lags at the highest-fs anchor-A cells; too wide lets the metric match unrelated bumps. Mitigation: inspect s13 backfill output — if >5% of phase_lag values saturate at ±20 steps, widen window and rerun backfill before s14 launch.
- **Sign convention bugs.** `phase_lag_steps` positive-means-sim-leads must be consistently implemented and documented; wrong sign inverts the winner definition. Mitigation: unit test P2.
- **Shoulder-muscle imbalance.** Setting `fs_shoulder=1.3` uniformly across anchors A and C may not be each anchor's best-R operating point. In s13, at fs=1.3, anchor A had R=415–438 and anchor C had R=413–431 — both ≥ 400, so the risk is small, but cells that underperform on R may be fs_shoulder-limited rather than ratio-limited. If s14 seed 1 at L0 (symmetric elbow, fs_shoulder=1.3) regresses below R=400 at either anchor, that anchor's fs_shoulder is wrong; pause that anchor's remaining runs and re-pick fs_shoulder. The coupled cell C6 (1.3, 1.3) is equivalent to s13 anchor-X fs=1.3; a sanity check that C6's R lands in s13's range gives early confirmation that the new eval pipeline reproduces s13.
- **Per-muscle override ÷ fs_shoulder rounding.** `0.1 * b_eff / fs_shoulder` must be computed in the shell script to float precision; rounding errors silently change effective force. Mitigation: compute via `python3 -c "print(0.1 * $B_EFF / $FS_S)"` in the launch script and echo the value so the log captures exact args.
- **Reverse falsifier F1 interpretation ambiguity.** If F1 produces any bcorr drop but no tcorr change, that's a null result for the falsifier, not a confirmation. Mitigation: pre-registered interpretation rule — F1 "falsifies" only if tcorr is ≥ L4's tcorr within ±0.05.

## Open questions NOT answered by this sweep

- **Does reducing triceps tau help more than reducing its force?** `--triceps-long-tau-act` and `--triceps-long-tau-deact` already exist (from s11 tau-extras). Orthogonal axis; defer to s15 if s14's ratio answer is partial.
- **Is there a single-muscle asymmetry on the biceps side alone?** Brachialis may behave differently from Biceps_Long despite both being flexors. s14 scales them together; s15 could un-pair.
- **Per-trial lag vs mean-trace lag — which is the actionable metric?** Answer comes from s13 backfill + s14 results. If per-trial lag median has high inter-trial variance, mean-trace lag is the better metric.
- **Does s14's mechanism transfer to the non-moving-shoulder (arm-only) XML?** Out of scope; s14 is moving-shoulder only.
