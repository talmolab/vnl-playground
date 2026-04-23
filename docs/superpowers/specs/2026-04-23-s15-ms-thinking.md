# s15-ms design thinking (live doc)

**Started:** 2026-04-23
**Status:** brainstorming

## Problem statement

Across s10–s14 we have a Pareto frontier but no single run clears all four gates simultaneously:

- Reward ≥ 400
- Biceps corr ≥ 0.4 AND MAE ≤ 0.15
- Triceps corr ≥ 0.4 AND MAE ≤ 0.15

The empirical pattern from the unified scatter (`plots/2026-04-22-s10-s14-ms-param-search/`):

| Sweep | Best at | Biceps | Triceps | Reward | Verdict |
|---|---|---|---|---|---|
| s10 | coupled fs=0.3–0.5 | corr=0.9 / MAE=0.30 | corr=0.7–0.9 / MAE=0.20 | 300–350 | Great shape, low amplitude, low reward |
| s11 | coupled fs=0.5–1.0 | wide scatter | wide scatter | 200–400 | Mixed — damping effect |
| s12 | fs=1.0 | corr≈0.6 / MAE=0.10 | corr≈0.6 / MAE=0.10 | ~400 | Balanced, hits gates marginally |
| s13 | coupled fs=1.1–1.4 | **corr=0.5–0.7 / MAE=0.08–0.12** | corr=0.3–0.5 / MAE=0.12–0.18 | **430–450** | High reward + decent biceps, triceps ceiling ~0.5 |
| s14 | (t=1.4, b=1.4) anchor A | corr=0.6 / **MAE=0.45** | corr=0.88 / MAE=0.17 | 399 | Great triceps corr — at cost of 5× biceps MAE |

## Goldilocks candidates (R≥400, both corr≥0.4, max MAE≤0.15)

From unified.csv scan:
1. **s13 anchor-A fs=1.1 seed2** — R=411, bcorr=0.69, tcorr=0.58, bmae=0.12, tmae=0.13 ★ best single-seed
2. s13 anchor-C fs=1.3 — R=431, bcorr=0.59, tcorr=0.44, bmae=0.08, tmae=0.14
3. s13 anchor-C fs=1.2 — R=428, bcorr=0.52, tcorr=0.44, bmae=0.08, tmae=0.12
4. s13 anchor-C fs=1.1 — R=407, bcorr=0.57, tcorr=0.44, bmae=0.10, tmae=0.11
5. s11 d5em7 fs=1.0 — R=410, bcorr=0.42, tcorr=0.52, bmae=0.12, tmae=0.10

All coupled (no per-muscle override), all fs in [1.0, 1.3]. Zone is narrow and seeded thinly.

## User's new observations (decisive for s15)

1. **s13 biceps look "really promising"** → keep s13 as the biceps anchor.
2. **s14 triceps corr "really great" but biceps MAE "insanely high"** → s14 destroyed biceps amplitude to get triceps shape. Per-muscle override didn't discover new physics, it sacrificed one muscle.
3. **"Might just be a phase shift"** → key hypothesis. If s13 and s14 encode the same underlying EMG waveform but with different timing offsets, lag-aware correlation would reveal that the "triceps ceiling" in s13 is actually a phase-shift ceiling, not a shape ceiling.
4. **EMG normalization mismatch** → reference EMG is standardized (can be any range), MuJoCo activation ∈ [0, 1] (saturates). Any time reference amplitude exceeds 1 or dips below 0, the MAE comparison is comparing apples to oranges. **Fix:** rescale reference EMG to [0, 1] per muscle (divide by max, clip at 0).

## Hypotheses to pressure-test

H1. **Phase-shift hypothesis**: triceps ceiling at s13 is a systematic lag, not a shape limit. Evidence needed: run top s13 checkpoints through a lagged-corr metric (shift ±50 ms, take max corr and argmax lag).
- If true: s15 should optimize a phase-corrected metric, or add a lag-penalty term to the reward.

H2. **Amplitude saturation hypothesis**: reference EMG has peaks > 1 that MuJoCo activation can't match. When the policy learns to match peak triceps, it bleeds into persistent biceps activation (postural), which kills biceps corr.
- If true: renormalize reference EMG to [0, 1] per muscle, and the same s13 config becomes the "complete" winner.

H3. **Goldilocks is real but seed-fragile**: s13 anchor-A fs=1.1 (seed2) is the answer; we just need 5-seed confirmation.
- If true: s15 = 5-seed replication of top 3 s13 cells. Boring but decisive.

H4. **Joint physics wins, not force physics**: tuning damping/armature + slight fs boost could cross all 4 gates without any per-muscle tricks.
- If true: s15 = damping × fs mini-grid at fixed s13 baseline.

## s15 candidate strategies (to present to user)

### A. "Pipeline upgrade first, sweep second" (recommended)
Before any sweep:
1. Implement **per-muscle [0,1] EMG renormalization** as a task config option.
2. Add **lagged correlation metric** (shift-max corr, shift-argmax lag_ms) to eval.
3. Re-score top s13 checkpoints with new metrics. If any already clear all 4 gates, done.
4. Only then sweep — a narrow 5-seed replication of top 2–3 cells under the new pipeline.

### B. "Goldilocks seed-depth" (cheap & decisive)
Skip pipeline changes. 5 seeds × top 3 s13 cells (anchor-A fs=1.1, anchor-C fs=1.2, anchor-C fs=1.3) + 5 seeds × s11 fs=1.0 baseline. 20 runs total.

### C. "Asymmetric anchors" (riskier)
Freeze biceps at s13 settings (that we know work), sweep triceps physics only — maybe damping or armature on triceps insertion, or tau_triceps alone.

### D. "Hybrid: B + a shoulder-fs 1-axis scan"
B plus: fix elbow muscles at fs=1.4, scan shoulder_fs ∈ {1.1, 1.2, 1.3, 1.4}. Tests whether the s14 C7 win was really shoulder-fs<elbow-fs, cheaply.

## Findings from code + CSV exploration (2026-04-23)

### EMG pipeline (`train_mouse_janelia_sigmoid_moving_shoulder.py`)

| Item | Current behavior | Where |
|---|---|---|
| Reference EMG source | 30 kHz raw ADC → bandpass 20–1000 Hz → rectify + 50 Hz LPF → resample to sim timesteps | lines 79–143 |
| Reference normalization | `arr / np.percentile(arr, 98)` **per-muscle, dataset-wide** | line ~140 |
| Reference clipping | Clipped to [0, 1] in eval | line 2017 |
| Sim "EMG" source | `data.act` (post Hill-model activation filter) — **NOT** `data.ctrl` | line 1542 |
| Sim range | [0, 1] natively | — |
| **mean_corr** | `np.corrcoef(sim_mean, emg_mean_trace)` on trial-averaged traces | lines 159–182 |
| **mean_mae** | `np.mean(np.abs(sim_mean - emg_mean_trace))` on trial-averaged traces | 159–182 |
| **trial_mae** | Per-trial MAE, averaged | 159–182 |
| **trial_corr** (spec'd in s13) | **NOT IMPLEMENTED** | — |
| **lagged_corr / phase_lag** (spec'd in s14) | **NOT IMPLEMENTED** | — |

### CLI knobs already wired

| Flag | Effect | Default use in s13 |
|---|---|---|
| `--force-scale` | uniform multiplier on all `actuator_gainprm[:, 0]`, applied AFTER per-muscle overrides | 1.1–1.7 swept |
| `--biceps-force`, `--triceps-long-force` | absolute pre-fs gainprm | used in s14 per-muscle overrides |
| `--brachialis-force`, `--triceps-lat-force` | same for accessory muscles | unused in recent sweeps |
| `--joint-damping`, `--shoulder-damping`, `--elbow-damping` | per-group damping | anchor A: 9e-7, armature 4e-10 |
| `--muscle-tau-act`, `--muscle-tau-deact` (+per-muscle overrides) | Hill activation filter time constants | **never swept post-s11** |
| `--control-cost`, `--control-diff-cost`, `--saturation-cost` | reward penalties | anchor A: cc=0.025, cdc=0.025 |

### Stats from `unified.csv` (refined)

- Top single run by min(bcorr, tcorr) with R≥400: **s13-armM-anchorA-fs1.1-seed2** (R=411, bcorr=0.695, tcorr=0.578).
- 20 of top-20 min-corr runs have force_scale ∈ [1.0, 1.4]. None at fs≥1.5. None at fs≤0.9 with R≥400.
- **s14 tcorr vs tmae Pearson r = +0.57** — high triceps corr comes *with* high MAE (amplitude blow-up, not noise).
- **s14 R≥400 ∧ bmae≤0.15 → 0 runs.** **s13 R≥400 ∧ bmae≤0.15 → 24 runs.** s13 regime wins any composite metric.
- Seed-level std: median bcorr std = 0.20, max 0.58. Median tcorr std = 0.10, max 1.04. Single-seed cell comparisons with gaps <0.2 bcorr are not trustworthy.

## Re-interpretation of user's questions

**"EMG not max=1, MuJoCo max=1"** — **CONFIRMED BUG, promoted to top priority.**
User verified in wandb plots: reference EMG peaks frequently exceed 1.0 after p98 normalization (line 143). Sim activation is hard-clipped at 1.0 (line 2017, Hill-model bound).
Consequence: on every EMG burst, the reference has headroom the sim cannot reach. The policy is forced to plateau at 1.0 for the peak duration, which **distorts peak timing** — inducing an apparent phase shift even when the policy has learned the correct shape. This likely couples with the user's "phase shift" intuition: the phase shift is itself a symptom of amplitude clipping, not a separate problem.

**Fix:** Change p98 → p100 (or 99.9 to still trim rare spikes). Add `--emg-norm-percentile` CLI knob (default 100). Reference peak now equals 1.0 in the worst bursts, sim can match it. Optionally add `--emg-norm-mode {dataset,per_trial}` for completeness.

**"Might just be a phase shift"** — this is now **directly testable** because lagged-corr is NOT implemented. Adding it is a ~30-line change. If the top s13 cells have |phase_lag| > 0 with lagged_corr much higher than `mean_corr`, that *is* the ceiling. Then s15 can either (a) report the phase-corrected score as the real answer or (b) add a lag term to reward.

**"Keep biceps like s13 but triceps like s14"** — false dichotomy under the new evidence. s14 didn't make the triceps *shape* better than s13; it **broke the biceps amplitude** and the `mean_corr` increase on triceps was likely (a) the muscle now firing against *less* biceps antagonism (cleaner timing) and/or (b) simultaneously a phase-shift artifact from changed shoulder/elbow torque balance. Need lagged corr to separate.

## Strategic shift for s15

The leverage point is NOT another parameter grid. It's **metrics**. Two specific metric additions (per-trial corr, lagged corr) + a re-evaluation of existing s13 checkpoints will tell us whether the "triceps ceiling" is biology or measurement. Only then do we sweep.

## s15 candidate strategies (revised after normalization confirmation)

The structure is now: **fix infra, re-score on disk if possible, then retrain with fixed infra and confirm.**

### Stage 1 — Infra fixes (all in one PR, no GPU)
**1a. EMG reference renormalization.** Change `arr / np.percentile(arr, 98)` → `arr / np.percentile(arr, args.emg_norm_percentile)` with default **100**. Add `--emg-norm-percentile` CLI flag. `train_mouse_janelia_sigmoid_moving_shoulder.py:143`.
**1b. Per-trial correlation.** In `compute_emg_metrics` (line 159), compute `trial_corr_mean`, `trial_corr_median` across paired bio/sim trials. Log `eval/emg_{muscle}_trial_corr`.
**1c. Lagged (phase-shift) correlation.** Compute `lagged_corr_max` (max of corr over lag ∈ [−20, +20] steps) and `phase_lag_steps` (argmax lag). Log both per muscle.
**1d. Optional — `--emg-norm-mode {dataset, per_trial}`.** Skip initially; add only if Stage 2 shows per-trial normalization matters.

### Stage 2 — Re-evaluate existing checkpoints with fixed infra (eval only, cheap)
Pick 8 checkpoints spanning the best cells: s13-armM-anchorA-fs1p1-s2, s13-anchorC-fs1p2, s13-anchorC-fs1p3, s14-anchorA-C7, s14-anchorA-C4, s12-fs1p0-best, s11-d5em7-fs1p0, s10-shape-king. Run a new `--eval-only` pass (or short 1-step resume) under the fixed metrics. Record:

| cell | R | bcorr_old | bcorr_new_p100 | biceps_trial_corr | biceps_lag_ms | tcorr_old | tcorr_new_p100 | triceps_trial_corr | triceps_lag_ms | bmae | tmae |
|---|---|---|---|---|---|---|---|---|---|---|---|

### Stage 3 — Branch decision based on Stage 2 results

| If Stage 2 shows | Then s15 is |
|---|---|
| **Some existing s13 cell clears all 4 gates under p100 + lagged_corr ≥ 0.7 on both muscles** | 5-seed replication of that cell. ~15 runs. Ship. |
| **Lagged_corr is high (≥0.7) on both muscles but with |lag| > 4 steps** | Retrain 2 cells × 5 seeds with a phase-alignment reward term (advances the sim action by the measured lag in the comparison). ~10 runs. |
| **p100 renormalization alone doesn't close the gap, shape genuinely caps** | 5 seeds × top 2 s13 cells under new infra PLUS 4-cell shoulder-fs scan (elbow fs=1.4, shoulder fs ∈ {1.1, 1.2, 1.3, 1.4}) × 3 seeds. ~22 runs. |
| **Normalization mode matters (per-trial >> dataset)** | Land --emg-norm-mode flag, retrain 2 cells × 5 seeds with per-trial. ~10 runs. |

### Rejected strategies

- Another ratio grid (like s14). Falsified — per-muscle asymmetry blows up bmae.
- Tau sweep in isolation. No evidence tau is the bottleneck; the normalization + lag story is more specific.
- Reward shaping before metric audit. We'd be shaping toward a broken metric.

## Recommendation

**Stage 1 is mandatory** regardless of which Stage 3 branch we end up on — every future metric is contaminated by the amplitude-clipping bug. Stage 2 is nearly free and may short-circuit the whole sweep. Only commit to Stage 3 compute after Stage 2 tells us what to fix.

## Parameter budget if Stage 3 happens

Anchor on s13 **anchor-A** baseline (best-documented, top single run):
- `--force-scale` ∈ {1.0, 1.1, 1.2, 1.3} — already swept but with broken metric
- `--joint-damping` = 9e-7 (fixed at anchor A)
- `--joint-armature` = 4e-10 (fixed)
- `--control-cost` = 0.025, `--control-diff-cost` = 0.025 (fixed)
- `--muscle-tau-act/deact` = defaults (fixed, not changed)
- `--emg-norm-percentile` = 100 (new, default)
- seeds = 5 per cell
- ~15 runs at 1 GPU each, ~1 day wall-clock

Total s15 compute if we go full Stage 3: 15–22 runs. Roughly 1/4 of s14's footprint for a far more decisive result.

