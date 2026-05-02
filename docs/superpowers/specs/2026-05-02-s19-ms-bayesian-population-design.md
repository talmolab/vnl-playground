# s19-ms — Bayesian population framework + multi-seed σ² anchor + γ regime revisits

**Status:** spec, 2026-05-02.
**Companion specs:** `2026-05-02-hierarchical-bayesian-emg-population-design.md` (framework architecture), `2026-05-01-s18-ms-cc-cdc-percentile-design.md` (predecessor sweep), `2026-05-01-s17-ms-multi-animal-design.md` (cohort infra).
**Time-box:** ~6h GPU wall on 6 GPUs (9 cells × 800M steps), parallel CPU build of the framework.

---

## What's new vs s17/s18

s17 produced cohort-trained M2 (fs=1.1, cc=cdc=0.025, p98_per_muscle) as the best-known cell. s18 swept cc/cdc/percentile/fs *around* M2 across 22 cells. Both sweeps optimized for the single-best-cell rather than for *characterizing the manifold of valid solutions across 5 mice*.

**Triceps doesn't matter for this reach** — anatomical AD + biceps drive elbow flexion in the IK reference. The s18 results re-ranked by `min(cohort_AD_corr, cohort_biceps_corr)`:

| s18 cell | cc | cdc | fs | AD_corr | biceps_corr | min | reward |
|---|---|---|---|---|---|---|---|
| **C1** (winner) | 0.00 | 0.00 | 1.1 | 0.651 | 0.500 | **0.500** | 442.7 |
| F4 | 0.025 | 0.025 | 1.2 | 0.633 | 0.480 | 0.480 | 450.0 |
| F2 | 0.025 | 0.025 | 1.0 | 0.417 | 0.485 | 0.417 | 435.1 |
| F3 (anchor) | 0.025 | 0.025 | 1.1 | 0.584 | 0.236 | 0.236 | 446.1 |

**C1 (cc=0, cdc=0)** wins under the corrected (AD, biceps) metric — confirming the s18 H3 hypothesis that *zero magnitude penalty* is AD-friendly without hurting biceps. F3 (the s17 M2 replicate) loses on biceps under this metric. s19's σ² anchor moves to C1.

**Single-animal s10–s16 cells achieved min_corr 0.5–0.84** under the legacy single-animal-trained, p98_per_muscle setup. Cohort training closed that gap on AD but opened a gap on biceps for several cells. The γ track tests *whether the regimes that worked single-animal recover under cohort + p98_per_muscle*.

## Framing

The Bayesian framework spec (`2026-05-02-hierarchical-bayesian-emg-population-design.md`) is **an analysis layer over existing checkpoints — not a trainer change**. It importance-reweights a population of networks against each mouse's empirical EMG to produce per-mouse posteriors, then validates with cross-mouse discrimination + within-mouse coverage + permutation null.

The Bayesian framework needs three things s17+s18 don't yet provide:

1. **A σ² estimator from real seed scatter** at one anchor cell (the framework's correlation likelihood requires it).
2. **A diversity-spanning population** that samples qualitatively distinct motor modes (so per-mouse posteriors *can* differ — if all networks are near-replicates, the 5×5 cross-mouse matrix is uniform by construction).
3. **A UCM-alignment test** (Latash) — does the cross-network EMG covariance at fixed kinematics align with within-mouse cross-trial EMG covariance? This is the headline biological claim.

s19 is the smallest sweep that supplies (1) and (2), shipped alongside the framework code that delivers (3).

## Tri-track structure

### Track 1 — Bayesian framework MVP M3 (CPU/code, parallel with the GPU sweep, ships first)

Builds `vnl_playground/bayesian_emg/` from the existing s17+s18 cache (~44 networks × 5 mice). Layout:

```
vnl_playground/bayesian_emg/
  data.py                    # NetworkMouseFit cache (parquet)
  likelihoods/
    correlation.py           # Fisher-z Gaussian on per-(mouse, muscle) Pearson r
  posterior.py               # importance reweighting, ESS, credible sets
  validation/
    discrimination.py        # 5×5 cross-mouse log-likelihood matrix + permutation null
    coverage.py              # leave-trial-out within-mouse 90% credible band
    ucm.py                   # principal-angle alignment between
                             #   cross-network EMG cov (at fixed kinematics)
                             #   and within-mouse cross-trial EMG cov
  preregistration.py         # YAML loader + SHA-256 hash validator
  report.py                  # HTML aggregator
scripts/
  bayes_emg_build_cache.py   # one-time: globs checkpoints, rolls out, writes cache
  bayes_emg_run.py           # full pipeline → report
configs/bayesian_emg/
  preregistration.yaml       # σ² source, ESS threshold, ε quantile, cache hash, perm seed
```

Cache scope: all s17 + s18 finished checkpoints from the cohort + p98_per_muscle sub-population (s17 M2, s18 F1–F6, P1–P5, C1–C11) — restricting to networks trained under the same norm method makes the importance-reweighting interpretable. Adds s19 cells incrementally as they finish.

**UCM alignment test specifics.** For each mouse `m` and muscle `μ ∈ {AD, biceps}` (triceps deferred):

- `Σ_bio[m, μ]` = inter-trial covariance (T × T, T=60 timesteps) of empirical envelope.
- `Σ_sim[μ]` = inter-network covariance of cell-mean envelopes (over the population that passes kinematics).
- Compute principal angles between top-`k=3` eigenvectors of `Σ_bio` and `Σ_sim`.
- Headline metric per `(m, μ)`: `mean(cos(principal_angle))` ∈ [0, 1]. 1 = perfect alignment, 0 = orthogonal.
- Permutation null: shuffle network labels across mice; recompute alignments 10,000 times.

**Acceptance for "networks span biology's UCM" claim:** mean alignment ≥ 0.6 on at least 6 of 10 (mouse, muscle) cells, permutation p < 0.05. (10 = 5 mice × 2 muscles, with triceps excluded.)

### Track 2a — multi-seed σ² anchor (4 GPU runs)

Re-train the **C1 cell** (cc=0, cdc=0, fs=1.1, p98_per_muscle, jd=1.5e-6, sd=6e-7, percentile=98) at seeds 1, 2, 3, 4. Combined with the existing seed-0 from s18 → 5 seeds at C1.

The cross-seed scatter at C1 → σ²_(mouse, muscle) for the framework's correlation likelihood. C1 is the right anchor because:

- Under the corrected (AD, biceps) metric, C1 is the s18 winner.
- The zero-penalty cell is conceptually clean: any seed-to-seed variance reflects optimizer/policy stochasticity, not penalty trade-offs.
- C1 is far enough from the s17 M2 hyperparams (cc=cdc=0.025) to register as a genuinely different operating point — useful for the framework's "is the σ² mode-dependent?" check.

### Track 2b — γ regime revisits (5 GPU runs, 1 seed each)

Five cells re-running historical leaders from s10/s11/s13/s15/s16 under cohort + p98_per_muscle norm. Each cell is a known qualitatively-distinct motor mode.

| ID | Origin (single-animal era) | fs | jd | sd | cc | cdc | tau (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| **γ1** | s10 d9em7-fs0p5 (all-time leader, 0.84) | **0.5** | 9e-7 | 9e-7 | 0.05 | 0.10 | default 25 | low-fs / low-effort regime |
| **γ2** | s11-goldi (B06; s16 anchor) | 1.0 | **5e-7** | **5e-7** | 0.05 | 0.10 | default 25 | low-damping symmetric |
| **γ3** | s13-anchorA | 1.1 | 9e-7 | 9e-7 | 0.025 | 0.025 | default 25 | mid-damping symmetric |
| **γ4** | s15-F1 | 1.0 | 1.2e-6 | **1.2e-6** | 0.025 | 0.025 | default 25 | coupled-equal damping |
| **γ5** | s16-T1f (uniform tau=40) | 1.1 | 1.5e-6 | 6e-7 | 0.025 | 0.025 | **40** | extended tau (lag-fix lever) |

These five cells span:

- fs ∈ {0.5, 1.0, 1.1} (low to mid)
- jd ∈ {5e-7, 9e-7, 1.2e-6, 1.5e-6} (full damping ladder)
- sd ∈ {5e-7, 6e-7, 9e-7, 1.2e-6} (decoupled, low-mid)
- cc, cdc ∈ {0.025, 0.05} × {0.025, 0.10} (low-effort regime)
- tau ∈ {default 25 ms, 40 ms} (timing-control axis)

If γ1 (very low fs=0.5) recovers single-animal-era min_corr=0.84 even under cohort, the entire fs=1.0–1.4 band the s17/s18 sweeps explored is suspect.

## Sweep cells (9 total)

| Track | ID | fs | jd | sd | cc | cdc | tau (ms) | seed | Wandb tags |
|---|---|---|---|---|---|---|---|---|---|
| 2a | A1.s1 | 1.1 | 1.5e-6 | 6e-7 | 0.0 | 0.0 | default | 1 | s19-ms cohort sigma-anchor C1-replicate |
| 2a | A1.s2 | 1.1 | 1.5e-6 | 6e-7 | 0.0 | 0.0 | default | 2 | s19-ms cohort sigma-anchor C1-replicate |
| 2a | A1.s3 | 1.1 | 1.5e-6 | 6e-7 | 0.0 | 0.0 | default | 3 | s19-ms cohort sigma-anchor C1-replicate |
| 2a | A1.s4 | 1.1 | 1.5e-6 | 6e-7 | 0.0 | 0.0 | default | 4 | s19-ms cohort sigma-anchor C1-replicate |
| 2b | γ1 | 0.5 | 9e-7 | 9e-7 | 0.05 | 0.10 | default | 0 | s19-ms cohort gamma s10-revisit |
| 2b | γ2 | 1.0 | 5e-7 | 5e-7 | 0.05 | 0.10 | default | 0 | s19-ms cohort gamma s11-revisit |
| 2b | γ3 | 1.1 | 9e-7 | 9e-7 | 0.025 | 0.025 | default | 0 | s19-ms cohort gamma s13-revisit |
| 2b | γ4 | 1.0 | 1.2e-6 | 1.2e-6 | 0.025 | 0.025 | default | 0 | s19-ms cohort gamma s15-revisit |
| 2b | γ5 | 1.1 | 1.5e-6 | 6e-7 | 0.025 | 0.025 | **40** | 0 | s19-ms cohort gamma s16-revisit tau-extended |

Held-fixed across all 9 cells: cohort training (`A36-1 AT006 AT009 AT012 AT013`), `p98_per_muscle` norm, percentile=98, walker_xml=`mouse_forelimb_right_moving_shoulder_ik.xml`, episode_length=100, qvel_init=zeros, joint_armature=4e-10, ctrl_dt=0.0025, sim_dt=0.00125, num_timesteps=800M, num_evals=8.

**Hard preflight gate (option ii from brainstorming):** Each γ cell runs a 50M-step pilot (`--num-timesteps 50000000 --num-evals 1 --no-wandb`) on the same hyperparams; must reach `eval/episode_reward ≥ 250` to proceed to the full 800M run. The 4 anchor seeds skip preflight (C1 already converges from s18). Pilot adds ~30 min/cell × 5 cells = ~2.5 GPU-hours total, easily absorbed inside the 6h × 6 GPU envelope.

## Wallclock budget

| Track | Cells | Per-cell wall | GPU-hours |
|---|---:|---:|---:|
| 2a anchor seeds | 4 | 4.0 h | 16 |
| 2b γ cells (incl. preflight) | 5 | ~4.5 h | 22.5 |
| **Total** | **9** | — | **~38.5** |

6 GPUs × 6h wall = 36 GPU-hours, plus ~2.5h slack = exactly fits when distributed across 6 sweep scripts. One γ cell may extend to ~7h wall on the slowest GPU but the budget guard handles overflow gracefully.

## Pipeline

1. **Trainer infra unchanged** — same `train_mouse_janelia_sigmoid_moving_shoulder.py` and CLI flags as s17/s18.
2. **Launch sweep** — 6 GPUs × 6 sweep scripts (`sweep_s19_ms_{1..6}.sh`); each script handles 1–2 cells.
3. **Build framework MVP M3 in parallel** — code work in `vnl_playground/bayesian_emg/`, tested against the s17+s18 cache.
4. **Pre-registration YAML committed** before the s19 cache freezes.
5. **Post-sweep analysis** — re-run framework with s19 cells added; ship the report.

## Pre-registration YAML pin list

`configs/bayesian_emg/preregistration.yaml` pins **before** the cache freezes:

- σ² estimator: cross-seed scatter at the A1 anchor (seeds 0–4, total 5 seeds at C1) → 10 σ²_(mouse, muscle) estimates (5 mice × 2 muscles, triceps excluded).
- ESS threshold for "claim supported": ESS ≥ 5 per (likelihood, mouse).
- Discrimination diagonal-margin threshold: Δ ≥ 0.5 nats per (mouse, muscle).
- Coverage band: 90% nominal, ±5 pts acceptance.
- UCM alignment threshold: `mean cos ≥ 0.6` on ≥6/10 (mouse, muscle) cells, permutation p < 0.05.
- Permutation: 10,000 shuffles, seed=42.
- Cache content hash: SHA-256 of the parquet, recorded after cache freeze.

The runner computes the YAML's SHA-256 at start, embeds it in every report, refuses to produce a final report on hash mismatch.

## Falsifiable predictions

1. **σ² at C1 is comparable across mice for AD** (within a factor of 2) but more variable for biceps (where mouse-to-mouse motor strategy matters more). If σ²_AD has order-of-magnitude spread across mice, the framework's pooled σ² is wrong and per-mouse σ² is needed.
2. **At least 2 γ cells beat the s18 leaders on `cohort_min(AD, biceps)_corr`.** If none do, cohort training has saturated the practical (fs, jd, sd, cc, cdc, tau) box and s19+ should pivot to architectural levers (e.g., per-animal conditioning).
3. **γ1 (fs=0.5)** *fails* (preflight ≤ 250 reward or full-run AD < 0.2). Cohort training cannot accommodate the very-low-fs regime that s10 thrived in single-animal. If γ1 *succeeds*, that's a major finding — the s17–s18 fs window was too narrow.
4. **Cross-mouse discrimination matrix from the framework on s17+s18+s19 has Δ ≥ 0.5 nats per (mouse, muscle), permutation p < 0.01.** Per-mouse posteriors are mouse-specific.
5. **UCM alignment ≥ 0.6 on ≥6/10 (mouse, muscle) cells.** Networks span biology's UCM on AD and biceps.

## Decision gates before launching

- [x] s17 + s18 results in CSV form (`s18.csv`, `s17_s18.csv`) and triceps deprioritization confirmed.
- [x] Trainer entry point unchanged (`train_mouse_janelia_sigmoid_moving_shoulder.py`); CLI flags for `--muscle-tau-act`, `--biceps-tau-act`, etc., already present per s16 prep.
- [x] Reference clip dir exists at `vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/`.
- [ ] All 6 sweep scripts have correct `WANDB_GROUP=s19-ms-part{N}` and ESTIMATED_RUN_SECONDS budgets.
- [ ] Pre-registration YAML drafted (cache hash placeholder filled after framework cache build).

## Out of scope (deferred)

- ABC likelihood (Phase 2 in the framework doc, ~1 week of work).
- Gaussian envelope likelihood with DTW (Phase 3).
- Bayes factors for sweep-design comparisons (post-Phase 1, after coverage curve passes).
- Latent manifold structure across mice (separate spec).
- EMG in the reward (separate plan: `BIOMECH_POPULATION_SWEEP_PLAN.md`).
- Per-animal specialists (s17 Track A covered).
- Animal-conditioned policies (`BERNSTEIN_DOF_PLAN.md`).
- Multi-seed at γ cells (γ cells are 1 seed each; s20+ if a γ cell wins and seed-variance becomes load-bearing).
- Triceps-specific debugging (deferred per task definition; reach is AD + biceps).
