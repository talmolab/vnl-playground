# Hierarchical Bayesian EMG — population-of-networks vs population-of-mice

**Status:** spec, 2026-05-02.
**Owner:** eric@talmolab.org.
**Companion specs:** `2026-05-01-s17-ms-multi-animal-design.md` (multi-animal eval infrastructure this builds on), `2026-05-01-s18-ms-cc-cdc-percentile-design.md` (multi-seed sweep that supplies σ² estimates).
**Out-of-band runtime context:** this is an analysis framework, not a trainer change. It consumes existing checkpoints + per-trial empirical envelopes; no XML or training code is modified.

---

## Framing

Bernstein's redundancy problem says the motor system has many valid solutions to any given reach. The empirical EMG from one mouse is a single sample from that mouse's distribution over solutions. A single trained network is a single sample from the network class's distribution over solutions, induced by the sweep design (hyperparameter prior, seed, initial conditions). Single-network-vs-single-mouse comparison answers the wrong question.

The right question: **does the manifold of motor solutions our networks discover overlap with the manifold biology occupies, and does the structure of variation across our 5 mice's posteriors mirror the structure of biological variation?**

We have:

- 5 mice with trial-level EMG envelopes (A36-1, AT006, AT009, AT012, AT013).
- A growing population of trained networks across s10–s18 sweeps (~60+ checkpoints in scope), each producing per-trial sim envelopes via post-hoc rollout.

We layer a hierarchical Bayesian framework on top:

1. **Per-mouse posterior over networks** — for each mouse `m`, importance-weight the sweep by likelihood under that mouse to get `p(θ | EMG_m)`.
2. **Population-level claim** — the per-mouse posteriors should differ from each other in ways that mirror measurable mouse-to-mouse variation, and each posterior should fit *its own* mouse better than the other four.

The framework reports three likelihoods (correlation, ABC, Gaussian envelope), three validation tests (within-mouse coverage, between-mouse discrimination, label-permutation null), and Bayes factors for sweep-design comparisons. Pre-registration discipline (a YAML file pinning all hyperparameters before the cache is unfrozen) is the structural protection against post-hoc tuning.

## Goals

A. **Validation** — assign principled posterior weights to networks so we can say "these N networks are biologically plausible for mouse M, the rest are not," with calibration confirmed by held-out coverage.

B. **Model comparison** — quantitatively compare design choices (moving-shoulder vs. static; fs=1.1 vs. fs=1.0; z_baseline_x2 vs. p98_per_muscle; cohort-trained vs. specialist) on each mouse, via Bayes factors.

C. **Generative claim** — show that the *distribution* of solutions our sweep produces per mouse mirrors the distribution biology produces. Operationalized as: (i) per-mouse posteriors are mouse-specific (cross-likelihood matrix has a clean diagonal); (ii) per-mouse posterior centroids in hyperparameter space align with measurable mouse covariates.

## Failure modes the framework explicitly defends against

| Failure | Defense |
|---|---|
| Post-hoc prior selection | YAML preregistration pinned + hashed before cache freeze; runner refuses final report on mismatch |
| "Any network that fits is fine" | Within-mouse coverage curve must be calibrated (90% credible band contains ~90% of held-out trials); over-broad posteriors flagged |
| Per-mouse posteriors are not actually mouse-specific | 5×5 cross-likelihood matrix; diagonal must dominate by ≥0.5 nats per (mouse, muscle); permutation p-value reported |
| Spurious effects from sweep noise | Full label-shuffle null re-runs everything; real effects must collapse |
| Single-network domination | Effective sample size (ESS) reported per (likelihood, mouse); ESS < 5 → claim "population that mirrors mouse" is not supported by data |
| Likelihood choice driving conclusions | Three likelihoods reported separately, never averaged or ensembled |

## Architecture

```
vnl_playground/bayesian_emg/
  data.py                    # NetworkMouseFit cache (per-(network, mouse, trial) sim+empirical envelopes)
  likelihoods/
    correlation.py           # Option 1 — Fisher-z Gaussian on per-(mouse, muscle) Pearson r
    envelope.py              # Option 2 — per-trial per-timestep Gaussian on EMG envelope (with optional DTW)
    abc.py                   # Option 3 — ABC on pre-registered summary statistics
  posterior.py               # importance-reweighting → per-mouse weights, ESS, credible sets
  validation/
    coverage.py              # within-mouse posterior predictive coverage
    discrimination.py        # 5×5 cross-likelihood matrix
    permutation.py           # EMG↔mouse label-shuffle null
  bayes_factors.py           # design-vs-design model evidence comparison
  preregistration.py         # YAML loader + hash validator
  report.py                  # HTML/PDF aggregator
scripts/
  bayes_emg_build_cache.py   # one-time: discover networks → roll out → cache envelopes
  bayes_emg_run.py           # full pipeline: cache → likelihoods → posteriors → validation → report
configs/bayesian_emg/
  preregistration.yaml       # pinned hyperparameters (committed before cache freeze)
```

The trainer is not modified. The cache is the auditable substrate for every claim.

## Components

### Data layer (`data.py`, `bayes_emg_build_cache.py`)

A single Parquet/HDF5 store keyed by `(network_id, animal, trial_idx, muscle, timestep)`, with columns `sim_envelope`, `empirical_envelope`. Sidecar `networks.parquet` stores hyperparameters per network (`force_scale`, `joint_damping`, `shoulder_damping`, `control_cost`, `control_diff_cost`, `norm_method`, `train_animals`, `seed`, `wandb_run_id`).

Build sequence:

1. Glob `vnl_playground/checkpoints/<run_name>/`, filter to s17+ (extensible to s10–s16 via opt-in).
2. Pull hyperparameters from wandb (`_fields.*` and Command field), **not** from local `config.json` — config.json is unreliable when `--run-name` collides.
3. For each (network, animal): reuse `scripts/emg_comparison.py` rollout logic to produce per-trial sim envelopes at the same timestep grid as `process_emg_data` empirical envelopes.
4. Write to cache. Idempotent — skip (network_id, animal) if already present.

Empirical envelopes load via `process_emg_data` (already trial-level), stored once per (animal, trial, muscle).

Cache size estimate: ~50 trials × 60 timesteps × 3 muscles × 60 networks × 5 animals × 8 bytes ≈ 22 MB for the sim cube; empirical envelopes add <1 MB; with metadata and Parquet overhead expect <200 MB. Fits comfortably on local disk; no streaming needed.

Cache freeze: after build, the runner records the cache content hash. The preregistration YAML must reference this hash. Any change to the cache requires a new YAML version.

### Likelihoods (`likelihoods/`)

Uniform interface:

```python
class Likelihood(Protocol):
    name: str
    def log_likelihood(self, fit: NetworkMouseFit, mouse: str, *, holdout_trials: list[int] | None = None) -> float: ...
    def posterior_predictive(self, fits: list[NetworkMouseFit], weights: np.ndarray, mouse: str) -> CredibleBand: ...
```

**Option 1 — `correlation.py`.** Per (mouse, muscle), compute Pearson `r` between mean-trial sim and mean-trial empirical envelope. Apply Fisher-z transform `z = atanh(r)`. Sum `z` across muscles. Likelihood under Gaussian: `log p(EMG_m | θ) = -∑_μ (z_obs - z_θ(m,μ))² / (2 σ²_μ)` with `σ²_μ` estimated from across-seed scatter at fixed cells. For s17, σ² is bootstrapped from cross-seed variance in s11–s16 leader cells; for s18, σ² uses the multi-seed runs directly.

Cost: ignores amplitude bias and absolute scale — known and accepted for v1.

**Option 3 — `abc.py`.** Pre-registered summary statistic vector per (mouse, muscle):
- `onset_latency_ms` — first crossing of 30% trial-mean peak after reach start.
- `peak_amplitude` — max of trial-mean envelope.
- `peak_time_ms` — argmax of trial-mean envelope.

Plus per-mouse cross-muscle:
- `biceps_triceps_coactivation` — within-trial Pearson correlation of biceps and triceps envelopes, averaged across trials.
- `ad_recruitment_fraction` — `peak(AD) / max(peak(biceps), peak(triceps))`.

ABC distance is Mahalanobis on this vector, with covariance estimated from empirical inter-trial scatter (so the metric is naturally calibrated to noise scale). Acceptance threshold ε set as the 80th percentile of empirical inter-trial Mahalanobis distance — a network has to be no worse than a typical empirical trial.

The posterior weight under ABC is binary in the strict form (`accept` / `reject`); the framework also emits a soft kernel-weighted version `w ∝ K_h(d - 0)` with bandwidth `h` from Silverman's rule, for ESS-positive posteriors when strict acceptance is too sparse.

**Option 2 — `envelope.py`.** Per-trial per-timestep Gaussian: `EMG_t ~ N(sim_t, σ²_{m,μ})` where `σ²_{m,μ}` is per-(mouse, muscle) inter-trial empirical scatter. Both sim and empirical envelopes live on the `TARGET_TIMESTEPS=60`-step grid spanning the 250 ms reach window (4.17 ms/step). Optional DTW alignment with warp budget ≤20 ms (≤5 timesteps) to absorb timing jitter. Most sensitive to misspecification — ships last, when the calibration plot from Option 1 has told us how miscalibrated to expect.

All three run end-to-end and report separately. **No averaging or ensembling across likelihoods** — that would be the vibe-fitting trap. Each is a distinct claim.

### Posterior (`posterior.py`)

Importance-reweighting:

```
w_n^(m) ∝ exp(log p(EMG_m | θ_n) - log p_0(θ_n))
```

where `p_0` is the implicit uniform prior over swept cells. Normalize across networks. Report:

- Per-(likelihood, mouse) effective sample size `ESS = (∑ w_n)² / ∑ w_n²`.
- Per-(likelihood, mouse) credible set: smallest set of networks whose summed weight ≥ 0.90.
- Per-(likelihood, mouse) posterior mean and covariance over hyperparameters.

ESS < 5 prints a warning to the report and disqualifies the (likelihood, mouse) pair from generative-claim conclusions; validation tests still run.

### Validation suite (`validation/`)

All three tests run automatically for all three likelihoods. Each produces both a numerical output and a plot.

**`coverage.py` — within-mouse posterior predictive coverage.** Leave-trial-out per mouse:

1. Hold out 20% of mouse `m`'s trials.
2. Compute posterior weights from the remaining 80%.
3. For each held-out trial, compute the posterior predictive 90% credible band over networks (weighted quantiles).
4. Record whether the empirical trial envelope falls within the band at each timestep.
5. Repeat for nominal coverage levels {50, 80, 90, 95}%.
6. Report calibration curve: nominal vs. empirical coverage.

Acceptance: empirical coverage within ±5 points of nominal at 90%. Over-coverage (>97%) flagged as "posterior too broad — sweep covers everything." Under-coverage (<70%) flagged as "sweep does not contain this mouse."

**`discrimination.py` — between-mouse cross-likelihood matrix.** 5×5 matrix `L[i, j] = log p(EMG_j | posterior fit on EMG_i)`. Report:

- Diagonal-vs-off-diagonal mean: `Δ = mean(diag(L)) - mean(L - diag)`. Bar: `Δ ≥ 0.5` nats per (mouse, muscle).
- Permutation p-value: shuffle rows of L 10,000 times, recompute `Δ`, report fraction of shuffles ≥ observed.
- Heatmap with diagonal highlighted.

This is the killer test for goal C. If the matrix is uniform, the per-mouse posteriors are not actually mouse-specific.

**`permutation.py` — full EMG↔mouse label shuffle.** Randomly permute which empirical trial set is assigned to which mouse label. Re-run posterior + tests 1–2. Report null distribution of `Δ` and of coverage. Real effects must collapse under the null.

### Pre-registration (`preregistration.py`, `configs/bayesian_emg/preregistration.yaml`)

YAML file pins:

- σ² estimators per likelihood (which cells / which seeds → σ²)
- ABC summary statistics (names, computation specs, ε quantile)
- Coverage acceptance band (default ±5 pts at 90%)
- Discrimination diagonal-margin threshold (default 0.5 nats)
- Cache content hash (set when cache is frozen)
- Permutation seed and count (default 10,000)

The runner computes the YAML's SHA-256 hash and embeds it in every report. The runner refuses to produce a final report if the YAML hash doesn't match what was committed to git before the cache was frozen.

Changing the YAML after data inspection requires a new version tag and explicit acknowledgment in the report that the analysis is exploratory rather than confirmatory.

### Bayes factors (`bayes_factors.py`)

For two sweep slices A and B (e.g., moving-shoulder vs. static-shoulder, fs=1.1 vs. fs=1.0, z_baseline_x2 vs. p98_per_muscle), report per-mouse log Bayes factor:

```
log BF_{A,B}^(m) = log mean_{n ∈ A} L(EMG_m | θ_n) - log mean_{n ∈ B} L(EMG_m | θ_n)
```

and the cohort-summed log evidence. Each comparison's null is a permutation test on which networks belong to which slice (10,000 shuffles).

### Report (`report.py`)

Single HTML/PDF output:

- Header: cache hash, preregistration hash, git SHA, run timestamp.
- Per-likelihood section, each containing: per-mouse credible-set summary, posterior centroid table, coverage calibration plot, 5×5 discrimination heatmap with permutation p-value, ESS table.
- Bayes factor section: pre-specified comparisons with permutation nulls.
- Cross-likelihood comparison section: agreement table on which networks are in each per-mouse credible set across the three likelihoods.
- Findings flag list: any over-coverage, under-coverage, ESS<5, or YAML-hash mismatch is surfaced at the top.

## Build order

### Phase 1 (this week)

- Cache + ingestion (`data.py`, `bayes_emg_build_cache.py`).
- Option 1 (correlation likelihood).
- All three validation tests, plumbed through Option 1.
- Report v1.

Deliverable: cross-likelihood matrix + permutation null + coverage curve, on Option 1, on the existing s17 sweep. Either the matrix has a clear diagonal (we have a real framework) or it doesn't (and we know now, before investing in Options 2 and 3).

**Phase 1 gate:** if the discrimination matrix has no diagonal (Δ < 0.2 nats per (mouse, muscle), permutation p > 0.1), stop and revisit whether the sweep covers enough hyperparameter variation to produce mouse-distinguishing fits. The fix would be sweep design (broader prior, more cells per mouse), not framework changes.

### Phase 2 (after Phase 1 passes)

- Pre-registration YAML committed and hashed.
- Option 3 (ABC).
- Re-run §4 tests on Option 3.
- Bayes factors for s17 design comparisons (moving-shoulder vs. legacy, norm method, train_animals choice).

**Phase 2 gate:** if Option 3 disagrees with Option 1 on which networks are in each per-mouse credible set, that's a finding worth reporting (different likelihoods favor different solutions on the manifold), not a bug. Both reported side by side.

### Phase 3 (after Phase 2 is consistent with Phase 1)

- Option 2 (Gaussian envelope, with DTW).
- Final cross-likelihood report.
- Plan multi-seed s18+ runs at the per-mouse posterior modes to estimate σ² from real seed scatter rather than across-cell scatter.

## Falsifiable predictions

1. **Phase 1 cross-likelihood matrix has Δ ≥ 0.5 nats per (mouse, muscle), permutation p < 0.01.** Per-mouse posteriors are mouse-specific.
2. **Coverage curve at 90% lies within [85%, 95%] for at least 4 of 5 mice under Option 1.** Sweep is approximately calibrated.
3. **Per-mouse posterior centroids in hyperparameter space cluster by animal, not by sweep cell.** PCA across mice shows the leading components separate animals more than nuisance hyperparameters.
4. **ABC and correlation likelihoods agree on credible-set membership for ≥60% of networks per mouse.** Different likelihoods pick out overlapping but non-identical regions of the solution manifold.
5. **Cohort-trained models (s17 M1) have higher per-mouse log-evidence than specialists for AT animals.** Cohort training generalizes; if false, specialists are needed and the cohort claim from s17 is overstated.

## Out of scope (deferred)

- **Latent manifold structure across mice** — is there a low-dimensional manifold of valid mouse motor solutions? Separate spec.
- **EMG in the reward** — remains eval-only per s17 spec.
- **Cross-task transfer** — predicting EMG on a held-out task.
- **Multi-mouse hierarchical priors** — treating mice as draws from a meta-population. Needs >5 mice to be honest.
- **Continuous-prior Bayesian inference** — replacing importance reweighting with neural posterior estimation (SBI). Worth doing eventually; not Phase 1–3.

## Decision gates before starting Phase 1

- [ ] Confirm s17 cache scope: 22 cells × applicable animals = network list locked.
- [ ] Confirm `scripts/emg_comparison.py` rollout logic is the canonical sim-envelope source (not a divergent copy).
- [ ] Confirm trial indexing in `process_emg_data` matches the trial indexing assumed by the per-animal eval in s17.
- [ ] σ² bootstrap source agreed: which s11–s16 cells contribute to the cross-seed variance estimator for Option 1.
- [ ] Cache storage location agreed (default: `vnl_playground/bayesian_emg/cache/v1.parquet`, gitignored).
