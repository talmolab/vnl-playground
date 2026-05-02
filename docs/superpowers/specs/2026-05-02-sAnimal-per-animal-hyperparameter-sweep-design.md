# sAnimal — per-animal hyperparameter fractional-factorial sweep

**Status:** spec, 2026-05-02.
**Companion:** `2026-05-01-s17-ms-multi-animal-design.md` (multi-animal eval infrastructure this builds on); `2026-05-02-hierarchical-bayesian-emg-population-design.md` (consumes sAnimal outputs as the per-animal posterior input).

## Goal

For each of the 5 animals, train a small per-animal hyperparameter grid on that animal's kinematics only, then compare the per-animal best cells. Tests whether different mice want different physics + penalty parameters — a richer per-animal characterization than s17 Track A (which only varied hyperparameters for A36-1, via the single A6 cell).

The output is the per-animal "best cell" plus the full per-animal response surface across `(force_scale, joint_damping, control_cost, control_diff_cost)`. This becomes the input population for the hierarchical Bayesian framework's per-mouse posterior.

## What's new vs s17 Track A

| Question | s17 Track A | sAnimal |
|---|---|---|
| Per-animal training | ✓ (one cell each) | ✓ |
| Per-animal hyperparameter grid | only A36-1 (A6 cell) | all 5 animals |
| Axes varied | none (or 1) | 4: fs, jd, cc, cdc |
| Cells per animal | 1 (or 2 for A36-1) | 9 (8 fractional + 1 center) |
| Total runs | 5 (+1) | 45 |

## Sweep design

A 2^(4-1) resolution-IV fractional factorial on `{fs, jd, cc, cdc}`, plus one center cell. The fractional design sits at 8 of the 16 corners of the 4D cube — main effects are clear of two- and three-factor interactions, two-factor interactions are confounded in pairs (acceptable for screening). The center cell anchors all comparisons and detects pure curvature against the s16 leader.

**Held fixed across all 45 runs:** `shoulder_damping = 6e-7` (s17 default); moving-shoulder XML; v16 5-animal clip dir; `z_baseline_x2` EMG normalization; `seed = 0`; everything in the s17 "Held fixed" block (ctrl_dt, sim_dt, episode_length, joint_armature, qvel_init, joints/wrist/bodies weights, num_timesteps=8e8, num_evals=8).

**Coding for the fractional design:**
- `fs` low/high = 1.0 / 1.2
- `jd` low/high = 1e-6 / 2e-6
- `cc` low/high = 0.0 / 0.05
- `cdc` sign = sign(fs) × sign(jd) × sign(cc) (resolution-IV generator D = ABC)

### Cells (per animal)

| Cell | fs | jd | cc | cdc | Note |
|---|---|---|---|---|---|
| **C0** | 1.1 | 1.5e-6 | 0.025 | 0.025 | center, s16 leader |
| **F1** | 1.0 | 1e-6 | 0.0 | 0.0 | weak elbow, no penalty |
| **F2** | 1.2 | 1e-6 | 0.0 | 0.05 | strong forces, weak elbow, smoothness only |
| **F3** | 1.0 | 2e-6 | 0.0 | 0.05 | weak forces, stiff elbow, smoothness only |
| **F4** | 1.2 | 2e-6 | 0.0 | 0.0 | strong forces, stiff elbow, no penalty |
| **F5** | 1.0 | 1e-6 | 0.05 | 0.05 | weak forces, weak elbow, full penalty |
| **F6** | 1.2 | 1e-6 | 0.05 | 0.0 | strong forces, weak elbow, magnitude only |
| **F7** | 1.0 | 2e-6 | 0.05 | 0.0 | weak forces, stiff elbow, magnitude only |
| **F8** | 1.2 | 2e-6 | 0.05 | 0.05 | strong forces, stiff elbow, full penalty |

Replicated identically for each of `{A36-1, AT006, AT009, AT012, AT013}`. Run name pattern: `sAnimal-<animal>-<cell>-<timestamp>`. Wandb tag pattern: `<animal>-<cell>` (e.g., `A36-1-C0`, `AT006-F3`).

### Eval scope

Every run is eval'd against **all 5 animals** (per-animal `eval/emg_<animal>_<muscle>_<metric>` keys, already wired by s17). This gives the cross-animal generalization signal — a per-animal model trained on AT012 should fit AT012 best but may also fit AT013 well if their motor strategies overlap. The cohort-mean curve is also logged for back-compat.

## Hypotheses

**H1 — per-animal optima differ.** At least 2 of the 5 animals have a best cell that is not C0 (the s16 leader). If false, the s16 leader generalizes across animals and per-animal hyperparameter tuning is unnecessary.

**H2 — `fs` is the dominant per-animal axis.** Across animals, the largest within-animal swing in cohort-mean correlation is on the `fs` axis. (s10–s16 lesson; we test whether it holds at the per-animal level.)

**H3 — small mice prefer lower force.** AT* animals (smaller than A36-1) achieve their best per-animal correlation at `fs ≤ 1.1`; A36-1 achieves its best at `fs ≥ 1.1`. If true, body-size scaling of `fs` is empirically defensible.

**H4 — `cc × cdc` interaction matters per animal.** The fractional-IV design confounds `cc × cdc` with `fs × jd`; if either main effect dominates the F1–F8 spread, we can decouple. If both look equal, we'll need the full 2^4 (B3 in the design dialogue) in a follow-up.

## Falsifiable predictions

"Best cell for animal X" is defined as the cell in {C0, F1..F8} that maximizes the mean of X's `biceps_corr` and `triceps_corr` (AD ignored for ranking — too sparse on AT* animals).

1. Each per-animal best cell achieves `biceps_corr ≥ 0.6` and `triceps_corr ≥ 0.55` on its own animal. If not, the s16-leader-relative grid is too narrow for that animal — widen and re-run.
2. The per-animal best cells span at least 3 distinct grid points across the 5 animals (i.e., not all 5 land on C0). If they all collapse to C0, H1 is falsified — abandon per-animal hyperparameter tuning.
3. AT012's best cell has `fs ≤ 1.1`; A36-1's best cell has `fs ≥ 1.1`. If either is reversed, H3 is falsified.

## Pipeline

1. **No trainer changes.** Uses existing `--train-animals`, `--force-scale`, `--joint-damping`, `--shoulder-damping`, `--control-cost`, `--control-diff-cost`, `--emg-animals`, `--emg-norm-method`, `--reference-data-path`. All shipped in s17.
2. **Smoke test before launch.** Single-cell C0 run for `A36-1` with `--num-timesteps 5_000_000 --num-evals 1`. Confirm: per-animal eval loads for all 5 animals, EMG plots render, no NaN.
3. **Launch.** 6 sweep scripts (matching s17 GPU pattern), 7–8 cells per script, `BUDGET_HOURS=30` to absorb variance. Wall budget ~26–28h.
4. **Post-sweep analysis.**
   - Per-animal heatmap: 9 cells × 5 animals × 3 muscles, cohort-mean correlation. Identify each animal's best cell.
   - Cross-animal heatmap: per-animal-best model vs all 5 animals' EMG. Diagonal should dominate.
   - Main-effects table: per-animal mean effect of `fs`, `jd`, `cc`, `cdc` (each axis estimated from the 4 cells where the sign is +). Sign + magnitude tells us which axis matters per animal.
   - C0-vs-best gap per animal: how much better is the per-animal best than the universal s16 leader?
5. **Hand-off to Bayesian framework.** The 45 trained checkpoints become the network population for the per-mouse Bayesian posterior (Phase 1 of the Bayesian framework spec). The per-animal cross-mouse cell layout — same 9 cells per mouse — is what makes the framework's importance reweighting well-conditioned.

## GPU partition (matches s17 layout)

6 sweep scripts, one per GPU; budget 30h each. Cells distributed across animals so a single script doesn't bottleneck on one animal:

- `sweep_sAnimal_1.sh` — GPU0 — A36-1: C0, F1, F2, F3, F4 (5 cells)
- `sweep_sAnimal_2.sh` — GPU1 — A36-1: F5, F6, F7, F8 + AT006: C0, F1, F2 (7 cells)
- `sweep_sAnimal_3.sh` — GPU2 — AT006: F3, F4, F5, F6, F7, F8 (6 cells)
- `sweep_sAnimal_4.sh` — GPU3 — AT009: all 9 cells (9 cells)
- `sweep_sAnimal_5.sh` — GPU4 — AT012: all 9 cells (9 cells)
- `sweep_sAnimal_6.sh` — GPU5 — AT013: all 9 cells (9 cells)

Per-script wall time: 5–9 cells × 3.5h = 17–32h. The 30h budget covers the longest script with margin. Scripts have early-exit on budget like s17.

## Decision gates before launching

- [ ] Smoke C0 run for A36-1 succeeds in <10 min (5M timesteps, 1 eval).
- [ ] Per-animal EMG eval loads cleanly for all 5 animals from the smoke run's wandb dashboard.
- [ ] All 6 sweep scripts have correct `CUDA_VISIBLE_DEVICES`, `WANDB_GROUP=sAnimal-part{N}`, and budget envs.
- [ ] No s18 or other sweeps holding the GPUs (check `nvidia-smi` and `nohup` jobs).

## Out of scope (deferred)

- Multi-seed per cell — not in this sweep. Phase 1 of the Bayesian framework uses across-cell scatter as a σ² proxy; if it's inadequate, a follow-up `sAnimal-seed` sweep adds 3-seed replicates at each per-animal best cell.
- `shoulder_damping` axis — held fixed at 6e-7. Adding it doubles the design size.
- Full 2^4 design — only used as a fallback if H4 turns out to require disambiguating `cc × cdc` from `fs × jd`.
- Per-animal XML / body-parameter calibration — separate spec; this sweep only varies the policy training cell, not the body model.
- Animal-conditioning embeddings (one-hot input) — superseded by the per-animal training approach.
