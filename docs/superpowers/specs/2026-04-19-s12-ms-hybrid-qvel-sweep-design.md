# s12-ms: Hybrid-regime sweep design

**Date:** 2026-04-19
**Status:** design

## Goal

Find a region of parameter space that simultaneously satisfies:

1. `eval/episode_reward ≥ 400` (the kinematic-fidelity bar)
2. `eval/emg_biceps_corr ≥ 0.7` (shape fidelity — s10's strength)
3. `eval/emg_biceps_mae ≤ 0.17` (amplitude fidelity — s11's strength)

No point in the combined s10+s11 data (237 runs) meets all three. s11 hits (1) and (3) but caps bcorr at ~0.66. s10 reaches bcorr=0.90 but never exceeds R=365.

## Context

### What s10 and s11 taught us

| sweep | best bcorr | R at that run | best R (R≥400) | cc, cdc at best bcorr |
|---|---|---|---|---|
| s10 | **0.90** | 344 (fs=0.7, damp=5e-7) | none ≥ 400 | cc=0.05, cdc=0.1 |
| s11 | 0.66 | 379 (fs=1.0, damp=3e-7) | **409** (bcorr=0.60) | cc=0.05, cdc=0 |

Key conclusions from heatmaps in `plots_s12_hybrid/`:

- **`cc` is the shape knob.** bcorr climbs from −0.27 to 0.66 as cc goes 0 → 0.05 at fs=1.0 (plot 14). Flat above.
- **`cdc` barely affects shape** but does drag reward. Pushing cdc=0 adds ~20 reward with no shape penalty.
- **`damp` past 5e-7 was never tested at fs=1.0.** s10 got bcorr=0.90 at damp=5e-7 fs=0.7 — might transfer.
- **`tau_deact` has a single s10 pilot at fs=0.3** showing drop from 0.04 → 0.03 raises bcorr from 0.29 → 0.63 (n=1). Never tested at fs=1.0.
- **`qvel_init`** has never been varied in ms. Bio biceps peaks at t=20ms; sim always at t≥30ms. Reference qvel might allow the policy to fire immediately instead of accelerating from rest.

### R≥400 constraint

All 16 R≥400 runs in the pooled data are s11 at `fs=1.0`. s10's ceiling is R=365. **fs=1.0 is non-negotiable** if we insist on R≥400.

## Grid

| axis | values | n |
|---|---|---|
| `force_scale` | 1.0 | 1 |
| `joint_damping` | 3e-7, 5e-7, 8e-7 | 3 |
| `control_cost` | 0.05 | 1 |
| `control_diff_cost` | 0.0, 0.025 | 2 |
| `qvel_init` | zeros, reference | 2 |
| `muscle_tau_deact` | 0.03, 0.04 | 2 |
| `muscle_tau_act` | 0.01 (fixed at default) | 1 |
| seeds | 2 per cell | 2 |

**48 runs total.** 24 cells × 2 seeds. All other task/training hyperparameters frozen at s11-ms defaults for direct comparability.

### Rationale for each axis

- **damp 3e-7, 5e-7, 8e-7.** Extends s11's low-damp regime (which tops out at 5e-7) into s10's range without going to s10's extremes (9e-7, 1e-6) where s11 would pay a reward penalty. 8e-7 is the novel bet.
- **cdc {0, 0.025}.** s11's R≥400 winners split between these; cdc=0 is the reward-optimal, 0.025 is a mild-regularization backstop.
- **qvel_init {zeros, reference}.** Tests whether reference-initialization lets policies achieve the early biceps onset that distinguishes real mouse EMG. `"reference"` mode already exists at `vnl_playground/tasks/mouse/imitation.py:278-286` — just needs the CLI flag set.
- **tau_deact {0.03, 0.04}.** s10's n=1 pilot at fs=0.3 showed bcorr doubled (0.29 → 0.63) when deact dropped from 0.04 → 0.03. We're testing it at the regime we actually care about (fs=1.0 R≥400).
- **2 seeds.** Compromise between budget and noise estimation; s11 seed-to-seed variance in bcorr is ~±0.05 based on adjacent-cell spread, so 2 seeds separate real effects from noise for the differences we care about (~0.1-0.2 in bcorr).

### Why NOT these axes in s12

- **`fs < 1.0`:** Uncompetitive under R≥400 constraint. s10 showed at fs=0.7 bcorr=0.90 is achievable but reward maxes at 344. No cheap way to fix that without changing training budget.
- **`cc ≠ 0.05`:** Plot 14 shows cc=0.05 already near the shape maximum at fs=1.0. Lower cc drops shape; higher cc drops reward (s11 cc=0.1 median R=382 vs cc=0.05 median R=394).
- **`saturation_cost`:** Strong candidate but displaced by tau_deact. Deferred to s13 if s12 fails to bridge the gap.
- **`muscle_tau_act`:** s10 pilot showed raising tau_act from 0.01 aggressively destroys bcorr. Keep fixed at default.
- **`num_timesteps`:** Separate question (is s10 undertrained?) — worth a focused experiment, not a sweep axis.

## Success criteria

s12 succeeds if at least one cell produces, averaged over its 2 seeds:

- R ≥ 400
- bcorr ≥ 0.7
- bmae ≤ 0.17

Partial success: any single run (not cell average) meeting all three criteria — worth replication but not a declared answer.

If *no* cell meets all three, the likely diagnosis is that fs=1.0 is incompatible with higher damping for reward, and s13 should pursue: (a) longer training at fs=0.7-0.8 to see if s10-regime policies catch up, (b) saturation_cost at fs=1.0 as the anti-bang-bang regularizer, or (c) direct EMG-trace supervision as a reward term.

## Execution notes

- Launcher pattern: copy `sweep_s11_ms_6.sh` as `sweep_s12_ms.sh`, adapt the axis loops.
- Tag: `s12-ms`.
- All runs use `train_mouse_janelia_sigmoid_moving_shoulder.py` (moving-shoulder env).
- Store results to a new `s12_ms.csv` wandb export; reuse `plot_s12_hybrid.py` for analysis with that CSV.
- The tau_deact override flows through `mj_model.actuator_dynprm[:, 1]` (see `vnl_playground/tasks/mouse/base.py:292`) — applies to all 12 actuators uniformly. A per-muscle override is possible but not needed for this sweep.

## Risks

- **qvel_init=reference might destabilize early training.** Previous round-3 data showed it underperforms top-5 defaults (see `train_mouse_janelia_sigmoid_moving_shoulder.py:435-438`). We're re-testing at better parameters.
- **tau_deact changes the muscle plant itself** — reward numbers are only directly comparable within same tau_deact. Cell comparisons should group by tau_deact.
- **48 runs at current per-run budget** — estimate wallclock against current queue before launch.
