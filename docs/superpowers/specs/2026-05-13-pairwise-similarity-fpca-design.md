# Pairwise kinematic / EMG similarity + fPCA comparison for the Physics-Aware checkpoint

**Status:** spec, 2026-05-13.
**Owner:** eric@talmolab.org.
**Companion:** `2026-05-02-hierarchical-bayesian-emg-population-design.md` (manifold framing this spec instantiates a small piece of), `notebooks/talk_figures/talk_emg_figures.py` and `pca_figures.py` (sibling figure scripts using the same checkpoint and caches).

---

## Motivation

The `talk_figures/` pipeline already produces per-trial EMG comparison (`fig1–3`) and PCA-of-activations plots (`fig6/12/13`) for the Physics-Aware s18 checkpoint (`s18-ms-F4-fs1p2-20260502-014751`) across the 5-animal cohort (278 clips). What's missing is the *manifold* view: how reaches relate to each other, in both biology and simulation, in both kinematics and EMG. Concretely:

- **Does bio reach structure cluster by animal?** A pairwise cosine heatmap over the 278 bio kinematic trajectories (rows/cols sorted by animal) will either show block-diagonal structure (animals have distinct reach styles) or near-uniform similarity (one shared motor strategy).
- **Does the Physics-Aware network preserve that structure?** The same pairwise heatmap built from the network's rolled-out kinematics and from its simulated muscle activations tells us whether the policy reproduces the empirical between/within-animal block pattern, or collapses it.
- **Are bio EMG and sim muscle activations living on similar manifolds?** Functional PCA on each, then principal-angle comparison of the resulting bases, gives a basis-level overlap score that complements the per-trial-MAE numbers already in `talk_emg_figures.py`.

This is a focused, no-rollout, post-hoc analysis. All inputs already exist on disk (the talk-figures pipeline cached them). The deliverables are one new Python script under `notebooks/talk_figures/` and one handoff markdown so future sessions can pick up the talk-figures context without re-deriving paths.

## Goals

A. **Bio pairwise structure** — produce 278×278 cosine-similarity heatmaps for bio kinematics and bio EMG across all 5 animals, sorted to make animal-block structure visible.

B. **Sim pairwise structure (same checkpoint)** — produce the analogous 278×278 heatmaps from the Physics-Aware network's rollout kinematics and simulated muscle activations.

C. **Block-structure summary** — for each of the four matrices, report mean within-animal vs between-animal cosine.

D. **fPCA basis comparison (bio EMG vs sim muscle activations)** — build an fPCA basis from each, report principal angles between them, and plot mode shapes + a projection scatter.

E. **Handoff doc** — a `README.md` under `notebooks/talk_figures/` so the next session can identify checkpoints, caches, scripts, shapes, and conventions in one read.

## Non-goals (deferred)

- Per-reach sim-vs-bio scatter `(cos(sim_kin_i, bio_kin_i), cos(sim_emg_i, bio_emg_i))` — useful but deferred to a follow-up.
- Cross-modality scatter `(pairwise bio_kin_sim, pairwise bio_emg_sim)` — the Bernstein-redundancy probe. Deferred.
- Asymmetric sim×bio cross-heatmap. Deferred.
- Mantel / Procrustes between bio and sim pairwise matrices. Deferred (belongs with the Bayesian-population manifold-overlap claim).
- Fine-tuning per animal. Separate spec.
- The Joint-Reward-Only checkpoint (`s18-ms-C1-cc0-cdc0`). The script supports an arbitrary `--network`, but this spec only commits to running the Physics-Aware checkpoint.

## Architecture

```
notebooks/talk_figures/
├── README.md                       # handoff doc (new)
├── similarity_pairwise.py          # new analysis script
├── talk_emg_figures.py             # existing — sibling
├── pca_figures.py                  # existing — provides FK + rollout helpers (importable)
├── tracking_kinematics_figures.py  # existing — FK conventions to mirror
├── extract_pre_onset_emg.py        # existing
├── build_notebook.py               # existing
└── figs/
    ├── fig_sim_heatmap_bio_kin.{pdf,png}
    ├── fig_sim_heatmap_bio_emg.{pdf,png}
    ├── fig_sim_heatmap_sim_kin.{pdf,png}
    ├── fig_sim_heatmap_sim_emg.{pdf,png}
    ├── fig_sim_block_summary.{pdf,png}
    ├── fig_fpca_modes.{pdf,png}
    └── fig_fpca_bio_basis_scatter.{pdf,png}
```

## Inputs

All inputs already exist; the script reads from cache.

| Source | Path | Per-reach shape | Notes |
|---|---|---|---|
| Bio kinematics | `vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/*_ik.h5` → `kp_data` | `(50, 9)` float32 | STAC mocap markers (Shoulder, Elbow, Wrist) × 3 coords, 50 steps × 5 ms = 250 ms. |
| Bio EMG | `vnl_playground/bayesian_emg/cache/v1/envelopes/<network>/<animal>.npz` → `empirical` | `(n_trials, 60, 3)` | (AD, Triceps, Biceps) post-onset envelopes, 60 steps × 4.17 ms = 250 ms. Per-(network, animal) cache; same `empirical` across networks for a given animal. |
| Sim EMG | same .npz → `sim` | `(n_trials, 60, 3)` | Physics-Aware network's simulated muscle activations on the same grid. |
| Sim kinematics | `notebooks/talk_figures/figs/rollout_activations/<network>_278clips.npz` → `qposes_rollout` | `(278, 100, 7)` float32 | Full rollout, 100 steps × 4.17 ms. Slice `[:, :60, :]` to align with the post-onset 250 ms window, then FK to (60, 3 markers, 3 coords) = (60, 9). FK conventions: `tracking_kinematics_figures.py` (`KP_BODY = {"Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist"}`, IK XML at `/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml`). |

Animal ordering for stable indexing: `("A36-1", "AT006", "AT009", "AT012", "AT013")`. Per-animal counts: `46, 60, 45, 83, 44` (Σ = 278). Within each animal, trials are concatenated in the EMG cache's `trial_idx` order; the rollout cache `qposes_rollout` is reordered to match by animal+trial_idx using the bio reference filenames as the join key. The script records `animal_labels[i]` and `trial_idx[i]` for every row of the pairwise matrices.

Time alignment: bio `kp_data` is 50 steps at 5 ms; sim is 60 steps at 4.17 ms. Both span 250 ms post-onset. Bio kin is linearly resampled from 50→60 along the time axis so cross-modality comparison uses one common 60-step grid. EMG envelopes are already on the 60-step grid in the cache.

## Pipeline

### Step 1 — Build the four feature matrices `X ∈ ℝ^{N×D}`

For each modality, stack one row per reach (N = 278). Each row is the flattened, L2-normalized feature vector for that modality:

| Matrix | Build | D |
|---|---|---|
| `X_bio_kin` | Load `kp_data` per reach (50, 9), resample to (60, 9), flatten, L2-normalize. | 540 |
| `X_bio_emg` | Load `empirical[i]` per reach (60, 3), flatten, L2-normalize. | 180 |
| `X_sim_kin` | Load `qpos[i]` (60, 7), FK to (60, 9) marker positions with STAC XML, flatten, L2-normalize. | 540 |
| `X_sim_emg` | Load `sim[i]` (60, 3), flatten, L2-normalize. | 180 |

Centering choice: we use the raw envelope / position vector (no mean subtraction) so cosine reflects shape *including* baseline offsets. This is the same convention as the existing per-trial MAE in `talk_emg_figures.py`. A `--center` flag is exposed but defaults to off.

### Step 2 — Pairwise cosine matrices

For each of the four matrices, `S = X @ X.T` (since rows are L2-normalized, `S[i,j] = cos(x_i, x_j)`). All four matrices share the same row/column index order, so they can be compared directly.

### Step 3 — Plot heatmaps

Each heatmap uses identical layout for visual comparability:

- Rows/cols sorted by `(animal, trial_idx)` so the 5 animal blocks appear along the diagonal.
- Thin white separator lines between animals.
- Per-animal labels along the left/bottom.
- Colorbar fixed to `[-1, 1]` (cosine can be negative for the kinematics rows because positions span both signs in body-local frame; EMG cosine will stay in `[0, 1]` since envelopes are non-negative).

### Step 4 — Block-structure summary

For each matrix `S`, compute:

```
within = mean(S[i, j])  for i, j with same animal, i ≠ j
between = mean(S[i, j]) for i, j with different animals
gap = within − between
```

Produce one bar chart `fig_sim_block_summary` with 4 grouped bars (bio_kin, bio_emg, sim_kin, sim_emg), each group showing `within`, `between`, and `gap`. Print the table to stdout for the spec's expected-output check.

### Step 5 — Rankings (auxiliary, saved as `.npz`, not plotted)

For each matrix and each reach `i`, sort the row by descending cosine and save:

```
top10_idx[i]      # (10,) int — top-10 nearest reaches (excluding self)
top10_cos[i]      # (10,) float
top10_animal[i]   # (10,) str — animal label of each neighbor (for animal-purity stat)
```

Save to `figs/similarity_rankings.npz` with one set per modality. No plot — this is a data drop for downstream analysis.

### Step 6 — fPCA bases (bio EMG vs sim muscle activations)

Build two PCA bases:

```python
X_bio = X_bio_emg − mean(X_bio_emg, axis=0)      # (N, 180), per-feature centered
X_sim = X_sim_emg − mean(X_sim_emg, axis=0)

U_bio, S_bio, _ = svd(X_bio, full_matrices=False)
U_sim, S_sim, _ = svd(X_sim, full_matrices=False)
```

The functional basis is `V_bio = Vt_bio[:k].T` (i.e., the right singular vectors of the centered data), where `k` is chosen as the smallest integer satisfying `cumulative_variance(S_bio**2) ≥ 0.85`. Same for `V_sim`. Expected `k ∈ {3, 4, 5}` based on the EMG dimensionality in the briefing doc.

Diagnostics:

(a) **Principal angles** `θ_1 ≤ … ≤ θ_k` between `V_bio` and `V_sim` via `scipy.linalg.subspace_angles`. Report `cos(θ_i)` for each and the mean. Print to stdout.

(b) **Mode shapes**: each basis vector reshapes back to `(60, 3)` — overlay bio mode `j` (solid) vs sim mode `j` (dashed) for `j = 1..min(k, 3)`, one subplot per muscle (AD, Triceps, Biceps) and one column per mode. Save `fig_fpca_modes.{pdf,png}`.

(c) **Projection scatter**: project both `X_bio_emg` and `X_sim_emg` (centered with their respective means) onto `V_bio`, plot PC1 vs PC2 with marker style indicating bio/sim and color indicating animal. Save `fig_fpca_bio_basis_scatter.{pdf,png}`. (Using only the bio basis keeps the axes interpretable; the sim-basis equivalent is computable but redundant for this first pass.)

## CLI

```
python notebooks/talk_figures/similarity_pairwise.py \
  --network s18-ms-F4-fs1p2-20260502-014751 \
  [--out-dir notebooks/talk_figures/figs] \
  [--center] \
  [--k-fpca auto]
```

`--network` is required; defaults match `talk_emg_figures.py` constants if omitted. `--k-fpca` accepts `auto` (the 85%-variance rule) or an integer.

## Implementation notes

- Reuse `pca_figures.py` for the rollout-qpos loader if it already exposes one; otherwise load directly from the `_278clips.npz` file. **Do not** re-roll; this is a fast post-hoc script.
- FK from qpos to marker positions: mirror the convention in `tracking_kinematics_figures.py` exactly. The STAC XML at `/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml` has the `shoulder_base` body needed for the correct body local frames; the vnl IK XML does not.
- Matplotlib style: match `talk_emg_figures.py` (`pdf.fonttype = 42`, `font.size = 8`, `figure.facecolor = "w"`, `savefig.dpi = 300`).
- The script must be runnable from a CPU-only node (`JAX_PLATFORMS=cpu`, `MUJOCO_GL=egl` for FK). Matches `pca_figures.py` and `tracking_kinematics_figures.py` env setup.
- Pure NumPy / SciPy for similarity and PCA; no JAX needed except for the FK step (and only if a JAX MuJoCo path is more convenient than CPU `mujoco.MjData`).

## Test plan

1. **Smoke run**: `python similarity_pairwise.py --network s18-ms-F4-fs1p2-20260502-014751` completes in under 60 seconds on CPU, produces all 7 figures, and prints the four within/between/gap rows.
2. **Shape checks** logged to stdout: `X_bio_kin.shape == (278, 540)`; `X_bio_emg.shape == (278, 180)`; sim shapes match. All cosines in `[-1, 1]` (and in `[0, 1]` without `--center` for non-negative envelopes).
3. **Heatmap diagonal sanity**: each pairwise matrix has `S[i,i] == 1.0` (within float tolerance). The script asserts this.
4. **Block-summary sanity**: for `S_bio_kin`, `within > between` by a non-trivial margin (≥ 0.02). This is a weak prior; if it's violated, sorting or animal labels are likely mis-assigned.
5. **fPCA variance check**: `cumulative_variance(S_bio**2)[k_bio − 1] ≥ 0.85` and same for sim. `k_bio, k_sim ∈ {3, 4, 5}` (warning printed if outside that range).

## Falsifiable predictions

1. **Bio-kin block structure exists**: `gap_bio_kin ≥ 0.05`. If not, animals share one kinematic style on this dataset and per-animal analyses lose most of their motivation.
2. **Bio-EMG block structure is weaker than bio-kin block structure**: `gap_bio_emg < gap_bio_kin`. Manifold-framing prediction — EMG is one redundant draw per kinematic trajectory.
3. **Sim preserves bio kin block structure**: `gap_sim_kin ≥ 0.5 × gap_bio_kin`. If not, the Physics-Aware policy is averaging over animals despite being trained on the 5-animal cohort.
4. **fPCA mean principal angle small**: `mean(θ_i) ≤ 30°` between bio-EMG and sim-EMG bases. Weak overlap → sim EMG is on a different manifold than bio, regardless of per-trial MAE.

## Handoff doc (`notebooks/talk_figures/README.md`)

A separate deliverable (not a code file): a ~2-page markdown reference covering everything a new session needs to resume the talk-figures work without re-deriving paths.

Sections:

1. **What's here.** One-line summary of each `.py` file and the figures it produces.
2. **Checkpoints.** `s18-ms-F4-fs1p2-20260502-014751` (Physics-Aware: `fs=1.2`, `cc=0.025`, `cdc=0.025`) — primary; `s18-ms-C1-cc0-cdc0-20260502-051429` (Joint Reward Only: `cc=0`, `cdc=0`) — contrast. Both clear the >400 kinematic-fit bar.
3. **Caches.**
   - `vnl_playground/bayesian_emg/cache/v1/envelopes/<network>/<animal>.npz` — per-(network, animal) EMG envelopes. Arrays: `sim`, `empirical`, both `(n_trials, 60, 3)` over `(AD, Triceps, Biceps)`, 250 ms post-onset.
   - `notebooks/talk_figures/figs/rollout_activations/<network>[_278clips].npz` — rollout qpos + sown intention/decoder activations.
   - `notebooks/talk_figures/figs/pre_onset_cache/` — extended `-50..+250 ms` EMG envelopes (72 steps).
   - `vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/*_ik.h5` — bio kinematics (`kp_data`, `marker_sites`, `qpos`, `qvel`) at 5 ms / step over 50 steps.
4. **Cohort.** `("A36-1", "AT006", "AT009", "AT012", "AT013")`, 278 clips total.
5. **Coordinate / FK convention.** Use the STAC XML (`/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml`) for FK on rollout qpos so sim markers match bio `kp_data` frames. `KP_BODY = {"Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist"}`. The vnl IK XML lacks the `shoulder_base` body — do not use it for FK.
6. **Time alignment.** EMG cache is post-onset 60 steps × 4.17 ms = 250 ms. Bio kin is 50 steps × 5 ms = 250 ms; resample bio kin to 60 steps for cross-modality work. Pre-onset cache extends to −50 ms.
7. **Env setup.** `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `JAX_PLATFORMS=cpu`. `_MJX_PLAYGROUND_SRC = /root/vast/scott-yang/mujoco_playground` is sys-path'd in all scripts.
8. **Style.** Matplotlib: `pdf.fonttype=42`, `ps.fonttype=42`, `font.size=8`, `figure.facecolor="w"`, `savefig.dpi=300`. Palette: purple `#8E44AD` = empirical EMG, blue `#1f77b4` = Physics-Aware, orange `#ef7307` = Joint Reward Only.
9. **Figure inventory.** Brief list of which fig number comes from which script (e.g., `fig1-3` from `talk_emg_figures.py`; `fig6, 12, 13` from `pca_figures.py`; `fig8-11` from `tracking_kinematics_figures.py`; `fig_sim_*` from `similarity_pairwise.py`).
10. **Pick up here prompt.** A copy-pasteable paragraph for the next session: "talk_figures/ is built around the s18-ms-F4-fs1p2-20260502-014751 physics-aware checkpoint and there's already a per-(network, animal) envelope cache at `vnl_playground/bayesian_emg/cache/v1/envelopes/<network>/<animal>.npz` storing `sim` and `empirical` shape (n_trials, 60, 3) for (AD, Triceps, Biceps). Rollouts already exist; no need to re-roll. To pick up: read `notebooks/talk_figures/README.md` for full paths."

## Execution order

1. Write `notebooks/talk_figures/README.md` (the handoff doc).
2. Write `notebooks/talk_figures/similarity_pairwise.py` per the architecture above.
3. Smoke-run on `s18-ms-F4-fs1p2-20260502-014751`; confirm Test-plan items 1–5 pass.
4. Inspect figures; if predictions 1–4 are dramatically violated, sanity-check FK alignment and animal sorting before drawing conclusions.
5. Commit script + README + figures.

## Decision gates

- [x] User confirmed scope: bio + sim pairwise separately, plus fPCA EMG-vs-sim-actions. No per-reach scatter, no cross-modal scatter, no fine-tuning in this spec.
- [x] User confirmed handoff doc is in scope.
- [ ] Smoke run completes and all Test-plan checks pass before reporting done.
