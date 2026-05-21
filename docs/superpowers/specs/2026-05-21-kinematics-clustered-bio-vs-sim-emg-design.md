# Kinematics-Clustered Bio vs Sim Muscle Activation Analysis

**Date:** 2026-05-21
**Author:** Eric (spec drafted with Claude)
**Audience:** Eric (handoff to Talmo + Austin while Eric is in Japan)
**Status:** Spec draft, awaiting review before implementation plan

## 1. Background and scientific claim

The MIMIC-MJX cohort policy `s18-ms-F4-fs1p2-20260502-014751` reproduces mouse-forelimb reach kinematics with near-perfect per-trial correlation (bio-sim r approx 0.99 on kin / xpos / qpos) but matches bio EMG poorly (per-trial r approx 0.05 across the cohort). This is the textbook Bernstein motor-redundancy problem: many muscle commands produce the same end-effector trajectory.

Claim to support: **within a single animal, reaches that are kinematically similar should share similar muscle activations.** Kinematic clustering supplies a controlled substrate so that any bio-vs-sim comparison is made across reaches already matched on kinematics, not across arbitrary reaches.

## 2. Scope

### In scope (Phase 1, this spec)

- Per-animal kinematic clustering on qpos (7 joint angles x 60 timesteps).
- Within-cluster within-modality self-similarity for bio EMG and sim muscle activation.
- Bio-to-sim linear mapping per (animal, cluster, muscle), plus per-animal pooled baseline.
- Paired distributional comparison for deterministic MLE rollouts.
- A reproducible HDF5 layout that bundles everything for Talmo and Austin.

### Deferred (later phases, listed here so the schema accommodates them)

- **Phase 2: stochastic rollouts and JSD branch.** Generate S=30/40/50 policy samples per trial, store in `stochastic_rollouts.h5`, compute per-timestep JSD against bio.
- **Phase 3: random-seed training sweep.** Train N=6 seeds at F4 settings (cc=0.025, cdc=0.025, fs=1.2) and N=6 seeds at C1 settings (cc=0, cdc=0, fs=1.0). Re-run the Phase 1 pipeline per seed, aggregate slope / cluster-tightness distributions across seeds.
- **Phase 2.5: extended rollout capture.** Re-roll the existing checkpoint with qvel, actuator_force, per-step reward captured. Not in any current cache.

### Explicitly excluded

- Triceps. The F4 checkpoint is known to under-drive triceps. All EMG / muscle-activation analysis uses M=2 channels: AD and Biceps. The full 12-channel actuator state is preserved for completeness but the comparison is two-muscle.
- Dataset generation. The notebook reads HDF5 only.

## 3. Style and code conventions

The deliverable code must satisfy:

- Standalone Jupyter cells (Colab-ready). No package imports of project-internal modules.
- `Args:` / `Returns:` docstrings on every function.
- `tqdm` on loops over animals / clusters / trials.
- No long-dash character in any string, docstring, or comment.
- Avoid the word "delve" (Eric's stylistic preference).
- Treat audience as new to OOP. Comment WHY, never WHAT.
- Each function has one responsibility. Refactor by isolating only the lines that change.

## 4. Data in hand

### Upstream caches (already on disk)

- `notebooks/kinematics_emg_comparison/cache/features.npz` (3 MB) - paired bio + sim per-trial features, N=204 trials, 5 animals, T=60 post-onset grid. Source of all bio + sim arrays except policy internals.
- `notebooks/talk_figures/figs/rollout_activations/s18-ms-F4-fs1p2-20260502-014751_278clips.npz` (~120 MB) - full 278-clip rollout cache. Source of policy internals (action_raw, intention, decoder activations) and all 12 actuator channels.

### Animal sample sizes

| Animal  | Trials |
|---------|--------|
| A36-1   | 46     |
| AT006   | 43     |
| AT009   | 41     |
| AT012   | 39     |
| AT013   | 35     |
| Total   | 204    |

## 5. HDF5 schemas (three files)

### 5.1 `paired_deterministic.h5` (source for the notebook)

Source of truth for bio + sim deterministic rollouts, paired trial-by-trial. Sleap `analysis.h5` style: flat top-level groups, fixed matrix shapes, semantics in attrs.

```
paired_deterministic.h5
│
├── /meta
│   ├── animal         (N=204,)      bytes    animal ID per trial
│   ├── trial          (N,)          int32    session-local trial index
│   ├── rollout_row    (N,)          int32    index into 278-clip rollout cache
│   └── @attrs
│       ├── checkpoint                "s18-ms-F4-fs1p2-20260502-014751"
│       ├── checkpoint_step           <int>
│       ├── trial_duration_s          0.25
│       ├── target_T                  60
│       ├── ctrl_dt_s                 <float>
│       ├── emg_muscle_names          ["AD", "Biceps"]
│       ├── actuator_names            ["AD", "Biceps", "Triceps", <9 others>]
│       ├── actuator_AD_biceps_idx    [<int>, <int>]
│       ├── qpos_joint_names          [<7 names>]
│       ├── xpos_body_names           [<6 names>]
│       ├── kin_marker_names          [<9 names>]
│       ├── bio_emg_norm              "p98, no ceiling clip"
│       ├── kin_detrended             true
│       ├── source_features_npz       <relative path>
│       ├── source_rollout_npz        <relative path>
│       └── created_utc               <ISO 8601>
│
├── /bio
│   ├── kin   (N, T=60, K=9, 3)    float32    measured marker positions
│   ├── xpos  (N, T=60, B=6, 3)    float32    body-part 3D positions (IK -> FK)
│   ├── qpos  (N, T=60, J=7)       float32    IK joint angles
│   └── emg   (N, T=60, M=2)       float32    AD + Biceps envelope, p98-norm
│
└── /sim
    ├── kin                 (N, T=60, K=9, 3)    float32    FK of rollout qpos
    ├── xpos                (N, T=60, B=6, 3)    float32    body-part positions
    ├── qpos                (N, T=60, J=7)       float32    rollout qpos
    ├── muscle_act          (N, T=60, A=12)      float32    mujoco data.act, all 12
    ├── muscle_act_AD_biceps (N, T=60, M=2)      float32    subset matched to bio.emg
    ├── action_raw          (N, T=60, A=12)      float32    pre-filter tanh action
    ├── intention           (N, T=60, L=4)       float32    encoder latent z (deterministic)
    ├── decoder_layer_0     (N, T=60, H=512)     float32    post-LN, first hidden
    ├── decoder_layer_1     (N, T=60, H=512)     float32    post-LN, second hidden
    └── decoder_layer_2     (N, T=60, H=512)     float32    post-LN, third hidden
```

Read patterns:
- Last-axis-is-channel everywhere. `f["/bio/emg"][i, :, 0]` is AD for trial i.
- Per-animal subset: `mask = f["/meta/animal"][:] == b"AT006"`, then index.
- Per-cluster subset: load `/clustering/<animal>/labels_kmeans` from `analysis_results.h5`, build mask, index `paired_deterministic.h5`.
- Whole-file size estimate: ~3 GB raw, dominated by decoder layers (204 x 60 x 512 x 3 layers x 4 bytes = ~75 MB total for decoders, so total ~80-100 MB). Manageable for Colab.

Conversion script `scripts/build_paired_deterministic_h5.py` (Phase 1 deliverable):
- Inputs: `features.npz`, `<network>_278clips.npz`, the trial-info CSV per animal.
- Output: `paired_deterministic.h5`.
- One-time, idempotent. Re-running on the same inputs overwrites the file.

### 5.2 `stochastic_rollouts.h5` (Phase 2, schema reserved)

Same `/meta` plus a sample axis. Only sim (bio is one recording per trial).

```
stochastic_rollouts.h5
│
├── /meta
│   ├── animal       (N,)
│   ├── trial        (N,)
│   ├── rollout_row  (N,)
│   └── @attrs
│       ├── checkpoint               "s18-ms-F4-..."
│       ├── n_samples_per_trial      30 (or 40 or 50, one file per S)
│       ├── policy_temperature       1.0
│       ├── seed_set_id              "set_a"
│       └── (same units / muscle attrs as 5.1)
│
└── /sim
    ├── qpos                 (N, S, T=60, J=7)       float32
    ├── xpos                 (N, S, T=60, B=6, 3)    float32
    ├── muscle_act           (N, S, T=60, A=12)      float32
    ├── muscle_act_AD_biceps (N, S, T=60, M=2)       float32
    ├── action_raw           (N, S, T=60, A=12)      float32
    └── intention            (N, S, T=60, L=4)       float32
```

Bio EMG to compare against comes from `paired_deterministic.h5` keyed by matching (animal, trial). One file per S in {30, 40, 50}.

### 5.3 `analysis_results.h5` (bundle handed to Talmo + Austin)

Everything the notebook produces, indexed back to (animal, trial) so it joins to 5.1.

```
analysis_results.h5
│
├── /meta
│   ├── animal   (N,)    bytes    same order as paired_deterministic.h5
│   ├── trial    (N,)    int32
│   └── @attrs
│       ├── source_paired               "paired_deterministic.h5"
│       ├── source_paired_sha256        <hash>
│       ├── notebook_commit             <git sha>
│       ├── created_utc                 <ISO 8601>
│       ├── kin_feature_for_clustering  "qpos_flat (T*J = 420)"
│       ├── distance_for_clustering     "euclidean on qpos_flat"
│       ├── self_similarity_for_emg     "mean per-muscle Pearson r (AD, Biceps averaged)"
│       └── chosen_k_per_animal         {"A36-1": 3, "AT006": 4, ...}
│
├── /clustering/<animal>
│   ├── labels_kmeans     (n_a,)        int32     cluster id per trial
│   ├── labels_hier       (n_a,)        int32
│   ├── centroids_kmeans  (k, T=60, J=7) float32  mean qpos trajectory per cluster
│   ├── wcss              (5,)          float32   within-cluster SS, k = 2..6
│   ├── silhouette        (5,)          float32   silhouette score, k = 2..6
│   ├── linkage_hier      (n_a - 1, 4)  float32   scipy.cluster.hierarchy linkage
│   └── @attrs
│       ├── k_grid          [2, 3, 4, 5, 6]
│       ├── chosen_k        <int>
│       └── selection_rule  "argmax(silhouette); fallback k=3"
│
├── /self_similarity/<animal>
│   ├── bio_emg_pairwise_mean_pearson        (n_a, n_a) float32  symmetric, NaN on diag
│   ├── sim_muscle_act_pairwise_mean_pearson (n_a, n_a) float32
│   ├── mean_bio_emg                (k, T=60, M=2) float32  per-cluster mean trace
│   ├── mean_sim_muscle_act         (k, T=60, M=2) float32
│   ├── within_bio_summary          (k, 3)         float32  cols (mean, std, n_pairs)
│   └── within_sim_summary          (k, 3)         float32
│
├── /glm_bio_to_sim/<animal>
│   ├── per_cluster
│   │   ├── slope     (k, M=2)  float32   one (slope, intercept) per (cluster, muscle)
│   │   ├── intercept (k, M=2)  float32
│   │   ├── r2        (k, M=2)  float32
│   │   ├── n_samples (k,)      int32     trials * timesteps used in fit
│   │   └── @attrs    direction: "sim_muscle_act = slope * bio_emg + intercept"
│   └── pooled
│       ├── slope     (M=2,)    float32   pooled across all clusters in animal
│       ├── intercept (M=2,)    float32
│       ├── r2        (M=2,)    float32
│       └── n_samples scalar
│
└── /jsd
    └── @attrs requires: "stochastic_rollouts.h5"  # populated in Phase 2
```

Why per-animal groups: clustering is per-animal and k differs across animals. A flat (N, ...) layout would need NaN-padding and a separate cluster-id column. The per-animal layout reads cleanly with `f[f"/clustering/{animal}"]` and gives one block per panel of every figure.

### 5.4 Bundle for handoff

A single tarball `vnl-playground-kin-emg-bundle-2026-05-21.tar.gz` containing:

```
bundle/
├── paired_deterministic.h5
├── analysis_results.h5
├── notebook.ipynb              # the Colab notebook, runs end to end on the H5s
├── build_paired_deterministic_h5.py  # converter, for reproducibility
├── figures/
│   ├── fig01_cluster_trajectories.pdf
│   ├── fig02_cluster_emg_means.pdf
│   ├── fig03_bio_sim_glm_traces.pdf
│   ├── fig04_self_similarity_distributions.pdf
│   └── fig05_summary.pdf
└── README.md                   # one page: how to open the H5s, key index = (animal, trial)
```

Talmo and Austin only need `paired_deterministic.h5`, `analysis_results.h5`, and `README.md` to reproduce every claim. The notebook and converter are for full reproducibility.

## 6. Phase 1 analysis pipeline

### Step 1: Per-animal kinematic clustering

Inputs: `/bio/qpos` from `paired_deterministic.h5`, sliced per animal.

For each animal:
1. Flatten qpos trajectories from (n_a, T, J) to (n_a, T*J=420).
2. Compute both clusterings over k in {2, 3, 4, 5, 6}:
   - KMeans: `sklearn.cluster.KMeans(n_clusters=k, n_init=10, random_state=0)`.
   - Hierarchical: `scipy.cluster.hierarchy.linkage` with Ward.
3. For each k record WCSS and silhouette.
4. Select k via `argmax(silhouette)`; fallback to k=3 if silhouette is below 0.1 or the max is within 0.02 of k=2 (degenerate). Record the rule applied per animal.
5. Persist labels for both methods, centroids for KMeans, linkage matrix for hierarchical.

Diagnostics:
- Silhouette and WCSS curves per animal (Fig 1A).
- Member trajectories overlaid on cluster centroid for the chosen k (Fig 1B). Confirms the cluster is genuinely tight, not a label collapse.

Output goes to `/clustering/<animal>` in `analysis_results.h5`.

### Step 2: Within-cluster within-modality self-similarity

Inputs: bio EMG and sim muscle activation (AD + Biceps subset only) sliced per (animal, cluster).

For each (animal, cluster):
1. Per modality, compute pairwise self-similarity across trials in the cluster as **mean per-muscle Pearson r**: for each pair of trials (i, j), compute Pearson r on the AD channel and on the Biceps channel separately over T=60 timesteps, then average the two r values. Matches Eric's standing metric preference (see memory: `feedback_emg_similarity_metric`). Output a symmetric (n_cluster, n_cluster) matrix per (animal, cluster), NaN on diagonal.
2. Per modality, compute mean envelope across cluster trials (T, M).
3. Summary per cluster: mean of upper triangle, std, n_pairs.

Prior: sim is more self-similar than bio because the policy is deterministic and the latent is a low-D bottleneck. Report both distributions side by side.

Output: `/self_similarity/<animal>` arrays.

### Step 3: Cross-modality GLM (bio EMG -> sim muscle_act) within matched clusters

Per (animal, cluster, muscle), fit:

```
sim_muscle_act[t, m]  =  slope[m] * bio_emg[t, m]  +  intercept[m]
```

pooled across all trials and timesteps in that cluster, one fit per (cluster, muscle). With M=2 muscles this gives 2 affine maps per cluster.

Then per animal, pooled across all clusters: 2 affine maps as a less-constrained baseline.

Use `sklearn.linear_model.LinearRegression` or a closed-form fit. Report R^2 on the same data (in-sample) plus a 5-fold CV R^2 per (animal, cluster) to flag overfitting.

Plot per cluster: bio trace, raw sim trace, regression-corrected sim trace = `slope * bio + intercept` evaluated on cluster mean bio.

Output: `/glm_bio_to_sim/<animal>` per_cluster and pooled subgroups.

### Step 4: Distributional comparison (paired branch only)

Deterministic MLE rollouts are paired time series. For each (animal, cluster):
- Per-timestep paired difference distribution: `delta(t, m) = sim_muscle_act[trial, t, m] - bio_emg[trial, t, m]` across cluster trials.
- Plot `mean(delta)` and `+/- std(delta)` over t, per muscle, per cluster.

JSD branch documented but not implemented in Phase 1 (requires `stochastic_rollouts.h5`).

## 7. Notebook structure (Colab-ready)

`notebook.ipynb` is a sequence of standalone cells. Each cell has a markdown header and a function definition + a call. No cross-cell mutable state beyond the loaded HDF5 file handles and a single `RESULTS` dict.

Cells, in order:

1. **Setup.** Install minimum deps (`!pip install h5py scikit-learn scipy tqdm`), import, set seeds, open the two HDF5 files.
2. **Load helpers.** `load_animal(f_paired, animal) -> dict` returns a per-animal slice with bio + sim arrays.
3. **Step 1 cell A: clustering.** `cluster_animal(animal_data) -> ClusterResult` with KMeans + hierarchical + k selection.
4. **Step 1 cell B: trajectory plot.** `plot_cluster_trajectories(animal, result) -> Figure`.
5. **Step 2: within-cluster self-similarity.** `within_cluster_similarity(animal_data, labels) -> SelfSimResult`.
6. **Step 3 cell A: GLM fit.** `fit_glm_per_cluster(animal_data, labels) -> GLMResult`.
7. **Step 3 cell B: trace plot.** `plot_bio_sim_glm_traces(animal, glm_result) -> Figure`.
8. **Step 4: paired delta over time.** `plot_paired_delta(animal_data, labels) -> Figure`.
9. **Persistence.** `write_analysis_results(RESULTS, "analysis_results.h5")`.

Eric's per-function commit pattern: when iterating, the cell author edits one function's body; the surrounding cell structure does not change.

## 8. Implementation order

1. **Add deps.** `uv add scipy scikit-learn tqdm` from the project root. Pin nothing for now.
2. **Converter.** `scripts/build_paired_deterministic_h5.py`. Reads `features.npz` and `<network>_278clips.npz`, writes `paired_deterministic.h5` per the schema in 5.1. Verifies row alignment via `rollout_row`.
3. **Notebook scaffold.** `notebook.ipynb` with all 9 cells outlined, each with the function signature and a placeholder `raise NotImplementedError`.
4. **Step 1.** Clustering + diagnostics. Cell-by-cell.
5. **Step 2.** Self-similarity. Cell-by-cell.
6. **Step 3.** GLM. Cell-by-cell.
7. **Step 4.** Paired delta. Cell-by-cell.
8. **Persistence.** `write_analysis_results` + `analysis_results.h5` schema check.
9. **Bundle.** A `scripts/make_handoff_bundle.sh` that tars 5.4.

Each step is committable in isolation. The implementation plan generated from this spec will list the exact function signatures and tests per step.

## 9. Out-of-scope reminders (for the eventual implementer)

- No re-rollouts. The conversion script does not depend on JAX or MuJoCo. Eric's uv venv has h5py + numpy + scipy + sklearn after step 1; that's enough.
- No new network. Phase 3 trains seeds; Phase 1 reuses `s18-ms-F4-fs1p2-20260502-014751` only.
- No package-internal imports in the notebook. Colab portability is mandatory.

## 10. Open questions to confirm before writing the implementation plan

1. Schema shapes and attrs in 5.1, 5.2, 5.3 - any additions?
2. `analysis_results.h5` per-animal groups vs flat `(N, ...)` arrays - I picked per-animal because k differs; confirm.
3. Bundle format - tarball with HDF5s + notebook + figures + README is the default in 5.4. If you'd rather a Colab notebook that downloads from a hosted URL, name the URL pattern.
4. k selection rule (argmax silhouette, fallback k=3 if silhouette < 0.1 or near-tie with k=2) - acceptable?
5. (Removed - DTW dropped in favor of mean per-muscle Pearson r per Eric's stated preference.)
