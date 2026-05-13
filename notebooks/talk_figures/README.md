# talk_figures — handoff & resume notes

A reference for picking up the talk-figures pipeline without re-deriving paths, shapes, and conventions.

---

## Pick up here (paste into a fresh session)

> `talk_figures/` is built around the `s18-ms-F4-fs1p2-20260502-014751` physics-aware checkpoint and there's already a per-(network, animal) envelope cache at `vnl_playground/bayesian_emg/cache/v1/envelopes/<network>/<animal>.npz` storing `sim` and `empirical` shape `(n_trials, 60, 3)` for `(AD, Triceps, Biceps)`. Rollouts already exist at `notebooks/talk_figures/figs/rollout_activations/<network>_278clips.npz` (qpos + intention + decoder activations). No need to re-roll. To pick up: read `notebooks/talk_figures/README.md` for full paths and the figure inventory.

---

## What's here

| File | Produces |
|---|---|
| `talk_emg_figures.py` | `fig1_single_trial_<animal>`, `fig2_mean_trace`, `fig3_mae_box` — EMG vs simulated muscle activation, Physics-Aware vs Joint Reward Only. |
| `pca_figures.py` | `fig6/12/13` — PCA of intention latent + 3 decoder hidden layers, color-coded by shoulder extension. Also exposes the rollout/FK helpers used by the other scripts. |
| `tracking_kinematics_figures.py` | `fig8_tracking_error_boxplot`, `fig9_registration_error_boxplot`, `fig10_tracking_vs_registration_panel`, `fig11_kinematics_clip<NN>_<net>` — FK(rollout qpos) vs mocap, STAC marker registration error, 7-DOF qpos overlays. |
| `similarity_pairwise.py` *(in progress)* | `fig_sim_heatmap_{bio,sim}_{kin,emg}`, `fig_sim_block_summary`, `fig_fpca_modes`, `fig_fpca_bio_basis_scatter` — pairwise cosine heatmaps (bio/sim × kin/EMG), within/between-animal block stats, fPCA basis comparison between bio EMG and sim muscle activations. |
| `extract_pre_onset_emg.py` | `figs/pre_onset_cache/<animal>.npz` — extends EMG envelopes back to `-50 ms` (72-step `(−50..+250 ms)` cache). |
| `build_notebook.py` | Assembles `talk_emg_figures.ipynb` from `talk_emg_figures.py`. |

Spec for `similarity_pairwise.py`: `docs/superpowers/specs/2026-05-13-pairwise-similarity-fpca-design.md`.

## Checkpoints

| Alias | wandb run name | Cell | Notes |
|---|---|---|---|
| Physics-Aware (primary) | `s18-ms-F4-fs1p2-20260502-014751` | F4: `fs=1.2`, `cc=0.025`, `cdc=0.025` | Clears the >400 kinematic-fit bar (`eval/episode_reward_single = 436.72`). |
| Joint Reward Only (contrast) | `s18-ms-C1-cc0-cdc0-20260502-051429` | C1: `cc=0`, `cdc=0` | Also clears the bar (`426.99`). Same animals, same kin reward, no control penalty. |

(`talk_emg_figures.py` exposes these as `PHYSICS_RUN` and `JOINTS_RUN`; `pca_figures.py` exposes `AP_F4_RUN`, `AP_F3_RUN`, `NOAP_RUN`.)

## Caches

| Cache | Path pattern | Per-trial shape | Notes |
|---|---|---|---|
| EMG envelopes (sim + empirical) | `vnl_playground/bayesian_emg/cache/v1/envelopes/<network>/<animal>.npz` | `sim, empirical: (n_trials, 60, 3)`; `muscles: (3,) ["AD","Triceps","Biceps"]`; `trial_idx: (n_trials,)` | 60 steps × 4.17 ms = 250 ms post-onset. |
| Rollout activations (qpos + intention + decoder) | `notebooks/talk_figures/figs/rollout_activations/<network>[_278clips].npz` | `qposes_rollout: (278, 100, 7)`; `ctrl: (278, 100, 12)`; `intention: (278, 100, 4)`; `decoder_layer_{0,1,2}: (278, 100, 512)` | 100 steps × 4.17 ms; first 60 steps align with the post-onset EMG window. |
| Pre-onset EMG | `notebooks/talk_figures/figs/pre_onset_cache/<animal>.npz` | `(n_trials, 72, 3)` over `−50..+250 ms` | Built by `extract_pre_onset_emg.py`. |
| Bio kinematics | `vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/<clip>_ik.h5` | `kp_data: (50, 9)`, `marker_sites: (50, 3, 3)`, `qpos: (50, 7)`, `qvel: (50, 7)`, `xpos: (50, 8, 3)`, `xquat: (50, 8, 4)` | 50 steps × 5 ms = 250 ms (mocap rate). `kp_names = ["Shoulder","Elbow","Wrist"]`. |

## Cohort

Animal order (use this for all stable indexing):

```python
ANIMALS = ("A36-1", "AT006", "AT009", "AT012", "AT013")
```

Per-animal clip counts:

| Animal | Clips |
|---|---:|
| A36-1 | 46 |
| AT006 | 60 |
| AT009 | 45 |
| AT012 | 83 |
| AT013 | 44 |
| **Total** | **278** |

## Coordinate / FK conventions

- For FK on rollout qpos use the **STAC XML**:
  ```
  /root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml
  ```
  This has the `shoulder_base` body needed to match bio `kp_data` body frames. The vnl IK XML lacks it — do **not** use the vnl XML for FK.
- KP ↔ body map (matches `tracking_kinematics_figures.py`):
  ```python
  KP_NAMES = ["Shoulder", "Elbow", "Wrist"]
  KP_BODY  = {"Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist"}
  ```
- Joint order (`names_qpos`, 7 DOF):
  ```
  sh_tx, sh_ty, sh_tz, sh_rotation, sh_extension, sh_elevation, elbow
  ```

## Time alignment

| Source | Steps | dt (ms) | Window (ms) |
|---|---:|---:|---|
| EMG cache (`sim`, `empirical`) | 60 | 4.17 | `0 .. +250` (post-onset) |
| Rollout qpos | 100 | 4.17 | `0 .. +417` — slice `[:60]` for post-onset alignment |
| Bio kin (`kp_data`, `qpos`, `marker_sites`) | 50 | 5.00 | `0 .. +250` — resample 50→60 (linear) for cross-modality work |
| Pre-onset cache | 72 | 4.17 | `−50 .. +250` |

## Environment

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu   # all talk-figures scripts run on CPU
```

`mujoco_playground` is imported from outside the repo:
```python
_MJX_PLAYGROUND_SRC = Path("/root/vast/scott-yang/mujoco_playground")
sys.path.insert(0, str(_MJX_PLAYGROUND_SRC))
```

## Style

Matplotlib defaults shared across scripts:

```python
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 8
```

Palette:

| Role | Color |
|---|---|
| Empirical EMG | `#8E44AD` (purple) |
| Physics-Aware | `#1f77b4` (blue) |
| Joint Reward Only | `#ef7307` (orange) |

## Figure inventory

| Fig # | Script | Description |
|---|---|---|
| 1 | `talk_emg_figures.py` | Single-trial EMG vs both models, per animal. |
| 2 | `talk_emg_figures.py` | Mean trace ± SEM across 5 animals × clips. |
| 3 | `talk_emg_figures.py` | MAE box plot, Physics-Aware vs Joint Reward Only. |
| 6 | `pca_figures.py` | PCA of intention + decoder activations, Physics-Aware (278 clips). `fig6_pca_physics_aware*`. |
| 6b | `pca_figures.py` | Same, F3 anchor. |
| 7 | `pca_figures.py` | PCA, joints-only (no action penalty). |
| 8 | `tracking_kinematics_figures.py` | Tracking error box plot. |
| 9 | `tracking_kinematics_figures.py` | Registration error box plot. |
| 10 | `tracking_kinematics_figures.py` | Tracking vs registration panel. |
| 11 | `tracking_kinematics_figures.py` | Per-clip 7-DOF qpos overlays. |
| 12, 12b, 13 | `pca_figures.py` | PCA on 50-clip subsets (10 per animal), animal-color variant. |
| `fig_sim_*`, `fig_fpca_*` | `similarity_pairwise.py` | Pairwise cosine heatmaps + fPCA comparison. |

## Conventions for new scripts

If you add a new figure-producing script under `notebooks/talk_figures/`:

1. Use `_MJX_PLAYGROUND_SRC` insertion at the top (copy from `pca_figures.py`).
2. Set the env vars listed above before any `mujoco` / `brax` import.
3. Save figures as both `.pdf` and `.png` into `figs/`.
4. Use the shared palette and matplotlib rcParams.
5. Use the `ANIMALS` tuple in the order listed above for stable indexing.
6. For FK on rollout qpos, use the STAC XML and `KP_BODY` map.
7. Add the script + figures to the table in this README.
