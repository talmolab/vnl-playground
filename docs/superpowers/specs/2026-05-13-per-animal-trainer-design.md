# Per-animal trainer — purpose-built sAnimal entry point

**Status:** spec, 2026-05-13.
**Owner:** eric@talmolab.org.
**Predecessor:** `2026-05-02-sAnimal-per-animal-hyperparameter-sweep-design.md`.

---

## Motivation

The sAnimal sweep (5 animals × 9 hyperparameter cells = 45 specialist policies) launched on 2026-05-13 using the cohort trainer (`train_mouse_janelia_sigmoid_moving_shoulder.py`) with `--train-animals <X>` to restrict training kinematics to one animal. The eval pipeline still ran multi-animal EMG comparison (`--emg-animals A36-1 AT006 AT009 AT012 AT013`), producing three failure modes:

1. **`eval/reward_emg_summary` is artificially inflated.** The combined-summary plot tracks `eval/emg_cohort_<muscle>_trial_mae` — the mean across all 5 animals' trial MAE. For an AT013-specialist, that's 1 valid metric averaged with 4 garbage cross-animal pairings.
2. **`trial_mae` across mismatched animals is meaningless.** `compute_per_trial_metrics` pairs `sim_trial[i] ↔ bio_trial[i]` index-by-index. The single rollout (AT013 kinematics) is then paired against A36-1 trial 17, AT006 trial 17, etc. — different stimuli, undefined alignment, garbage numbers logged as `emg_<other>_<muscle>_trial_mae`.
3. **Cohort entry pollutes wandb panels.** Existing dashboards keyed on `emg_cohort_*` continue to populate but with cohort means dominated by 4-of-5 mismatched comparisons.

The fix is structural: separate per-animal training from cohort training at the entry-point level, so the per-animal eval pipeline never sees the cohort path.

## Goals

A. **One trainer per case.** Cohort training keeps `train_mouse_janelia_sigmoid_moving_shoulder.py` (unchanged). Per-animal training gets `train_mouse_janelia_per_animal.py`, purpose-built and single-animal-only at every layer of the eval pipeline.

B. **EMG eval matched to training scope.** For an AT013-specialist: train on AT013 kinematics, eval against AT013 EMG only. Trial-level pairing is within-AT013 (matched), so `trial_mae` is the real metric, not garbage.

C. **`reward_emg_summary` reflects home-animal fit.** The combined plot uses `eval/emg_<animal>_<muscle>_trial_mae` for the MAE history — the same animal the rollout was conditioned on.

D. **No contamination.** The cohort trainer is untouched. The cohort-trained networks already produced (s17/s18) remain interpretable under the cohort eval path.

## Architecture

```
train_mouse_janelia_per_animal.py                 # new, single-animal entry point
train_mouse_janelia_sigmoid_moving_shoulder.py    # unchanged, cohort entry point
sweep_sAnimal_{1..6}.sh                           # rewritten to call new trainer with --animal X
SANIMAL_LAUNCH.md                                 # updated launch commands
```

## CLI surface

`train_mouse_janelia_per_animal.py`:

- **Add**: `--animal <X>` (required; one of `A36-1 AT006 AT009 AT012 AT013`).
- **Remove**: `--train-animals` and `--emg-animals` are not parsed. Internally the trainer sets `args.train_animals = [args.animal]` and `args.emg_animals = [args.animal]` to avoid changing every downstream consumer.
- **Unchanged**: `--force-scale`, `--joint-damping`, `--shoulder-damping`, `--control-cost`, `--control-diff-cost`, `--seed`, `--num-timesteps`, `--num-evals`, `--emg-norm-method`, `--reference-data-path`, `--ctrl-dt`, `--sim-dt`, `--episode-length`, `--qvel-init`, `--joint-armature`, joint/wrist/bodies weight flags, `--wandb-group`, `--wandb-tags`, `--tag`, `--run-name`.

## EMG eval pipeline changes

Targeted changes to the eval block (currently lines 2280–2398 of the cohort trainer):

1. **Loading**: `emg_reference` is built via `load_emg_reference(..., animals=[args.animal])`. The synthetic `"cohort"` entry (produced by `load_emg_reference` for multi-animal configs) is bypassed or filtered out before the eval loop sees it.

2. **Loop**: `for animal, ref in emg_reference.items():` iterates exactly once over the home animal. The `non_cohort = [a for a in emg_reference if a != "cohort"]` filter collapses to `[args.animal]`. The cohort branch of `metrics_per_animal` is removed.

3. **Wandb keys logged**: `eval/emg_<animal>_<muscle>_<metric>` for the home animal (`mean_corr`, `trial_mae`, etc., logged as `corr` / `trial_mae`). No `emg_cohort_*` keys. No `emg_<other_animal>_*` keys.

4. **Combined summary plot**: `eval_history["triceps_mae"]`, `eval_history["biceps_mae"]`, `eval_history["AD_mae"]` are populated from `eval/emg_<animal>_<muscle>_trial_mae` (the home animal's per-trial MAE), not from the deleted `emg_cohort_*` keys. `eval/reward_emg_summary` then plots a meaningful trajectory.

## Sweep script rewrite

Each of `sweep_sAnimal_{1..6}.sh`:

- Drop `--emg-animals A36-1 AT006 AT009 AT012 AT013` from `BASE_ARGS`.
- Drop `--emg-animals X` from per-cell args (added earlier today as a stopgap before this design existed).
- Change per-cell `--train-animals X` to `--animal X`.
- Change Python entrypoint from `train_mouse_janelia_sigmoid_moving_shoulder.py` to `train_mouse_janelia_per_animal.py`.

`SANIMAL_LAUNCH.md`: bump pre-launch checklist to include "smoke test of new trainer passes". Launch commands themselves are unchanged (still `nohup bash sweep_sAnimal_N.sh`).

## Pipeline

1. Kill the 6 in-flight `sweep_sAnimal_*.sh` runs. Completed-cell checkpoints from those runs are discarded as part of this fix — kinematic learning is fine but the wandb metrics are unreliable, and the goal of the sAnimal sweep is the per-animal-EMG comparison that those runs cannot produce.
2. Build `train_mouse_janelia_per_animal.py`.
3. Smoke test: `python train_mouse_janelia_per_animal.py --animal A36-1 --num-timesteps 5_000_000 --num-evals 1 [...]`. Completes <10 min, wandb shows `eval/emg_A36-1_*` keys present, `eval/emg_cohort_*` keys absent, `eval/reward_emg_summary` panel populates without crash.
4. Rewrite the 6 sweep scripts. Lint: `for f in sweep_sAnimal_{1..6}.sh; do bash -n $f && echo OK; done` prints 6 OK lines.
5. Update `SANIMAL_LAUNCH.md`.
6. Relaunch the 6 sweep scripts. Expected wall: ~28 h, same as original plan.

## Test plan

Before relaunching the 45-cell sweep:

1. **Smoke test passes** as described in Pipeline step 3.
2. **Manual inspection of smoke wandb run**: `eval/emg_A36-1_biceps_trial_mae` is a finite number; `eval/emg_cohort_biceps_trial_mae` does not exist on the run; `eval/emg_AT006_biceps_trial_mae` does not exist either.
3. **Sweep lint**: All 6 sweep scripts pass `bash -n`, contain exactly the expected number of `--animal` occurrences (5 / 8 / 8 / 8 / 8 / 8 across scripts 1–6), and reference `train_mouse_janelia_per_animal.py` (not the cohort trainer).

## Falsifiable predictions

1. **Reward summary plot becomes informative for specialists.** After the rewrite, `eval/reward_emg_summary` for an AT013 cell shows a monotone-ish decrease in AT013-specific MAE as training progresses. Previously the cohort-mean MAE was 4-of-5 garbage and didn't move with training.
2. **Per-animal `trial_mae` is finite and ordered.** For each animal X, `eval/emg_X_<muscle>_trial_mae` for the X-trained cells is finite and lower (better) than the same metric for non-X-trained cells. Currently this is masked by the inflated cohort mean and cross-pairing nonsense.
3. **Smoke test passes in <10 min.** The new trainer is structurally identical to the cohort trainer; smoke time should match the cohort smoke time.

## Out of scope (deferred)

- **The cohort trainer's own eval-vs-cohort logic.** The cohort trainer uses `--emg-animals` to choose which animals to eval against, and its `--train-animals` may legitimately be all 5. For a cohort-trained model, cross-animal eval is meaningful (the policy was trained to fit all 5). No change needed there.
- **`compute_per_trial_metrics` cross-animal pairing.** The function correctly pairs sim trial *i* with bio trial *i* within a single (sim_actions, bio_traces) pair. The bug was at a higher level (which `bio_traces` it was paired with), not in the function itself.
- **Retroactive cleanup of cross-eval wandb runs.** The 6 sweep scripts launched earlier today (cohort trainer + my `--emg-animals X` per-cell stopgap) have already produced some logs. Those runs are orphaned by the relaunch; their wandb groups will simply stop accruing once killed.
- **UCM alignment, pre-reg YAML, Bayes factors** (from Approach B of the earlier brainstorm). Those happen after this fix, in a separate work block. Their inputs depend on a clean sAnimal cache, which requires this fix to land first.

## Decision gates

- [x] User confirmed: home-only eval, kill+relaunch acceptable, new trainer script preferred over patching cohort trainer.
- [x] User confirmed: scope is "build proposed design, ~1.5–2h."
- [ ] Smoke test passes before sweep rewrite.
- [ ] Sweep lint passes before relaunch.
