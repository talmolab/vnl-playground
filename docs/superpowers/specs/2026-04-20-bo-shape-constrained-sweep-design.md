# BO-driven shape-constrained parameter search

**Date:** 2026-04-20
**Status:** design (approved)
**Predecessors:** s10-ms, s11-ms, s12-ms (currently running)

## Goal

Replace offline surrogate analysis + hand-picked grids with a closed-loop Bayesian-ish optimization driver that:

1. Warm-starts from the 237-run s10+s11 pool.
2. Launches training runs one at a time on the free single-GPU node.
3. Updates an NSGA-II surrogate after each run and proposes the next config.
4. After 20 trials (~20h wallclock), reports a Pareto frontier of configs that satisfy the reward floor.

This is a tactical experiment, not permanent infrastructure. Goal is to see whether adaptive search finds a config that crosses all four of s12's targets where grid search has not:

- `eval/episode_reward ≥ 400`
- `eval/emg_biceps_corr ≥ 0.70`
- `eval/emg_triceps_corr ≥ 0.60`
- `eval/emg_biceps_mae ≤ 0.15`
- `eval/emg_triceps_mae ≤ 0.15`

## Optimizer choice

**Optuna with `NSGAIISampler`**, `JournalFileStorage` backend. Chosen over BoTorch / GPJax / sklearn-GP because:

- Sample efficiency gap between NSGA-II and GP-based MOBO is negligible on a 5-D problem with ~200 warm-start points + 20 new evals — the bottleneck is 1h/run, not acquisition quality.
- No new heavy deps (no torch, no JAX-side GP library). Optuna adds one pip install.
- `JournalFileStorage` is single-process-native, append-only, plain-file, resumable — no SQL machinery.

## Problem formulation

**Search space** (5 axes):

| axis | type | range |
|---|---|---|
| `force_scale` | continuous | [0.5, 1.0] |
| `joint_damping` | continuous, log-scale | [1e-7, 1.5e-6] |
| `control_cost` | continuous | [0.0, 0.1] |
| `control_diff_cost` | continuous | [0.0, 0.1] |
| `qvel_init` | categorical | {`zeros`, `reference`} |

Reopening `force_scale` down to 0.5 deliberately — s10's shape-king run (bcorr=0.90 at fs=0.7) lives in that region and we want the surrogate to keep it on the table.

**Objectives** (4, all on `eval/` keys from wandb):

- maximize `emg_biceps_corr`
- maximize `emg_triceps_corr`
- minimize `emg_biceps_mae`
- minimize `emg_triceps_mae`

**Constraint:**

- `eval/episode_reward ≥ 380` during BO (Optuna convention: `constraint_value = 380 - R`, negative = feasible).
- Post-hoc winner filter raises the bar to `R ≥ 400` — final reporting only considers feasible-at-400 trials.

Rationale for 380 during BO: ~40 warm-start runs clear 380 but only ~15 clear 400. The surrogate needs enough feasible seeds to learn the shape of the feasible region.

**Fixed across every trial:**

| param | value |
|---|---|
| `seed` | 1 (match s11-ms main-factorial convention) |
| `iterations` | 6 (s12 default; ~1h/run) |
| `episode_length` | 100 |
| `joint_armature` | 4e-10 |
| `ctrl_dt` | 0.0025 |
| `sim_dt` | 0.00125 |
| training script | `train_mouse_janelia_sigmoid_moving_shoulder.py` |
| env | moving-shoulder |
| muscle tau | default (no per-muscle overrides) |

## Warm-start

Source: `/root/vast/eric/vnl-playground/s11_ms_s10_ms_final.csv`.

**Filter:**
- moving-shoulder tag present
- no `tau-extra` tag (8 runs dropped — they use non-default muscle dynamics)
- no NaN in any of `eval/episode_reward`, `eval/emg_biceps_corr`, `eval/emg_triceps_corr`, `eval/emg_biceps_mae`, `eval/emg_triceps_mae`
- `state == "finished"`

Expected retained count: ~200 trials.

**Not filtered:** axis values outside the current search-space bounds are still useful to the surrogate. Optuna won't propose outside bounds regardless of warm-start content.

**s10 runs:** kept unless we discover an env/reward breaking change between s10 and s11. No such change is known at time of writing.

**Loader:** maps CSV columns to Optuna `FrozenTrial` objects with `state=COMPLETE` and feeds them via `study.add_trials()` before the first `ask()`.

## Architecture

Single-process Python driver. One long-running loop, serial trials, one GPU.

```
[ warm-start: CSV -> FrozenTrials -> study.add_trials() ]
                            |
                            v
  +-----------------------------------------------+
  | Optuna study (NSGAIISampler, 4 objectives,    |
  | R>=380 constraint, JournalFileStorage)        |
  +-----------------------------------------------+
                            |
                            v
                       study.ask()
                            |
                            v
      subprocess: train_mouse_janelia_sigmoid_moving_shoulder.py
                 --force-scale --joint-damping --control-cost
                 --control-diff-cost --qvel-init --seed 1
                 --wandb-tags bo-s13,trial-NNNN
                            |
                            v
                  [ training writes to wandb ]
                            |
                            v
                   subprocess exits
                            |
                            v
         wandb API: fetch run.summary by tag (3x retry, 30s backoff)
                            |
                            v
         study.tell(trial, [bcorr, tcorr, bmae, tmae],
                    constraint=[380 - R])
                            |
                            v
              append to bo_trials.jsonl
                            |
                            v
                [ loop, 20 iterations ]
                            |
                            v
   Pareto frontier -> filter R>=400 -> bo_frontier.csv + print winners
```

## Components

All in `scripts/bo_optimize.py`. Five functions, one entry point.

### `load_warmstart(csv_path: Path) -> list[FrozenTrial]`

Read CSV with pandas, apply filter rules (above), map columns to Optuna `FrozenTrial` with `COMPLETE` state, 4-objective values, and constraint value `380 - R`. Return list.

Unit-testable by passing a small fixture CSV.

### `make_study(study_name: str, journal_path: Path) -> optuna.Study`

```python
optuna.create_study(
    study_name=study_name,
    storage=JournalStorage(JournalFileBackend(str(journal_path))),
    directions=["maximize", "maximize", "minimize", "minimize"],
    sampler=NSGAIISampler(constraints_func=_constraints),
    load_if_exists=True,
)
```

`_constraints(trial)` pulls `trial.user_attrs["constraint"]` which `tell()` sets.

### `launch_training(params: dict, tag: str) -> int`

Blocking `subprocess.run()` on the training script. Returns exit code. All output tee'd to `bo_runs/<tag>.log` for post-hoc debugging.

Parameters mapped to CLI flags:
- `force_scale` -> `--force-scale`
- `joint_damping` -> `--joint-damping`
- `control_cost` -> `--control-cost`
- `control_diff_cost` -> `--control-diff-cost`
- `qvel_init` -> `--qvel-init`
- fixed: `--seed 1`, `--iterations 6`, `--wandb-tags bo-s13,<tag>`

### `read_metrics(tag: str) -> dict | None`

Query wandb API:
```python
api.runs(path="<entity>/<project>", filters={"tags": tag})
```
Pick the single matching run. Pull `eval/episode_reward`, `eval/emg_biceps_corr`, `eval/emg_triceps_corr`, `eval/emg_biceps_mae`, `eval/emg_triceps_mae` from `run.summary`.

Retry up to 3 times with 30s backoff — wandb upload can lag briefly after process exit. Return `None` on final failure or NaN values.

### `report_frontier(study: optuna.Study, out_csv: Path) -> None`

Compute Pareto front from `study.best_trials` (Optuna excludes infeasible automatically when `constraints_func` is set). Additionally filter to `R ≥ 400` using stored user_attrs. Print top 5 by each objective. Write full frontier to `bo_frontier.csv`.

### `main()`

```
parse args (--n-trials 20, --warmstart-csv, --study-name, ...)
load_warmstart -> study.add_trials
for i in range(n_trials):
    trial = study.ask()
    params = {
        "force_scale": trial.suggest_float("fs", 0.5, 1.0),
        "joint_damping": trial.suggest_float("damp", 1e-7, 1.5e-6, log=True),
        "control_cost": trial.suggest_float("cc", 0.0, 0.1),
        "control_diff_cost": trial.suggest_float("cdc", 0.0, 0.1),
        "qvel_init": trial.suggest_categorical("qvel_init", ["zeros", "reference"]),
    }
    tag = f"trial-{trial.number:04d}"
    exit_code = launch_training(params, tag)
    metrics = read_metrics(tag) if exit_code == 0 else None
    if metrics is None or any(isnan(v) for v in metrics.values()):
        study.tell(trial, state=FAIL)
        consecutive_fail += 1
        if consecutive_fail >= 5: abort_with_loud_log()
    else:
        trial.set_user_attr("constraint", [380 - metrics["R"]])
        trial.set_user_attr("R", metrics["R"])
        study.tell(trial, [metrics["bcorr"], metrics["tcorr"],
                           metrics["bmae"], metrics["tmae"]])
        append_jsonl(bo_trials_jsonl, params | metrics | {"trial": trial.number})
        consecutive_fail = 0
report_frontier(study, "bo_frontier.csv")
```

## Data flow per trial (summary)

1. `study.ask()` -> params dict
2. build CLI args + tag `trial-NNNN`
3. blocking subprocess on training script (~1h)
4. wandb API fetch by tag (3x retry, 30s backoff)
5. `study.tell(objectives, constraint=[380 - R])` or mark FAIL
6. append row to `bo_trials.jsonl`
7. journal file already updated by Optuna on `tell()`

## Error handling

| failure | handling |
|---|---|
| training subprocess non-zero exit | log, skip wandb, `study.tell(state=FAIL)`, increment `consecutive_fail` |
| wandb run missing / no metrics after 3 retries | same as above |
| NaN metrics | same as above |
| `consecutive_fail >= 5` | abort loop, log loudly (stderr + `bo_trials.jsonl` marker). Do not burn remaining budget on a broken loop |
| driver crash | journal + JSONL both persist. Restart with same `--study-name` resumes from last completed trial |
| ctrl-C | Optuna flushes journal on signal; next run with same study name resumes |

## Testing — before committing the 20h budget

Three cheap checks (~15 min total):

1. **Warm-start dry-run:** run `load_warmstart()` only. Assert ~200 trials retained, axis ranges sane, no NaN, feasibility count (R >= 380) reasonable.
2. **Single-trial end-to-end:** `--n-trials 1 --iterations 1` (~10 min total). Verifies CLI args flow correctly, wandb tag appears, `read_metrics` pulls all five fields, `tell()` accepts, both log files written.
3. **Resume check:** kill and restart the 1-trial test with same `--study-name`. Verifies journal-based resume works and NSGA-II sees the completed trial.

## Outputs

- `bo_study.log` — Optuna journal (authoritative trial store)
- `bo_trials.jsonl` — flat redundant log, one row per completed trial
- `bo_runs/<tag>.log` — per-trial training stdout/stderr
- `bo_frontier.csv` — final Pareto frontier filtered to R >= 400
- wandb tag `bo-s13` — all trials queryable as a set

## Success criteria

At end of 20 trials:

- **Primary:** Pareto frontier contains at least one trial with `R >= 400 AND bcorr >= 0.70 AND tcorr >= 0.60 AND bmae <= 0.15 AND tmae <= 0.15`. This is the "BO found a config s11/s12 grids missed" result.
- **Partial:** frontier contains a trial with 3 of 4 shape thresholds cleared at R >= 400 — suggests the 4th is in tension and needs a different axis (seeds, tau, reward shaping).
- **Null:** no feasible-at-400 trial clears more shape thresholds than the best s11 frontier cell — BO and grid agree there's a ceiling; pivot to direct EMG-trace supervision (deferred to future work).

## Out of scope

- Multi-GPU batched BO (this node has 1 GPU; batched BO is a separate spec if/when multiple nodes free up).
- Seed variance estimation (fixed at seed=1; seed sweep would be a replication pass over BO winners, not part of the search loop).
- Propriospinal ratios and per-muscle tau (no CLI flags; wiring deferred).
- Multi-fidelity (Hyperband) — considered, rejected due to multi-objective pruning ambiguity.
- Hot-swapping the search space mid-run — if bounds need to change, start a new study.

## Risks

- **Warm-start quality:** if s10 is silently non-comparable to s11/bo-s13 due to env drift, ~40 warm-start points mislead the surrogate. Mitigated by filter being loose — NSGA-II doesn't extrapolate aggressively from warm-starts.
- **NSGA-II with 4 objectives is information-hungry:** 20 new trials is modest. Expect the frontier to be dominated by warm-start points; the value of the 20 trials is filling specific gaps the surrogate identifies, not replacing the frontier.
- **Constraint-handling in NSGA-II:** infeasible trials still run (NSGA-II doesn't predict feasibility cheaply). If most of the feasible region is narrow, the first several trials may be infeasible — that's expected, not a bug.
- **Wandb flakiness:** 3-retry with backoff covers normal upload lag but not extended wandb outages. Extended outage -> trials FAIL and budget is consumed. Manual intervention required.
