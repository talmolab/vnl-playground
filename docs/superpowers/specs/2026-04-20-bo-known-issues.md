# BO driver — known issues discovered in final review

## 1. SEARCH_SPACE_DISTRIBUTIONS bounds wider than spec (scripts/bo_optimize.py:44-50)

Plan spec: `fs=[0.5,1.0], damp=log[1e-7,1.5e-6], cc/cdc=[0,0.1]`
Actual:    `fs=[0.1,1.5], damp=log[1e-8,1e-6], cc/cdc=[0,0.2]`

The `main()` loop clamps proposed params back to spec bounds before launching
training, so correctness (no out-of-spec training) is preserved. But NSGA-II's
internal surrogate is updated with the wider-bounds proposal it generated, not
the clamped value actually evaluated — so the first several trials may cluster
at the clamped boundary and give the surrogate muddled signal.

**Fix next run:** narrow distributions to spec bounds `[0.5,1.0]/[1e-7,1.5e-6]/[0,0.1]`.
This invalidates existing journal files (cannot resume old studies).

## 2. report_frontier dumps all R>=400 trials, not Optuna's best_trials

Plan: "Compute Pareto front from `study.best_trials`".
Actual: filters all COMPLETE trials with R>=400.

Likely more useful in practice (shows the full feasible set), but diverges from
spec wording. No impact on running production.

## 3. Fixture CSV column names diverge from plan

Plan's Task 1 sample CSV uses `_fields.control_cost`. Real CSV (and fixture)
use `reward_weights/control_cost`. Intentional fix in commit cd7e8b1.
Plan document remains stale — regenerate from code, not plan.
