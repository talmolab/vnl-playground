#!/bin/bash
# SWEEP S16-MS part 3/6 — GPU 3 (2-GPU machine #2, GPU 0)
# Priority-ordered: T4b aggressive per-muscle + C04 stacked biceps fix + X8 sat-cost 0.1
# 17 cells, single-seed, budget-aware (12h default).
# Spec: docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s16-ms-part3"
BUDGET_SECONDS=$(( ${BUDGET_HOURS:-12} * 3600 ))
ESTIMATED_RUN_SECONDS=${ESTIMATED_RUN_SECONDS:-12600}

BASE_ARGS=(
    --ctrl-dt 0.0025
    --sim-dt 0.00125
    --episode-length 100
    --qvel-init zeros
    --joint-armature 4e-10
    --joints-weight 5.0
    --joints-vel-weight 0.5
    --wrist-pos-weight 0.1
    --bodies-pos-weight 0.1
    --num-timesteps 800000000
    --num-evals 8
    --emg-norm-percentile 98
    --wandb-group "${WANDB_GROUP}"
)

TAU_SYM25="--muscle-tau-act 0.025 --muscle-tau-deact 0.040"
TAU_MILD="--muscle-tau-act 0.020 --biceps-tau-act 0.030 --brachialis-tau-act 0.025 --muscle-tau-deact 0.040"
TAU_AGGR="--muscle-tau-act 0.020 --biceps-tau-act 0.045 --brachialis-tau-act 0.030 --muscle-tau-deact 0.040"

START_TIME=$(date +%s)
CRASHED=()
OK=()
SKIPPED=()
TOTAL=17
CELL=0

run_cell() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS )); then
        echo "----------------------------------------------------------------"
        echo "[S16-MS-3 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s16-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S16-MS-3 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
    echo "  $@"
    echo "----------------------------------------------------------------"
    if python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" "$@" 2>&1 | tee "${LOG}"; then
        OK+=("${RUN_NAME}"); echo "[OK] ${RUN_NAME}"
    else
        CRASHED+=("${RUN_NAME}"); echo "[CRASHED] ${RUN_NAME} (see ${LOG})"
    fi
    echo
}

# ===== Priority-1 (must-run) =====

# Cell 1 — T4b: b=45 br=30 tl=15 tla=15 (aggressive biceps, fast triceps)
run_cell "T4b-b45tl15" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.015 --biceps-tau-act 0.045 --brachialis-tau-act 0.030 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T b45tl15 priority1

# Cell 2 — C04: S3 + biceps-force 0.07 + biceps-tau-act 0.055 (stacked biceps fix)
run_cell "C04-bforce07-btau55" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --biceps-force 0.07 --biceps-tau-act 0.055 \
    --wandb-tags s16-ms C C04 stacked-biceps asym-mild priority1

# Cell 3 — X8: saturation-cost 0.1 with margin 0.8 (strong sat penalty)
run_cell "X8-satcost01" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --saturation-cost 0.1 --saturation-margin 0.8 \
    --wandb-tags s16-ms X X8 satcost01 asym-mild priority1

# ===== Tier 2 =====

# Cell 4 — T1c: tau_act=20 global
run_cell "T1c-tau20" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.020 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T tau20

# Cell 5 — T3a: biceps-only tau_act=25 (mild biceps slowdown)
run_cell "T3a-biceps25" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.015 --biceps-tau-act 0.025 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T biceps25

# Cell 6 — B07-aggr: high fs + weak shoulder + asym-aggr
run_cell "B07-highFsWeakSh-aggr" \
    --force-scale 1.3 --joint-damping 9e-7 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B07 asym-aggr

# Cell 7 — B09-mild: fs=1.2 asym damp smoothOnly + asym-mild
run_cell "B09-fs12asymSmoothOnly-mild" \
    --force-scale 1.2 --joint-damping 1.2e-6 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B09 asym-mild

# Cell 8 — C09: S3 asym-aggr + biceps-force 0.07 (stack aggressive tau + force)
run_cell "C09-aggr-bforce07" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} --biceps-force 0.07 \
    --wandb-tags s16-ms C C09 aggr-bforce asym-aggr

# Cell 9 — C11: R2 asym-aggr + saturation-cost 0.02
run_cell "C11-R2aggr-sat002" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_AGGR} --saturation-cost 0.02 \
    --wandb-tags s16-ms C C11 R2-aggr-sat asym-aggr

# Cell 10 — B10-aggr: strong sym damp + asym-aggr
run_cell "B10-strongsym-aggr" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 1.5e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B10 asym-aggr

# Cell 11 — B13-sym25: mid-stiff weak shoulder smoothOnly + sym25
run_cell "B13-midStiffWeakSh-sym25" \
    --force-scale 1.1 --joint-damping 1.2e-6 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B13 sym25

# Cell 12 — B06-mild: s11 goldilocks + asym-mild
run_cell "B06-s11goldi-mild" \
    --force-scale 1.0 --joint-damping 5e-7 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.1 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B06 asym-mild

# Cell 13 — B14-aggr: S3-damp + bursty + asym-aggr
run_cell "B14-S3dampBursty-aggr" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B14 asym-aggr

# Cell 14 — N4: S3 + joint-armature 1e-9
run_cell "N4-armature-1em9" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --joint-armature 1e-9 \
    --wandb-tags s16-ms N N4 asym-mild

# Cell 15 — X2: sim-dt 0.000625 (2x finer integration)
run_cell "X2-simdt-6p25em4" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --sim-dt 0.000625 \
    --wandb-tags s16-ms X X2 sim-dt asym-mild

# Cell 16 — B05-mild: F4 slow-soft + asym-mild
run_cell "B05-F4-mild" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B05 asym-mild

# Cell 17 — B18-sym25: fs=1.4 stiff damp + sym25
run_cell "B18-fs14stiff-sym25" \
    --force-scale 1.4 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B18 sym25

echo "================================================================"
echo "=== S16-MS part 3/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
