#!/bin/bash
# SWEEP S16-MS part 6/6 — GPU 6 (1-GPU machine #2)
# Priority-ordered: N1 fs=0.7 + X6 joint-stiffness + V2 seed 3
# 17 cells, single-seed, budget-aware (12h default).
# Spec: docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s16-ms-part6"
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
        echo "[S16-MS-6 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s16-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S16-MS-6 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# ===== Priority-1 =====

# Cell 1 — N1: force-scale 0.7 (low-force corner, never tested)
run_cell "N1-fs07" \
    --force-scale 0.7 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms N N1 fs07 asym-mild priority1

# Cell 2 — X6: joint-stiffness 1e-5 (passive spring)
run_cell "X6-jstiff-1em5" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --joint-stiffness 1e-5 \
    --wandb-tags s16-ms X X6 joint-stiffness asym-mild priority1

# Cell 3 — V2: seed-variance anchor, seed 3
run_cell "V2-seed3" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 3 \
    ${TAU_MILD} \
    --wandb-tags s16-ms V V2 seed-variance seed3 asym-mild priority1

# ===== Tier 2 =====

# Cell 4 — V3: seed-variance anchor, seed 4
run_cell "V3-seed4" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 4 \
    ${TAU_MILD} \
    --wandb-tags s16-ms V V3 seed-variance seed4 asym-mild

# Cell 5 — B02-sym25: s13 anchor-C + sym25
run_cell "B02-anchorC-sym25" \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 1e-6 \
    --control-cost 0.035 --control-diff-cost 0.0 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B02 anchorC sym25

# Cell 6 — B04-mild: R2 smoothOnly + asym-mild
run_cell "B04-R2smoothOnly-mild" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B04 asym-mild

# Cell 7 — B08-sym25: slow-soft bursty + sym25
run_cell "B08-slowsoftBursty-sym25" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B08 sym25

# Cell 8 — B09-aggr: fs=1.2 asym damp smoothOnly + asym-aggr
run_cell "B09-fs12asymSmoothOnly-aggr" \
    --force-scale 1.2 --joint-damping 1.2e-6 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B09 asym-aggr

# Cell 9 — B10-sym25: strong sym damp + sym25
run_cell "B10-strongsym-sym25" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 1.5e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B10 sym25

# Cell 10 — B14-mild: S3-damp + bursty + asym-mild
run_cell "B14-S3dampBursty-mild" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B14 asym-mild

# Cell 11 — B15-sym25: low damp + bursty + sym25
run_cell "B15-lowdamp-bursty-sym25" \
    --force-scale 1.0 --joint-damping 5e-7 --shoulder-damping 5e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B15 sym25

# Cell 12 — B16-sym25: high-fs strong damp bursty + sym25
run_cell "B16-highFsBursty-sym25" \
    --force-scale 1.3 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B16 sym25

# Cell 13 — B17-mild: fs=1.2 S3-damp weaker shoulder + asym-mild
run_cell "B17-fs12weakSh-mild" \
    --force-scale 1.2 --joint-damping 1.5e-6 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B17 asym-mild

# Cell 14 — B18-mild: fs=1.4 stiff damp + asym-mild
run_cell "B18-fs14stiff-mild" \
    --force-scale 1.4 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B18 asym-mild

# Cell 15 — C12: s13 anchor-A + biceps-force 0.08 (mild biceps cut on historical leader)
run_cell "C12-anchorA-bforce08" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --biceps-force 0.08 \
    --wandb-tags s16-ms C C12 anchorA-bforce08 asym-mild

# Cell 16 — B07-sym25: high fs + weak shoulder + sym25
run_cell "B07-highFsWeakSh-sym25" \
    --force-scale 1.3 --joint-damping 9e-7 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B07 sym25

# Cell 17 — B13-mild: mid-stiff weak shoulder smoothOnly + asym-mild
run_cell "B13-midStiffWeakSh-mild" \
    --force-scale 1.1 --joint-damping 1.2e-6 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B13 asym-mild

echo "================================================================"
echo "=== S16-MS part 6/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
