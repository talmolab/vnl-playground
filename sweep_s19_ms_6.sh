#!/bin/bash
# SWEEP S19-MS part 6/6 — Job4 GPU0
# 1 cell: A1.s5 (BONUS anchor seed 5 at C1 — strengthens σ² estimator from 5 to 6 seeds)
# Spec: docs/superpowers/specs/2026-05-02-s19-ms-bayesian-population-design.md
#
# This is OPTIONAL relative to the spec's 9-cell baseline. Skip if other priorities
# (e.g., a 4th machine isn't available, or budget is constrained).
set -o pipefail

cd /root/vast/eric/vnl-playground
if [ -f /root/vast/eric/track-mjx/.venv/bin/activate ]; then
    source /root/vast/eric/track-mjx/.venv/bin/activate
else
    eval "$(conda shell.bash hook)"
    conda activate track_mjx
fi

WANDB_GROUP="s19-ms-part6"
BUDGET_SECONDS=$(( ${BUDGET_HOURS:-10} * 3600 ))
ESTIMATED_RUN_SECONDS=${ESTIMATED_RUN_SECONDS:-14400}

REF_DATA=/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals

BASE_ARGS=(
    --reference-data-path "${REF_DATA}"
    --emg-animals A36-1 AT006 AT009 AT012 AT013
    --emg-norm-method p98_per_muscle
    --emg-norm-percentile 98
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
    --wandb-group "${WANDB_GROUP}"
)

START_TIME=$(date +%s)
CRASHED=()
OK=()
SKIPPED=()
TOTAL=1
CELL=0

run_cell() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS )); then
        echo "[S19-MS-6 ${CELL}/${TOTAL}] ${TAG} — SKIPPED"
        SKIPPED+=("${TAG}"); return
    fi
    local RUN_NAME="s19-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "[S19-MS-6 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
    echo "  $@"
    if python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" "$@" 2>&1 | tee "${LOG}"; then
        OK+=("${RUN_NAME}"); echo "[OK] ${RUN_NAME}"
    else
        CRASHED+=("${RUN_NAME}"); echo "[CRASHED] ${RUN_NAME}"
    fi
    echo
}

# ===== Cell A1.s5 — BONUS anchor seed 5 at C1 =====
run_cell "A1-s5-C1" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 5 \
    --wandb-tags s19-ms cohort sigma-anchor C1-replicate bonus-seed

echo "================================================================"
echo "=== S19-MS part 6/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}";      do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
