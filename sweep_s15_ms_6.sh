#!/bin/bash
# SWEEP S15-MS part 6/6 — GPU 6 (1-GPU machine #2)
# Priority-ordered: elbow-strong + high-fs + slow+weakSh (+ no-penalty + weakSh+mildBursty if budget)
# Cells: S3 elbowStrong, F8 fs1p4_d1p2, I4 slow_weakSh, R4 noPenalty, I5 weakSh_mildBurst
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-part6"
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
    --emg-norm-percentile 100
    --wandb-group "${WANDB_GROUP}"
)

START_TIME=$(date +%s)
CRASHED=()
OK=()
SKIPPED=()
TOTAL=5
CELL=0

run_cell() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS )); then
        echo "----------------------------------------------------------------"
        echo "[S15-MS-6 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s15-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S15-MS-6 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — S3 elbowStrong  (stiff elbow, weak shoulder — "stiff-link biomechanics")
run_cell "S3-elbowStrong" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates S3 elbowStrong seed1 p100

# Cell 2 — F8 fs1p4_d1p2  (high-reward fs region at stronger damping)
run_cell "F8-fs1p4_d1p2" \
    --force-scale 1.4 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F8 fs1p4_d1p2 seed1 p100

# Cell 3 — I4 slow_weakSh  (slow arm dynamics + weak shoulder)
run_cell "I4-slow_weakSh" \
    --force-scale 1.0 --joint-damping 1.2e-6 --shoulder-damping 5e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates I4 slow_weakSh seed1 p100

# Cell 4 — R4 noPenalty  (cc=0, cdc=0 — no reward shaping at all)
run_cell "R4-noPenalty" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 1 \
    --wandb-tags s15-ms candidates R4 noPenalty seed1 p100

# Cell 5 — I5 weakSh_mildBurst  (weak shoulder + mild bursty reward)
run_cell "I5-weakSh_mildBurst" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 5e-7 \
    --control-cost 0.0 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates I5 weakSh_mildBurst seed1 p100

echo "================================================================"
echo "=== S15-MS part 6/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
