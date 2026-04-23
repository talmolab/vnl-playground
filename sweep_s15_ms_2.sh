#!/bin/bash
# SWEEP S15-MS part 2/6 — GPU 2 (2-GPU machine #1, GPU 1)
# Priority-ordered: shoulder-decouple + fs=1.2 fill + smooth-only (+ fs=1.3+weakSh + slow-dynamics if budget)
# Cells: S1 shWeak3e7, F2 fs1p2_d9e7, R2 smoothOnly, I3 fs1p3_weakSh, F1 fs1p0_d1p2
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-part2"
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
        echo "[S15-MS-2 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s15-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S15-MS-2 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — S1 shWeak3e7  (weak shoulder damping at leader fs)
run_cell "S1-shWeak3e7" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 3e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates S1 shWeak3e7 seed1 p100

# Cell 2 — F2 fs1p2_d9e7  (fs=1.2 at anchor-A damping)
run_cell "F2-fs1p2_d9e7" \
    --force-scale 1.2 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F2 fs1p2_d9e7 seed1 p100

# Cell 3 — R2 smoothOnly  (cc=0.05, cdc=0 — only action magnitude penalty)
run_cell "R2-smoothOnly" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    --wandb-tags s15-ms candidates R2 smoothOnly seed1 p100

# Cell 4 — I3 fs1p3_weakSh  (high fs + weak shoulder)
run_cell "I3-fs1p3_weakSh" \
    --force-scale 1.3 --joint-damping 9e-7 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates I3 fs1p3_weakSh seed1 p100

# Cell 5 — F1 fs1p0_d1p2  (slow arm at fs=1.0)
run_cell "F1-fs1p0_d1p2" \
    --force-scale 1.0 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F1 fs1p0_d1p2 seed1 p100

echo "================================================================"
echo "=== S15-MS part 2/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
