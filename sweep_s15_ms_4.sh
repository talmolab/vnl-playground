#!/bin/bash
# SWEEP S15-MS part 4/6 — GPU 4 (2-GPU machine #2, GPU 1)
# Priority-ordered: weakSh+bursty interaction + fs=1.3 + strong-shoulder (+ mild-bursty + s11-goldilocks if budget)
# Cells: I1 weakSh_bursty, F3 fs1p3_d1p2, S4 shStrong, R6 mildBursty, A3 s11goldilocks
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-part4"
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
        echo "[S15-MS-4 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s15-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S15-MS-4 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — I1 weakSh_bursty  (weak shoulder + bursty reward — combined top-hypothesis levers)
run_cell "I1-weakSh_bursty" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 3e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    --wandb-tags s15-ms candidates I1 weakSh_bursty seed1 p98clip

# Cell 2 — F3 fs1p3_d1p2  (fs=1.3 at higher damping)
run_cell "F3-fs1p3_d1p2" \
    --force-scale 1.3 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F3 fs1p3_d1p2 seed1 p98clip

# Cell 3 — S4 shStrong  (shoulder stiffer than elbow — reverse of S1)
run_cell "S4-shStrong" \
    --force-scale 1.1 --joint-damping 6e-7 --shoulder-damping 1.5e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates S4 shStrong seed1 p98clip

# Cell 4 — R6 mildBursty  (cc=0.015, cdc=0.035 — between bursty and default)
run_cell "R6-mildBursty" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.015 --control-diff-cost 0.035 --seed 1 \
    --wandb-tags s15-ms candidates R6 mildBursty seed1 p98clip

# Cell 5 — A3 s11goldilocks  (s11 d5em7 fs1p0 reference)
run_cell "A3-s11goldilocks" \
    --force-scale 1.0 --joint-damping 5e-7 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.1 --seed 1 \
    --wandb-tags s15-ms candidates A3 s11goldilocks seed1 p98clip

echo "================================================================"
echo "=== S15-MS part 4/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
