#!/bin/bash
# SWEEP S15-MS part 5/6 — GPU 5 (1-GPU machine #1)
# Priority-ordered: anchor-A-mid + interp+asym + shoulder@fs=1.3 (+ fs=1.25 + s11-style if budget)
# Cells: A4 anchorAmid, I2 interpAC_asym, S6 shWeak_fs1p3, F7 fs1p25_d1p1, R3 s11style
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-part5"
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
        echo "[S15-MS-5 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s15-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S15-MS-5 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — A4 anchorAmid  (damping between anchor A and C)
run_cell "A4-anchorAmid" \
    --force-scale 1.1 --joint-damping 7e-7 --shoulder-damping 7e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates A4 anchorAmid seed1 p100

# Cell 2 — I2 interpAC_asym  (between A and C + asymmetric shoulder)
run_cell "I2-interpAC_asym" \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 5e-7 \
    --control-cost 0.03 --control-diff-cost 0.0125 --seed 1 \
    --wandb-tags s15-ms candidates I2 interpAC_asym seed1 p100

# Cell 3 — S6 shWeak_fs1p3  (weak shoulder at fs=1.3)
run_cell "S6-shWeak_fs1p3" \
    --force-scale 1.3 --joint-damping 1e-6 --shoulder-damping 5e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates S6 shWeak_fs1p3 seed1 p100

# Cell 4 — F7 fs1p25_d1p1  (dense fs=1.25 fill)
run_cell "F7-fs1p25_d1p1" \
    --force-scale 1.25 --joint-damping 1.1e-6 --shoulder-damping 1.1e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F7 fs1p25_d1p1 seed1 p100

# Cell 5 — R3 s11style  (cc=0.05, cdc=0.1 — full s11 reward shaping)
run_cell "R3-s11style" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.05 --control-diff-cost 0.1 --seed 1 \
    --wandb-tags s15-ms candidates R3 s11style seed1 p100

echo "================================================================"
echo "=== S15-MS part 5/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
