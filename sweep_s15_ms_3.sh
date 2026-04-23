#!/bin/bash
# SWEEP S15-MS part 3/6 — GPU 3 (2-GPU machine #2, GPU 0)
# Priority-ordered: anchor-C + fs=1.15 fill + light-penalty (+ mid-shoulder + low-fs if budget)
# Cells: A2 anchorCfs1p2, F6 fs1p15_d9e7, R5 lightPenalty, S2 shMid6e7, F4 fs0p9_d6e7
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-part3"
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
        echo "[S15-MS-3 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s15-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S15-MS-3 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — A2 anchorCfs1p2  (d=1e-6, cc=0.035, cdc=0)
run_cell "A2-anchorCfs1p2" \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 1e-6 \
    --control-cost 0.035 --control-diff-cost 0.0 --seed 1 \
    --wandb-tags s15-ms candidates A2 anchorCfs1p2 seed1 p98clip

# Cell 2 — F6 fs1p15_d9e7  (dense mid-point fs=1.15)
run_cell "F6-fs1p15_d9e7" \
    --force-scale 1.15 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F6 fs1p15_d9e7 seed1 p98clip

# Cell 3 — R5 lightPenalty  (cc=0.01, cdc=0.02 — mild reward shaping)
run_cell "R5-lightPenalty" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.01 --control-diff-cost 0.02 --seed 1 \
    --wandb-tags s15-ms candidates R5 lightPenalty seed1 p98clip

# Cell 4 — S2 shMid6e7  (shoulder=6e-7, intermediate decouple)
run_cell "S2-shMid6e7" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates S2 shMid6e7 seed1 p98clip

# Cell 5 — F4 fs0p9_d6e7  (lower fs + lower damping)
run_cell "F4-fs0p9_d6e7" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F4 fs0p9_d6e7 seed1 p98clip

echo "================================================================"
echo "=== S15-MS part 3/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
