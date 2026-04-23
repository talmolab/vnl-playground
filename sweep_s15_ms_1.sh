#!/bin/bash
# SWEEP S15-MS part 1/6 — GPU 1 (2-GPU machine #1, GPU 0)
# Priority-ordered: leader + bursty-reward + mid-goldilocks (+ shoulder@fs=1.2 + strong-damp if budget)
# Cells: A1 baselineA, R1 bursty, F5 fs1p05_d8e7, S5 shWeak_fs1p2, A5 anchorCstrong
# Worker type: 2-GPU. 5 candidates, single-seed. Budget-aware (12h default).
# Spec: docs/superpowers/specs/2026-04-23-s15-ms-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-part1"
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
        echo "[S15-MS-1 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s15-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S15-MS-1 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — A1 baselineA  (s13 single-seed leader replication)
run_cell "A1-baselineA" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates A1 baselineA seed1 p98clip

# Cell 2 — R1 bursty  (cc=0, cdc=0.05 — allow sharp triceps bursts)
run_cell "R1-bursty" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    --wandb-tags s15-ms candidates R1 bursty seed1 p98clip

# Cell 3 — F5 fs1p05_d8e7  (dense goldilocks fill)
run_cell "F5-fs1p05_d8e7" \
    --force-scale 1.05 --joint-damping 8e-7 --shoulder-damping 8e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates F5 fs1p05_d8e7 seed1 p98clip

# Cell 4 — S5 shWeak_fs1p2  (shoulder decouple at fs=1.2, budget permitting)
run_cell "S5-shWeak_fs1p2" \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates S5 shWeak_fs1p2 seed1 p98clip

# Cell 5 — A5 anchorCstrong  (higher-damping anchor-C variant, budget permitting)
run_cell "A5-anchorCstrong" \
    --force-scale 1.2 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --wandb-tags s15-ms candidates A5 anchorCstrong seed1 p98clip

echo "================================================================"
echo "=== S15-MS part 1/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
