#!/bin/bash
# SWEEP sAnimal part 3/6 — Job2 GPU0
# 8 cells: AT006 × {F4, F5, F6, F7, F8} + AT009 × {C0, F1, F2}.
# Spec: docs/superpowers/specs/2026-05-02-sAnimal-per-animal-hyperparameter-sweep-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
eval "$(conda shell.bash hook)"
conda activate track_mjx

WANDB_GROUP="sAnimal-part3"
BUDGET_SECONDS=$(( ${BUDGET_HOURS:-30} * 3600 ))
ESTIMATED_RUN_SECONDS=${ESTIMATED_RUN_SECONDS:-14400}

REF_DATA=/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals

BASE_ARGS=(
    --reference-data-path "${REF_DATA}"
    --emg-animals A36-1 AT006 AT009 AT012 AT013
    --emg-norm-method z_baseline_x2
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
TOTAL=8
CELL=0

run_cell() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS )); then
        echo "----------------------------------------------------------------"
        echo "[sAnimal-3 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="sAnimal-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[sAnimal-3 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# AT006 × F4
run_cell "AT006-F4" \
    --train-animals AT006 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT006 F4 fractional

# AT006 × F5
run_cell "AT006-F5" \
    --train-animals AT006 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT006 F5 fractional

# AT006 × F6
run_cell "AT006-F6" \
    --train-animals AT006 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT006 F6 fractional

# AT006 × F7
run_cell "AT006-F7" \
    --train-animals AT006 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT006 F7 fractional

# AT006 × F8
run_cell "AT006-F8" \
    --train-animals AT006 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT006 F8 fractional

# AT009 × C0
run_cell "AT009-C0" \
    --train-animals AT009 \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
    --wandb-tags sAnimal AT009 C0 center

# AT009 × F1
run_cell "AT009-F1" \
    --train-animals AT009 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT009 F1 fractional

# AT009 × F2
run_cell "AT009-F2" \
    --train-animals AT009 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT009 F2 fractional

echo "================================================================"
echo "=== sAnimal part 3/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
