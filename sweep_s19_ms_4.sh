#!/bin/bash
# SWEEP S19-MS part 4/6 — Job2 GPU1
# 2 cells: A1.s4 (anchor seed 4 at C1) + γ4 (s15-F1 revisit, coupled-equal damping)
# Spec: docs/superpowers/specs/2026-05-02-s19-ms-bayesian-population-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
if [ -f /root/vast/eric/track-mjx/.venv/bin/activate ]; then
    source /root/vast/eric/track-mjx/.venv/bin/activate
else
    eval "$(conda shell.bash hook)"
    conda activate track_mjx
fi

WANDB_GROUP="s19-ms-part4"
BUDGET_SECONDS=$(( ${BUDGET_HOURS:-10} * 3600 ))
ESTIMATED_RUN_SECONDS=${ESTIMATED_RUN_SECONDS:-14400}
PREFLIGHT_SECONDS=${PREFLIGHT_SECONDS:-2400}
PREFLIGHT_REWARD_FLOOR=${PREFLIGHT_REWARD_FLOOR:-250}

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
PREFLIGHT_FAILED=()
TOTAL=2
CELL=0

run_cell() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS )); then
        echo "[S19-MS-4 ${CELL}/${TOTAL}] ${TAG} — SKIPPED"
        SKIPPED+=("${TAG}"); return
    fi
    local RUN_NAME="s19-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "[S19-MS-4 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
    echo "  $@"
    if python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" "$@" 2>&1 | tee "${LOG}"; then
        OK+=("${RUN_NAME}"); echo "[OK] ${RUN_NAME}"
    else
        CRASHED+=("${RUN_NAME}"); echo "[CRASHED] ${RUN_NAME}"
    fi
    echo
}

preflight_then_full() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS + PREFLIGHT_SECONDS )); then
        echo "[S19-MS-4 ${CELL}/${TOTAL}] ${TAG} — SKIPPED"
        SKIPPED+=("${TAG}"); return
    fi
    local PRE_NAME="s19-ms-${TAG}-preflight-$(date +%Y%m%d-%H%M%S)"
    local PRE_LOG="/tmp/sweep_${PRE_NAME}.log"
    echo "[S19-MS-4 ${CELL}/${TOTAL}] ${PRE_NAME} preflight 50M steps"
    if ! python train_mouse_janelia_sigmoid_moving_shoulder.py \
        --reference-data-path "${REF_DATA}" \
        --emg-animals A36-1 AT006 AT009 AT012 AT013 \
        --emg-norm-method p98_per_muscle --emg-norm-percentile 98 \
        --ctrl-dt 0.0025 --sim-dt 0.00125 --episode-length 100 --qvel-init zeros \
        --joint-armature 4e-10 \
        --joints-weight 5.0 --joints-vel-weight 0.5 --wrist-pos-weight 0.1 --bodies-pos-weight 0.1 \
        --num-timesteps 50000000 --num-evals 1 --no-wandb \
        --tag "${TAG}-preflight" --run-name "${PRE_NAME}" "$@" 2>&1 | tee "${PRE_LOG}"; then
        echo "[PREFLIGHT-CRASHED] ${PRE_NAME}"
        PREFLIGHT_FAILED+=("${TAG}"); return
    fi
    local FINAL_REWARD
    FINAL_REWARD=$(grep -oE "eval/episode_reward[^a-zA-Z_]*: *-?[0-9]+\.[0-9]+" "${PRE_LOG}" | tail -1 | grep -oE "\-?[0-9]+\.[0-9]+" | tail -1)
    if [ -z "${FINAL_REWARD}" ]; then
        echo "[PREFLIGHT-NOMETRIC] ${TAG} — proceeding"
    else
        local PASS
        PASS=$(awk -v r="${FINAL_REWARD}" -v f="${PREFLIGHT_REWARD_FLOOR}" 'BEGIN { print (r >= f) ? 1 : 0 }')
        if [ "${PASS}" != "1" ]; then
            echo "[PREFLIGHT-FAILED] ${TAG} reward ${FINAL_REWARD} < ${PREFLIGHT_REWARD_FLOOR}"
            PREFLIGHT_FAILED+=("${TAG}"); return
        fi
        echo "[PREFLIGHT-OK] ${TAG} reward ${FINAL_REWARD} >= ${PREFLIGHT_REWARD_FLOOR}"
    fi
    run_cell "${TAG}" "$@"
}

# ===== Cell A1.s4 — anchor seed 4 at C1 =====
run_cell "A1-s4-C1" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 4 \
    --wandb-tags s19-ms cohort sigma-anchor C1-replicate

# ===== Cell γ4 — s15-F1 revisit (coupled-equal damping at 1.2e-6) =====
preflight_then_full "g4-s15-F1" \
    --force-scale 1.0 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
    --wandb-tags s19-ms cohort gamma s15-revisit coupled-damp

echo "================================================================"
echo "=== S19-MS part 4/6 complete ==="
echo "  OK              (${#OK[@]}):";               for R in "${OK[@]}";              do echo "    OK  ${R}"; done
echo "  CRASHED         (${#CRASHED[@]}):";          for R in "${CRASHED[@]}";         do echo "    BAD ${R}"; done
echo "  PREFLIGHT_FAIL  (${#PREFLIGHT_FAILED[@]}):"; for R in "${PREFLIGHT_FAILED[@]}"; do echo "    PFL ${R}"; done
echo "  SKIPPED         (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
