#!/bin/bash
# SWEEP S14-MS part 6/6 — Anchor C (d1e-6 cc0.035 cdc0.0) asymmetric core
# 8 cells (L1–L8) × 2 seeds = 16 runs.
# fs_shoulder=1.3, per-muscle biceps and triceps overrides computed from t_eff/b_eff.
# Spec: docs/superpowers/specs/2026-04-21-s14-ms-per-muscle-fs-ratio-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s14-ms-part6"
FS_SHOULDER="1.3"
ANCHOR_TAG="anchorC"
DAMP="1e-6"
CC="0.035"
CDC="0.0"

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
    --wandb-group "${WANDB_GROUP}"
)

CRASHED=()
OK=()
TOTAL=16
CELL=0

compute_override() { python3 -c "print(0.1 * $1 / ${FS_SHOULDER})"; }
fs_str()           { echo "$1" | tr '.' 'p'; }

run_s14_cell() {
    local CELL_ID="$1"
    local T_EFF="$2"
    local B_EFF="$3"
    local SEED="$4"

    local T_OVR B_OVR
    T_OVR=$(compute_override "${T_EFF}")
    B_OVR=$(compute_override "${B_EFF}")

    local T_STR B_STR
    T_STR=$(fs_str "${T_EFF}")
    B_STR=$(fs_str "${B_EFF}")

    local RATIO_TAG="t${T_STR}b${B_STR}"
    local TAG="${ANCHOR_TAG}-${CELL_ID}-${RATIO_TAG}-s${SEED}"
    local RUN_NAME="s14-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_s14_ms_${TAG}.log"
    CELL=$((CELL + 1))

    echo "----------------------------------------------------------------"
    echo "[S14-MS-6 ${CELL}/${TOTAL}] ${RUN_NAME}"
    echo "  anchor=${ANCHOR_TAG} damp=${DAMP} cc=${CC} cdc=${CDC}"
    echo "  t_eff=${T_EFF} b_eff=${B_EFF} seed=${SEED}"
    echo "  fs_shoulder=${FS_SHOULDER}  triceps_override=${T_OVR}  biceps_override=${B_OVR}"
    echo "----------------------------------------------------------------"

    if python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --joint-damping "${DAMP}" --control-cost "${CC}" --control-diff-cost "${CDC}" \
        --force-scale "${FS_SHOULDER}" \
        --biceps-force "${B_OVR}" --brachialis-force "${B_OVR}" \
        --triceps-long-force "${T_OVR}" --triceps-lat-force "${T_OVR}" \
        --seed "${SEED}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" \
        --wandb-tags s14-ms moving-shoulder "${ANCHOR_TAG}" "${CELL_ID}" "${RATIO_TAG}" "seed${SEED}" qzero asymmetric \
        2>&1 | tee "${LOG}"; then
        OK+=("${RUN_NAME}"); echo "[OK] ${RUN_NAME}"
    else
        CRASHED+=("${RUN_NAME}"); echo "[CRASHED] ${RUN_NAME} (see ${LOG})"
    fi
    echo
}

# Core asymmetric zone (L1–L8): t_eff in {1.0, 1.1, 1.2}, b_eff >= t_eff
for SEED in 1 2; do
    run_s14_cell "L1" 1.0 1.1 "${SEED}"
    run_s14_cell "L2" 1.0 1.2 "${SEED}"
    run_s14_cell "L3" 1.0 1.3 "${SEED}"
    run_s14_cell "L4" 1.0 1.4 "${SEED}"
    run_s14_cell "L5" 1.1 1.3 "${SEED}"
    run_s14_cell "L6" 1.1 1.4 "${SEED}"
    run_s14_cell "L7" 1.2 1.4 "${SEED}"
    run_s14_cell "L8" 1.2 1.5 "${SEED}"
done

echo "================================================================"
echo "=== S14-MS part 6/6 complete ==="
echo "  Successful (${#OK[@]}):"; for R in "${OK[@]}"; do echo "    OK  $R"; done
echo "  Crashed    (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD $R"; done
echo "================================================================"
