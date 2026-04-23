#!/bin/bash
# s15-ms Branch 1 — 5-seed replication of the s13 goldilocks region under
# the new EMG metric pipeline (--emg-norm-percentile 100, lagged_corr,
# per-trial corr, phase_lag logged every eval cycle).
#
# Anchor cell: s13 anchor-A fs=1.1 (current single-run leader at R=411,
# bcorr=0.70, tcorr=0.58, bmae=0.12, tmae=0.13 under the old metric).
# Sibling cells fs=1.0 and fs=1.2 bracket the leader to catch any seed-shift
# in the peak. 3 cells × 5 seeds = 15 runs, ~1.5–2 h each.
#
# Spec: docs/superpowers/specs/2026-04-23-s15-ms-design.md
# Plan: docs/superpowers/plans/2026-04-23-s15-ms-implementation.md
# Parent sweep script this is modeled on: sweep_s13_ms_1.sh

set -o pipefail
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s15-ms-branch1"

BASE_ARGS=(
    --ctrl-dt 0.0025
    --sim-dt 0.00125
    --episode-length 100
    --qvel-init zeros
    --joint-armature 4e-10
    --joint-damping 9e-7
    --control-cost 0.025
    --control-diff-cost 0.025
    --joints-weight 5.0
    --joints-vel-weight 0.5
    --wrist-pos-weight 0.1
    --bodies-pos-weight 0.1
    --num-timesteps 800000000
    --num-evals 8
    --emg-norm-percentile 100
    --wandb-group "${WANDB_GROUP}"
)

CRASHED=()
OK=()
CELL=0
# Cells (fs values) × seeds. Run only the ones selected by CELLS_TO_RUN /
# SEEDS_TO_RUN env vars, if set — otherwise run all 15.
FSES=(${CELLS_TO_RUN:-"1.0 1.1 1.2"})
SEEDS=(${SEEDS_TO_RUN:-"1 2 3 4 5"})
TOTAL=$(( ${#FSES[@]} * ${#SEEDS[@]} ))

run_cell() {
    local FS="$1"
    local SEED="$2"
    local FS_TAG="fs${FS/./p}"                # 1.1 -> fs1p1
    local TAG="s15-ms-branch1-anchorA-${FS_TAG}-s${SEED}"
    local RUN_NAME="${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    CELL=$(( CELL + 1 ))
    echo "================================================================"
    echo "[s15-ms-branch1 ${CELL}/${TOTAL}] ${RUN_NAME}"
    echo "  --force-scale ${FS} --seed ${SEED}"
    echo "================================================================"
    if python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
            --force-scale "${FS}" --seed "${SEED}" \
            --tag "${TAG}" --run-name "${RUN_NAME}" \
            --wandb-tags s15-ms branch1 anchorA "${FS_TAG}" "seed${SEED}" p100 \
            2>&1 | tee "${LOG}"; then
        OK+=("${RUN_NAME}")
        echo "[OK] ${RUN_NAME}"
    else
        CRASHED+=("${RUN_NAME}")
        echo "[CRASHED] ${RUN_NAME} — see ${LOG}"
    fi
    echo
}

for FS in "${FSES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_cell "${FS}" "${SEED}"
    done
done

echo "================================================================"
echo "=== s15-ms Branch 1 complete ==="
echo "  OK      (${#OK[@]}/${TOTAL}):"; for R in "${OK[@]}"; do echo "    OK  $R"; done
echo "  CRASHED (${#CRASHED[@]}/${TOTAL}):"; for R in "${CRASHED[@]}"; do echo "    BAD $R"; done
echo "================================================================"
