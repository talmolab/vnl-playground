#!/bin/bash
# SWEEP S16-MS part 2/6 — GPU 2 (2-GPU machine #1, GPU 1)
# Priority-ordered: T3c biceps-only-45 + S3 asym-mild + X7 broken-kinematic-tracking
# 17 cells, single-seed, budget-aware (12h default).
# Spec: docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s16-ms-part2"
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

TAU_SYM25="--muscle-tau-act 0.025 --muscle-tau-deact 0.040"
TAU_MILD="--muscle-tau-act 0.020 --biceps-tau-act 0.030 --brachialis-tau-act 0.025 --muscle-tau-deact 0.040"
TAU_AGGR="--muscle-tau-act 0.020 --biceps-tau-act 0.045 --brachialis-tau-act 0.030 --muscle-tau-deact 0.040"

START_TIME=$(date +%s)
CRASHED=()
OK=()
SKIPPED=()
TOTAL=17
CELL=0

run_cell() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS )); then
        echo "----------------------------------------------------------------"
        echo "[S16-MS-2 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s16-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S16-MS-2 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# ===== Priority-1 (must-run) =====

# Cell 1 — T3c: biceps-only tau_act=45 (matches measured biceps lag)
run_cell "T3c-biceps45" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.015 --biceps-tau-act 0.045 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T biceps45 priority1

# Cell 2 — B03-mild: S3 config + asym-mild
run_cell "B03-S3-mild" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B03 S3 asym-mild priority1

# Cell 3 — X7: break kinematic-tracking dictatorship (all kinematic weights ÷10)
run_cell "X7-broken-tracking" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --joints-weight 0.5 --joints-vel-weight 0.05 --wrist-pos-weight 0.02 --bodies-pos-weight 0.02 \
    --wandb-tags s16-ms X X7 broken-tracking asym-mild priority1

# ===== Tier 2 (should-run) =====

# Cell 4 — T1f: tau_act=40 global (heavy slowdown)
run_cell "T1f-tau40" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.040 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T tau40

# Cell 5 — T3b: biceps-only tau_act=35 (mid biceps slowdown)
run_cell "T3b-biceps35" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.015 --biceps-tau-act 0.035 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T biceps35

# Cell 6 — B06-aggr: s11 goldilocks + asym-aggr
run_cell "B06-s11goldi-aggr" \
    --force-scale 1.0 --joint-damping 5e-7 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.1 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B06 asym-aggr

# Cell 7 — B02-aggr: s13 anchor-C + asym-aggr
run_cell "B02-anchorC-aggr" \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 1e-6 \
    --control-cost 0.035 --control-diff-cost 0.0 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B02 anchorC asym-aggr

# Cell 8 — C06: S3 + joints-vel-weight 0.2 (looser velocity tracking)
run_cell "C06-jvel02" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --joints-vel-weight 0.2 \
    --wandb-tags s16-ms C C06 jvel02 asym-mild

# Cell 9 — C05: S3 + saturation-cost 0.02
run_cell "C05-satcost002" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --saturation-cost 0.02 \
    --wandb-tags s16-ms C C05 sat002 asym-mild

# Cell 10 — B11-mild: weakest shoulder + asym-mild
run_cell "B11-weakestSh-mild" \
    --force-scale 1.0 --joint-damping 6e-7 --shoulder-damping 3e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B11 asym-mild

# Cell 11 — B13-aggr: mid-stiff + weak shoulder + smoothOnly + asym-aggr
run_cell "B13-midStiffWeakSh-aggr" \
    --force-scale 1.1 --joint-damping 1.2e-6 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B13 asym-aggr

# Cell 12 — B04-sym25: R2 smoothOnly + sym25
run_cell "B04-R2smoothOnly-sym25" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B04 sym25

# Cell 13 — B16-mild: high-fs strong damp bursty + asym-mild
run_cell "B16-highFsBursty-mild" \
    --force-scale 1.3 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B16 asym-mild

# Cell 14 — N5: S3 + joint-armature 4e-9 (much higher inertia)
run_cell "N5-armature-4em9" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --joint-armature 4e-9 \
    --wandb-tags s16-ms N N5 asym-mild

# Cell 15 — X3: S3 + discounting 0.995 (longer horizon)
run_cell "X3-disc0995" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --discounting 0.995 \
    --wandb-tags s16-ms X X3 discounting asym-mild

# Cell 16 — T2c: tau_deact=100 at tau_act=25 (very long decay)
run_cell "T2c-tdeact100" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.025 --muscle-tau-deact 0.100 \
    --wandb-tags s16-ms T group-T tdeact100

# Cell 17 — B18-aggr: fs=1.4 stiff damp + asym-aggr
run_cell "B18-fs14stiff-aggr" \
    --force-scale 1.4 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B18 asym-aggr

echo "================================================================"
echo "=== S16-MS part 2/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
