#!/bin/bash
# SWEEP S16-MS part 1/6 — GPU 1 (2-GPU machine #1, GPU 0)
# Priority-ordered: tau=25 global + S3 asym-aggr + qvel_init=reference (novel)
# 17 cells, single-seed, budget-aware (12h default).
# Spec: docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s16-ms-part1"
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

# Tau profiles (reused across cells)
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
        echo "[S16-MS-1 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s16-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S16-MS-1 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — T1d: tau_act=25 global at S3 (primary hypothesis midpoint)
run_cell "T1d-tau25" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms T group-T tau25 priority1

# Cell 2 — B03-aggr: S3 config + asym-aggr tau profile
run_cell "B03-S3-aggr" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B03 S3 asym-aggr priority1

# Cell 3 — X1: qvel_init=reference (novel probe)
run_cell "X1-qvel-reference" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --qvel-init reference \
    --wandb-tags s16-ms X X1 qvel-reference asym-mild priority1

# ===== Tier 2 (should-run) =====

# Cell 4 — T1e: tau_act=30 global at S3
run_cell "T1e-tau30" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.030 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T tau30

# Cell 5 — T3e: biceps-only tau_act=70 (extreme biceps slowdown)
run_cell "T3e-biceps70" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.015 --biceps-tau-act 0.070 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T biceps70

# Cell 6 — B01-sym25: s13 anchor-A + sym25
run_cell "B01-anchorA-sym25" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B01 anchorA sym25

# Cell 7 — B05-aggr: F4 slow-soft + asym-aggr
run_cell "B05-F4-aggr" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B05 F4 asym-aggr

# Cell 8 — B12-mild: S3-damp at fs=1.2 + asym-mild
run_cell "B12-fs12S3damp-mild" \
    --force-scale 1.2 --joint-damping 1.5e-6 --shoulder-damping 5e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B12 asym-mild

# Cell 9 — C02: S3 + biceps-force 0.5× (strong amplitude cut)
run_cell "C02-bforce05x" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --biceps-force 0.05 \
    --wandb-tags s16-ms C C02 biceps-force-05x asym-mild

# Cell 10 — B10-mild: strong sym damp + asym-mild
run_cell "B10-strongsym-mild" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 1.5e-6 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B10 asym-mild

# Cell 11 — B15-aggr: low damp bursty + asym-aggr
run_cell "B15-lowdamp-bursty-aggr" \
    --force-scale 1.0 --joint-damping 5e-7 --shoulder-damping 5e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B15 asym-aggr

# Cell 12 — B17-aggr: fs=1.2 S3-damp weaker shoulder + asym-aggr
run_cell "B17-fs12weakSh-aggr" \
    --force-scale 1.2 --joint-damping 1.5e-6 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B17 asym-aggr

# Cell 13 — N3: S3 + joint-armature 1e-10 (lower inertia corner)
run_cell "N3-armature-1em10" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --joint-armature 1e-10 \
    --wandb-tags s16-ms N N3 asym-mild

# Cell 14 — B14-sym25: S3-damp + bursty + sym25
run_cell "B14-S3dampBursty-sym25" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B14 sym25

# Cell 15 — B08-mild: slow-soft + bursty + asym-mild
run_cell "B08-slowsoftBursty-mild" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B08 asym-mild

# Cell 16 — X10: alternative XML (ratios) + asym-mild at S3 base
run_cell "X10-ratios-xml" \
    --walker-xml vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_ratios.xml \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms X X10 ratios-xml asym-mild

# Cell 17 — T4a: S3 + asym-mild profile (exact-config shared anchor for comparisons)
run_cell "T4a-S3-asym-mild" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms T group-T asym-mild anchor

echo "================================================================"
echo "=== S16-MS part 1/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
