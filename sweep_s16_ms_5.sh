#!/bin/bash
# SWEEP S16-MS part 5/6 — GPU 5 (1-GPU machine #1)
# Priority-ordered: C01 biceps-force 0.7× + X9 loose XML + V1 seed 2
# 17 cells, single-seed, budget-aware (12h default).
# Spec: docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s16-ms-part5"
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
        echo "[S16-MS-5 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s16-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S16-MS-5 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# ===== Priority-1 =====

# Cell 1 — C01: S3 + biceps-force 0.07 (0.7× — biceps amplitude fix)
run_cell "C01-bforce07x" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --biceps-force 0.07 \
    --wandb-tags s16-ms C C01 biceps-force-07x asym-mild priority1

# Cell 2 — X9: loose XML (structural alternative baseline)
run_cell "X9-loose-xml" \
    --walker-xml vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_loose.xml \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms X X9 loose-xml asym-mild priority1

# Cell 3 — V1: seed-variance anchor, seed 2
run_cell "V1-seed2" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 2 \
    ${TAU_MILD} \
    --wandb-tags s16-ms V V1 seed-variance seed2 asym-mild priority1

# ===== Tier 2 =====

# Cell 4 — T1g: tau_act=55 global (extreme slowdown)
run_cell "T1g-tau55" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.055 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T tau55

# Cell 5 — T4c: b=55 br=40 tl=25 tla=25 (all-slow asymmetric)
run_cell "T4c-b55-allslow" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.025 --biceps-tau-act 0.055 --brachialis-tau-act 0.040 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T b55-allslow

# Cell 6 — B03-sym25: S3 config + sym25 tau
run_cell "B03-S3-sym25" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B03 S3 sym25

# Cell 7 — B08-aggr: slow-soft + bursty + asym-aggr
run_cell "B08-slowsoftBursty-aggr" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B08 asym-aggr

# Cell 8 — B16-aggr: high-fs strong damp bursty + asym-aggr
run_cell "B16-highFsBursty-aggr" \
    --force-scale 1.3 --joint-damping 1.2e-6 --shoulder-damping 1.2e-6 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B16 asym-aggr

# Cell 9 — C03: S3 + triceps forces 0.08 (symmetric triceps amplitude cut)
run_cell "C03-tforce08x" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --triceps-long-force 0.08 --triceps-lat-force 0.08 \
    --wandb-tags s16-ms C C03 triceps-force-08x asym-mild

# Cell 10 — C10: F4 aggr + biceps-force 0.06 (biceps-bad cell + aggressive fix)
run_cell "C10-F4aggr-bforce06" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} --biceps-force 0.06 \
    --wandb-tags s16-ms C C10 F4-aggr-bforce asym-aggr

# Cell 11 — B01-mild: s13 anchor-A + asym-mild
run_cell "B01-anchorA-mild" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B01 anchorA asym-mild

# Cell 12 — B07-mild: high fs + weak shoulder + asym-mild
run_cell "B07-highFsWeakSh-mild" \
    --force-scale 1.3 --joint-damping 9e-7 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B07 asym-mild

# Cell 13 — B05-sym25: F4 slow-soft + sym25
run_cell "B05-F4-sym25" \
    --force-scale 0.9 --joint-damping 6e-7 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B05 sym25

# Cell 14 — B12-aggr: fs=1.2 S3-damp + asym-aggr
run_cell "B12-fs12S3damp-aggr" \
    --force-scale 1.2 --joint-damping 1.5e-6 --shoulder-damping 5e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B12 asym-aggr

# Cell 15 — T2a: tau_deact=30 at tau_act=25 (fast decay)
run_cell "T2a-tdeact30" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.025 --muscle-tau-deact 0.030 \
    --wandb-tags s16-ms T group-T tdeact30

# Cell 16 — B06-sym25: s11 goldilocks + sym25
run_cell "B06-s11goldi-sym25" \
    --force-scale 1.0 --joint-damping 5e-7 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.1 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B06 sym25

# Cell 17 — B11-sym25: weakest shoulder + sym25
run_cell "B11-weakestSh-sym25" \
    --force-scale 1.0 --joint-damping 6e-7 --shoulder-damping 3e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B11 sym25

echo "================================================================"
echo "=== S16-MS part 5/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
