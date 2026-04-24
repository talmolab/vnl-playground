#!/bin/bash
# SWEEP S16-MS part 4/6 — GPU 4 (2-GPU machine #2, GPU 1)
# Priority-ordered: T2b tau_deact=60 + B01 anchor-A asym-aggr + X4 high inertia
# 17 cells, single-seed, budget-aware (12h default).
# Spec: docs/superpowers/specs/2026-04-24-s16-ms-tau-asymmetry-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

WANDB_GROUP="s16-ms-part4"
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
        echo "[S16-MS-4 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="s16-ms-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[S16-MS-4 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
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

# Cell 1 — T2b: tau_deact=60 at tau_act=25 (long decay probe)
run_cell "T2b-tdeact60" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.025 --muscle-tau-deact 0.060 \
    --wandb-tags s16-ms T group-T tdeact60 priority1

# Cell 2 — B01-aggr: s13 anchor-A + asym-aggr
run_cell "B01-anchorA-aggr" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B01 anchorA asym-aggr priority1

# Cell 3 — X4: body-diaginertia 5e-6 (higher inertia)
run_cell "X4-bodyinertia-5em6" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --body-diaginertia 5e-6 \
    --wandb-tags s16-ms X X4 bodyinertia-5em6 asym-mild priority1

# ===== Tier 2 =====

# Cell 4 — T1b: tau_act=15 global (near-default)
run_cell "T1b-tau15" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.015 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T tau15

# Cell 5 — T3d: biceps-only tau_act=55 (over-shoot test)
run_cell "T3d-biceps55" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.015 --biceps-tau-act 0.055 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T biceps55

# Cell 6 — B02-mild: s13 anchor-C + asym-mild
run_cell "B02-anchorC-mild" \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 1e-6 \
    --control-cost 0.035 --control-diff-cost 0.0 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B02 anchorC asym-mild

# Cell 7 — B04-aggr: R2 smoothOnly + asym-aggr
run_cell "B04-R2smoothOnly-aggr" \
    --force-scale 1.1 --joint-damping 9e-7 --shoulder-damping 9e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B04 asym-aggr

# Cell 8 — B09-sym25: fs=1.2 asym damp smoothOnly + sym25
run_cell "B09-fs12asymSmoothOnly-sym25" \
    --force-scale 1.2 --joint-damping 1.2e-6 --shoulder-damping 5e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B09 sym25

# Cell 9 — C07: S3 + joints-vel-weight 0.0 (drop velocity tracking)
run_cell "C07-jvel00" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --joints-vel-weight 0.0 \
    --wandb-tags s16-ms C C07 jvel00 asym-mild

# Cell 10 — C08: S3 + muscle-tau-deact 0.100 (long-decay probe)
run_cell "C08-tdeact100" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.020 --biceps-tau-act 0.030 --brachialis-tau-act 0.025 \
    --muscle-tau-deact 0.100 \
    --wandb-tags s16-ms C C08 tdeact100-mild asym-mild

# Cell 11 — B11-aggr: weakest shoulder + asym-aggr
run_cell "B11-weakestSh-aggr" \
    --force-scale 1.0 --joint-damping 6e-7 --shoulder-damping 3e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_AGGR} \
    --wandb-tags s16-ms B B11 asym-aggr

# Cell 12 — B12-sym25: fs=1.2 S3-damp + sym25
run_cell "B12-fs12S3damp-sym25" \
    --force-scale 1.2 --joint-damping 1.5e-6 --shoulder-damping 5e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B12 sym25

# Cell 13 — B15-mild: low damp bursty + asym-mild
run_cell "B15-lowdamp-bursty-mild" \
    --force-scale 1.0 --joint-damping 5e-7 --shoulder-damping 5e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms B B15 asym-mild

# Cell 14 — B17-sym25: fs=1.2 S3-damp weaker shoulder + sym25
run_cell "B17-fs12weakSh-sym25" \
    --force-scale 1.2 --joint-damping 1.5e-6 --shoulder-damping 4e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_SYM25} \
    --wandb-tags s16-ms B B17 sym25

# Cell 15 — N2: force-scale 1.5 (high-force corner)
run_cell "N2-fs15" \
    --force-scale 1.5 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} \
    --wandb-tags s16-ms N N2 fs15 asym-mild

# Cell 16 — X5: body-diaginertia 2e-7 (lower inertia)
run_cell "X5-bodyinertia-2em7" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    ${TAU_MILD} --body-diaginertia 2e-7 \
    --wandb-tags s16-ms X X5 bodyinertia-2em7 asym-mild

# Cell 17 — T1a: tau_act=10 global (MuJoCo default — reproduces s15)
run_cell "T1a-tau10" \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 1 \
    --muscle-tau-act 0.010 --muscle-tau-deact 0.040 \
    --wandb-tags s16-ms T group-T tau10 s15-baseline

echo "================================================================"
echo "=== S16-MS part 4/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
