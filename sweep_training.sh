#!/bin/bash
# Sweep 2: Training/RL Parameters (11 runs, ~9 hours)
# Uses baseline physics (or best from Sweep 1).
# Reward weights/exp_scales held CONSTANT.
set -euo pipefail

SCRIPT="vnl_playground/train_mouse_janelia_imitation.py"
GROUP="sweep2-training"
COMMON="--wandb-group $GROUP"

# If Sweep 1 found better physics, add them here:
# PHYSICS="--joint-damping 1e-6 --force-scale 2.0"
PHYSICS=""

run() {
    local name="$1"; shift
    local tag="$1"; shift
    echo "============================================"
    echo "Starting: $name ($tag)"
    echo "============================================"
    python "$SCRIPT" --run-name "$name" --tag "$tag" \
        --wandb-tags janelia sweep2 "$tag" \
        $COMMON $PHYSICS "$@"
    echo "Finished: $name"
    echo ""
}

# ── Phase 2A: Reference Clip Structure (3 runs) ─────────────────────────────

run "S2-01-ref-5" "ref-5" \
    --reference-length 5

run "S2-02-ref-10" "ref-10" \
    --reference-length 10

run "S2-03-long-ep" "long-ep" \
    --reference-length 5 --episode-length 90

# ── Phase 2B: Entropy Cost (3 runs) ─────────────────────────────────────────

run "S2-04-ent-1e3" "ent-1e3" \
    --entropy-cost 1e-3

run "S2-05-ent-5e3" "ent-5e3" \
    --entropy-cost 5e-3

run "S2-06-ent-1e1" "ent-1e1" \
    --entropy-cost 1e-1

# ── Phase 2C: Control Cost (2 runs) ─────────────────────────────────────────

run "S2-07-no-ctrl" "no-ctrl" \
    --control-cost 0.0 --control-diff-cost 0.0

run "S2-08-ctrl-smooth" "ctrl-smooth" \
    --control-cost 0.01 --control-diff-cost 0.01

# ── Phase 2D: PPO Hyperparameters (3 runs) ──────────────────────────────────

run "S2-09-lr-3e4" "lr-3e4" \
    --learning-rate 3e-4

run "S2-10-disc-99" "disc-99" \
    --discounting 0.99

run "S2-11-big-batch" "big-batch" \
    --batch-size 2048 --num-minibatches 32

echo "============================================"
echo "Sweep 2 complete!"
echo "============================================"
