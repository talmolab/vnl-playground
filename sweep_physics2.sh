#!/bin/bash
# Sweep 2: Fine-grained damping × force grid, anchored on best S1 results
# Best from S1: damp=1e-7/arm=4e-10 (~240) and damp=1e-6/arm=4e-10 (~235)
# Fix armature=4e-10, sweep damping {1e-5..1e-10} × force {1.0, 0.5, 0.333, 0.25}
# Total: 24 runs
set -euo pipefail

SCRIPT="vnl_playground/train_mouse_janelia_imitation.py"
GROUP="sweep2-physics"
COMMON="--wandb-group $GROUP"
N=0

run() {
    local tag="$1"; shift
    N=$((N + 1))
    local name=$(printf "S2-%02d-%s" "$N" "$tag")
    echo "============================================"
    echo "Starting: $name ($tag)"
    echo "============================================"
    python "$SCRIPT" --run-name "$name" --tag "$tag" \
        --wandb-tags janelia sweep2 "$tag" \
        $COMMON "$@"
    echo "Finished: $name"
    echo ""
}

ARM=4e-10  # fixed at the S1 winner

# ── Full grid: damping × force_scale ─────────────────────────────────────────
for DAMP in 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10; do
    for FSCALE in 1.0 0.5 0.333 0.25; do
        run "d${DAMP}-f${FSCALE}-arm${ARM}" \
            --joint-damping "$DAMP" --joint-armature "$ARM" --force-scale "$FSCALE"
    done
done

echo "============================================"
echo "Sweep 2 complete! ($N runs)"
echo "============================================"
