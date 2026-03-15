#!/bin/bash
# Sweep 1: XML Physics Parameters — full combinatorial grid
# Axes: damping {1e-5, 1e-6, 1e-7} x force_scale {1.0, 2.0, 3.0} x armature {4e-8, 4e-6, 4e-10} x stiffness {1e-12, 1e-6, 1e-3}
# Plus half-force. Total: ~40 runs, ~33 hours at ~50 min each.
set -euo pipefail

SCRIPT="vnl_playground/train_mouse_janelia_imitation.py"
GROUP="sweep1-physics"
COMMON="--wandb-group $GROUP"
N=0

run() {
    local tag="$1"; shift
    N=$((N + 1))
    local name=$(printf "S1-%02d-%s" "$N" "$tag")
    echo "============================================"
    echo "Starting: $name ($tag)"
    echo "============================================"
    python "$SCRIPT" --run-name "$name" --tag "$tag" \
        --wandb-tags janelia sweep1 "$tag" \
        $COMMON "$@"
    echo "Finished: $name"
    echo ""
}

# ── 0. Baseline (XML defaults, no overrides) ─────────────────────────────────
run "baseline"

# ── 1. Damping x Force grid (3x3 = 9 runs) ──────────────────────────────────
for DAMP in 1e-5 1e-6 1e-7; do
    for FSCALE in 1.0 2.0 3.0; do
        # Skip the (1e-5, 1.0) combo — that's basically baseline
        if [ "$DAMP" = "1e-5" ] && [ "$FSCALE" = "1.0" ]; then
            continue
        fi
        run "d${DAMP}-f${FSCALE}" \
            --joint-damping "$DAMP" --force-scale "$FSCALE"
    done
done

# ── 2. Armature axis (2 levels, at default damping & force) ──────────────────
run "arm-4e-6" \
    --joint-armature 4e-6

run "arm-4e-10" \
    --joint-armature 4e-10

# ── 3. Armature x Damping x Force combos (best-guess cross) ─────────────────
# High armature with low damping and force boost
for DAMP in 1e-6 1e-7; do
    for FSCALE in 2.0 3.0; do
        run "d${DAMP}-f${FSCALE}-arm4e-6" \
            --joint-damping "$DAMP" --force-scale "$FSCALE" --joint-armature 4e-6
    done
done

# Low armature with low damping
for DAMP in 1e-6 1e-7; do
    run "d${DAMP}-arm4e-10" \
        --joint-damping "$DAMP" --joint-armature 4e-10
done

# ── 4. Stiffness axis (at default damping/force) ────────────────────────────
run "stiff-1e-6" \
    --joint-stiffness 1e-6

run "stiff-1e-3" \
    --joint-stiffness 1e-3

# ── 5. Stiffness x Damping combos ───────────────────────────────────────────
for DAMP in 1e-6 1e-7; do
    for STIFF in 1e-6 1e-3; do
        run "d${DAMP}-stiff${STIFF}" \
            --joint-damping "$DAMP" --joint-stiffness "$STIFF"
    done
done

# Stiffness + force boost
run "d1e-6-f2-stiff1e-6" \
    --joint-damping 1e-6 --force-scale 2.0 --joint-stiffness 1e-6

run "d1e-6-f2-stiff1e-3" \
    --joint-damping 1e-6 --force-scale 2.0 --joint-stiffness 1e-3

# ── 6. Full combos: damping x force x armature x stiffness ──────────────────
# Best-guess promising corners of the 4D space
run "d1e-6-f2-arm4e-6-stiff1e-6" \
    --joint-damping 1e-6 --force-scale 2.0 --joint-armature 4e-6 --joint-stiffness 1e-6

run "d1e-6-f2-arm4e-6-stiff1e-3" \
    --joint-damping 1e-6 --force-scale 2.0 --joint-armature 4e-6 --joint-stiffness 1e-3

run "d1e-7-f2-arm4e-6-stiff1e-6" \
    --joint-damping 1e-7 --force-scale 2.0 --joint-armature 4e-6 --joint-stiffness 1e-6

run "d1e-7-f3-arm4e-6-stiff1e-6" \
    --joint-damping 1e-7 --force-scale 3.0 --joint-armature 4e-6 --joint-stiffness 1e-6

# ── 7. Force-only (no damping change) ───────────────────────────────────────
run "half-force" \
    --force-scale 0.5

echo "============================================"
echo "Sweep 1 complete! ($N runs)"
echo "============================================"
