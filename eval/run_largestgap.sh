#!/usr/bin/env bash
# Isolated-per-width gap sweep to avoid the in-loop Warp/JAX memory leak that
# OOMs gap_sweep.py after ~4 widths. Each width runs in its own process so the
# CUDA allocator is fully reclaimed between widths.
#
# Usage: run_largestgap.sh <gpu_id> <ckpt_dir> <tag>
set -u
GPU="$1"; CKPT="$2"; TAG="$3"
PY=/root/vast/keming/SalkResearch/.venv-train/bin/python
SCRIPT=/root/vast/keming/SalkResearch/eval/gap_sweep.py
OUT=/root/vast/keming/SalkResearch/eval/gap_sweep_out
WIDTHS="0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28"

for w in $WIDTHS; do
  echo "===== [$TAG] width $w ====="
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    "$PY" "$SCRIPT" --ckpt "$CKPT" --widths "$w" --n_envs 256 \
    --out "$OUT/${TAG}_w${w}.json" 2>&1
done
echo "ALL WIDTHS DONE for $TAG"
