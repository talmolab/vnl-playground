#!/bin/bash
# Overlap sweep training — 1 billion timesteps per condition
# Waits for GPU to be free before starting.
#
# Usage: bash scripts/run_overlap_sweep_1b.sh

set -e

VENV="/home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate"
WORKDIR="/home/talmolab/Desktop/SalkResearch/vnl-playground"
POLL_INTERVAL=600  # 10 minutes

# GPU utilization and memory thresholds (percent)
GPU_UTIL_THRESHOLD=10   # consider free if utilization < 10%
GPU_MEM_THRESHOLD=5     # consider free if memory usage < 5%

CONFIGS=(
    "rodent_run_gap/overlap_sweep/bino_overlap_80"
    "rodent_run_gap/overlap_sweep/bino_overlap_57"
    "rodent_run_gap/overlap_sweep/bino_overlap_50"
    "rodent_run_gap/overlap_sweep/bino_overlap_40"
    "rodent_run_gap/overlap_sweep/bino_overlap_20"
    "rodent_run_gap/overlap_sweep/bino_overlap_0"
)

gpu_is_free() {
    # Query GPU 0 utilization and memory usage
    local util mem_used mem_total mem_pct
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ')
    mem_pct=$(( mem_used * 100 / mem_total ))

    echo "$(date '+%Y-%m-%d %H:%M:%S') | GPU util: ${util}%, mem: ${mem_used}/${mem_total} MiB (${mem_pct}%)"

    if [ "$util" -lt "$GPU_UTIL_THRESHOLD" ] && [ "$mem_pct" -lt "$GPU_MEM_THRESHOLD" ]; then
        return 0  # free
    else
        return 1  # busy
    fi
}

echo "=== Overlap Sweep (1B timesteps) ==="
echo "Polling GPU every ${POLL_INTERVAL}s until free..."
echo ""

# Wait for GPU to be free
while ! gpu_is_free; do
    echo "  GPU busy, waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

echo ""
echo "GPU is free! Starting sweep..."
echo ""

source "$VENV"
cd "$WORKDIR"

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg")
    # Random seed per run (from /dev/urandom, 0–999999)
    SEED=$(od -An -tu4 -N4 /dev/urandom | tr -d ' ' | head -c6)
    echo "============================================================"
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting: $name (1B timesteps, seed=$SEED)"
    echo "============================================================"

    python -m vnl_playground.train_highlvl \
        --config-name="$cfg" \
        train_setup.train_config.num_timesteps=1_000_000_000 \
        train_setup.train_config.seed="$SEED"

    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Finished: $name"
    echo ""
done

echo "============================================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') | All sweep runs complete!"
echo "============================================================"
