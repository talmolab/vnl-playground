#!/bin/bash
# Benchmark num_envs x naconmax combos on RTX 5090.
# 5M timesteps, no eval, wandb off.

source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate
cd /home/talmolab/Desktop/SalkResearch/vnl-playground

export WANDB_MODE=disabled

RESULTS_FILE="/tmp/gpu_bench_results.txt"
printf "%-10s %-10s %-10s %-12s\n" "num_envs" "naconmax" "SPS" "GPU_mem_MB" > "$RESULTS_FILE"

CONFIGS=(
    "1024 8192"
    "1024 6144"
    "2048 6144"
    "2048 4096"
    "1536 6144"
    "1536 4096"
    "2048 8192"
)

for CFG in "${CONFIGS[@]}"; do
    read -r NUM_ENVS NACON <<< "$CFG"
    LABEL="${NUM_ENVS}_${NACON}"
    echo "===== Testing num_envs=$NUM_ENVS naconmax=$NACON ====="
    LOG="/tmp/bench_${LABEL}.log"
    > "$LOG"

    timeout 480 python -m vnl_playground.train_highlvl \
        --config-name=rodent_run_gap/binocular_progress_gap_reward \
        train_setup.train_config.num_envs=$NUM_ENVS \
        train_setup.train_config.batch_size=$NUM_ENVS \
        train_setup.train_config.num_timesteps=5_000_000 \
        train_setup.eval_every=999_999_999 \
        env_config.env_args.naconmax=$NACON \
        2>&1 | tee "$LOG"
    EXIT_CODE=$?

    if grep -q "training/sps" "$LOG" 2>/dev/null; then
        SPS=$(grep "training/sps" "$LOG" | tail -1 | grep -oP "training/sps.: np.float64\(\K[0-9.]+")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        printf "%-10s %-10s %-10s %-12s\n" "$NUM_ENVS" "$NACON" "$SPS" "$GPU_MEM" >> "$RESULTS_FILE"
        echo "  -> SPS=$SPS, GPU_MEM=${GPU_MEM}MB"
    elif grep -qi "out of memory\|oom\|RESOURCE_EXHAUSTED" "$LOG" 2>/dev/null; then
        printf "%-10s %-10s %-10s %-12s\n" "$NUM_ENVS" "$NACON" "OOM" "N/A" >> "$RESULTS_FILE"
        echo "  -> OOM"
    else
        printf "%-10s %-10s %-10s %-12s\n" "$NUM_ENVS" "$NACON" "FAIL($EXIT_CODE)" "N/A" >> "$RESULTS_FILE"
        echo "  -> FAILED (exit=$EXIT_CODE)"
    fi

    sleep 5
done

echo ""
echo "===== RESULTS ====="
# Include the baseline we already measured
echo "(baseline: 1024 / 12288 -> 28588 SPS, 15873 MB)"
cat "$RESULTS_FILE"
