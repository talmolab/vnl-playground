#!/usr/bin/env bash
# run_with_autoresume.sh — Auto-restart training on crash/OOM.
#
# The Python training script handles its own run state discovery and wandb
# resume (via vnl_playground.run_state).  This wrapper is just a retry loop:
# if the training process crashes (non-zero exit), wait and re-launch.
# On restart, the Python code auto-discovers the previous run state file
# and resumes from the latest checkpoint.
#
# Usage:
#   ./vnl_playground/run_with_autoresume.sh [--fresh] [--config-name CONFIG] [HYDRA_OVERRIDES...]
#
# Options:
#   --fresh    Remove existing run state files so training starts from scratch.
#
# Examples:
#   ./vnl_playground/run_with_autoresume.sh --config-name=rodent_run_gap/vision_task_obs_transfer
#   ./vnl_playground/run_with_autoresume.sh --fresh --config-name=rodent_run_gap/vision_task_obs_transfer
#
# Set MAX_RETRIES to limit restart attempts (default: 100).
# Set SLEEP_BETWEEN_RETRIES to wait between restarts (default: 10 seconds).

set -uo pipefail

MAX_RETRIES="${MAX_RETRIES:-100}"
SLEEP_BETWEEN="${SLEEP_BETWEEN_RETRIES:-10}"

# Parse --fresh flag and collect remaining arguments
FRESH=0
EXTRA_ARGS=()
for _arg in "$@"; do
    if [[ "$_arg" == "--fresh" ]]; then
        FRESH=1
    else
        EXTRA_ARGS+=("$_arg")
    fi
done

# --- Auto-detect MODEL_PATH from config YAML ---
if [[ -z "${MODEL_PATH:-}" ]]; then
    _CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)/config"
    _CONFIG_NAME=""
    _next_is_config=""
    for _arg in "${EXTRA_ARGS[@]}"; do
        case "$_arg" in
            --config-name=*) _CONFIG_NAME="${_arg#--config-name=}" ;;
            --config-name)   _next_is_config=1 ;;
            *)
                if [[ "${_next_is_config}" == "1" ]]; then
                    _CONFIG_NAME="$_arg"
                    _next_is_config=""
                fi
                ;;
        esac
    done
    _CONFIG_NAME="${_CONFIG_NAME:-rodent_run_gap/vision_task_obs_transfer}"
    _CONFIG_FILE="${_CONFIG_DIR}/${_CONFIG_NAME}.yaml"
    if [[ -f "$_CONFIG_FILE" ]]; then
        MODEL_PATH=$(grep -Po '^\s*model_path:\s*\K\S+' "$_CONFIG_FILE" | head -1) || true
    fi
    MODEL_PATH="${MODEL_PATH:-highlvl_checkpoints}"
fi

# Derive a unique tag for log files
_HOST="$(hostname -s 2>/dev/null || echo local)"
if [[ -n "${JOB_TAG:-}" ]]; then
    _TAG="${_HOST}_${JOB_TAG}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _TAG="${_HOST}_gpu${CUDA_VISIBLE_DEVICES//,/-}"
else
    _TAG="${_HOST}_default"
fi

echo "=== Auto-Resume Training Wrapper ==="
echo "Job tag: ${_TAG}"
echo "Model path: ${MODEL_PATH}"
echo "Max retries: ${MAX_RETRIES}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

# ---- Handle --fresh: remove run state files matching this host ----
if [[ "$FRESH" -eq 1 ]]; then
    echo "--fresh requested. Removing run_state files for this host..."
    find "${MODEL_PATH}" -maxdepth 1 -name "run_state_${_HOST}_*.json" -delete 2>/dev/null || true
fi

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "=========================================="
    echo "  Attempt ${attempt}/${MAX_RETRIES}"
    echo "  $(date)"
    echo "=========================================="

    CMD=(python -m vnl_playground.train_highlvl "${EXTRA_ARGS[@]}")
    echo "Running: ${CMD[*]}"
    echo ""

    ATTEMPT_LOG="training_attempt_${attempt}_${_TAG}.log"

    set +e
    "${CMD[@]}" 2>&1 | tee "$ATTEMPT_LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "=== Training completed successfully! ==="
        exit 0
    fi

    echo ""
    echo "=== Training exited with code ${EXIT_CODE} ==="
    echo "Python run_state will auto-discover checkpoint on restart."

    if [[ $attempt -lt $MAX_RETRIES ]]; then
        echo "Waiting ${SLEEP_BETWEEN}s before retry..."
        sleep "$SLEEP_BETWEEN"
    fi

    set -e
done

echo ""
echo "=== Max retries (${MAX_RETRIES}) exhausted. Giving up. ==="
exit 1
