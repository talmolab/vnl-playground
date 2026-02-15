#!/usr/bin/env bash
# run_with_autoresume.sh — Auto-resume training on kill/crash/OOM.
#
# Usage:
#   ./vnl_playground/run_with_autoresume.sh [--config-name CONFIG] [HYDRA_OVERRIDES...]
#
# Examples:
#   ./vnl_playground/run_with_autoresume.sh --config-name=run_gap_vision_task_obs_transfer
#   ./vnl_playground/run_with_autoresume.sh --config-name=run_gap_vision_task_obs_transfer \
#       train_setup.train_config.num_envs=512
#
# Multi-GPU usage (run two jobs in the same directory):
#   CUDA_VISIBLE_DEVICES=0 ./vnl_playground/run_with_autoresume.sh --config-name=...
#   CUDA_VISIBLE_DEVICES=1 ./vnl_playground/run_with_autoresume.sh --config-name=...
#
# Each instance gets its own state file (.autoresume_state_gpu0, .autoresume_state_gpu1)
# and log files (training_attempt_N_gpu0.log, training_attempt_N_gpu1.log).
#
# You can also set JOB_TAG explicitly to override the auto-derived tag:
#   JOB_TAG=experiment_a ./vnl_playground/run_with_autoresume.sh --config-name=...
#
# The script:
#   1. Runs training normally on first launch
#   2. Captures the run_id from the checkpoint directory
#   3. On crash/kill (exit code != 0), restarts with resume_run_id=<run_id>
#   4. Loops until training completes successfully (exit code 0) or max retries
#
# Set MAX_RETRIES to limit restart attempts (default: 50).
# Set SLEEP_BETWEEN_RETRIES to wait between restarts (default: 10 seconds).

set -euo pipefail

MAX_RETRIES="${MAX_RETRIES:-50}"
SLEEP_BETWEEN="${SLEEP_BETWEEN_RETRIES:-10}"
MODEL_PATH="highlvl_checkpoints"  # Must match cfg.logging_config.model_path

# Collect all arguments to forward to the training script
EXTRA_ARGS=("$@")

# Derive a job tag for unique state/log files when running multiple jobs in the same dir.
# Priority: explicit JOB_TAG env var > CUDA_VISIBLE_DEVICES > "default"
if [[ -n "${JOB_TAG:-}" ]]; then
    _TAG="$JOB_TAG"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Replace commas with dashes so "0,1" becomes "0-1" (safe for filenames)
    _TAG="gpu${CUDA_VISIBLE_DEVICES//,/-}"
else
    _TAG="default"
fi

# State file to persist run_id across restarts (unique per job tag)
STATE_FILE=".autoresume_state_${_TAG}"

echo "=== Auto-Resume Training Wrapper ==="
echo "Job tag: ${_TAG}"
echo "State file: ${STATE_FILE}"
echo "Max retries: ${MAX_RETRIES}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

# ---- Check for existing state from a previous wrapper run ----
RESUME_RUN_ID=""
if [[ -f "$STATE_FILE" ]]; then
    RESUME_RUN_ID=$(cat "$STATE_FILE")
    if [[ -d "${MODEL_PATH}/${RESUME_RUN_ID}" ]]; then
        echo "Found previous run state: run_id=${RESUME_RUN_ID}"
    else
        echo "WARNING: State file references ${RESUME_RUN_ID} but directory missing. Starting fresh."
        RESUME_RUN_ID=""
        rm -f "$STATE_FILE"
    fi
fi

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "=========================================="
    echo "  Attempt ${attempt}/${MAX_RETRIES}"
    echo "  $(date)"
    echo "=========================================="

    # Build command
    CMD=(python -m vnl_playground.train_highlvl "${EXTRA_ARGS[@]}")

    if [[ -n "$RESUME_RUN_ID" ]]; then
        echo "RESUMING from run_id=${RESUME_RUN_ID}"
        CMD+=("+train_setup.resume_run_id='${RESUME_RUN_ID}'")
    else
        echo "Starting FRESH run"
    fi

    echo "Running: ${CMD[*]}"
    echo ""

    # Run training, capture exit code
    set +e
    "${CMD[@]}" 2>&1 | tee "training_attempt_${attempt}_${_TAG}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "=== Training completed successfully! ==="
        rm -f "$STATE_FILE"
        exit 0
    fi

    echo ""
    echo "=== Training exited with code ${EXIT_CODE} ==="

    # If this was the first run, detect the run_id from the checkpoint directory
    if [[ -z "$RESUME_RUN_ID" ]]; then
        # Find the most recently created directory in MODEL_PATH
        LATEST_DIR=$(ls -td "${MODEL_PATH}"/*/ 2>/dev/null | head -1)
        if [[ -n "$LATEST_DIR" ]]; then
            RESUME_RUN_ID=$(basename "$LATEST_DIR")
            echo "$RESUME_RUN_ID" > "$STATE_FILE"
            echo "Captured run_id=${RESUME_RUN_ID} for resume"
        else
            echo "ERROR: Could not find checkpoint directory in ${MODEL_PATH}"
            echo "Cannot resume. Exiting."
            exit 1
        fi
    fi

    # Check if there are actually checkpoints to resume from
    CKPT_COUNT=$(ls -d "${MODEL_PATH}/${RESUME_RUN_ID}"/PPONetwork_* 2>/dev/null | wc -l)
    if [[ "$CKPT_COUNT" -eq 0 ]]; then
        echo "WARNING: No PPONetwork checkpoints found. Starting fresh on next attempt."
        RESUME_RUN_ID=""
        rm -f "$STATE_FILE"
    else
        echo "Found ${CKPT_COUNT} checkpoint(s) to resume from"
    fi

    if [[ $attempt -lt $MAX_RETRIES ]]; then
        echo "Waiting ${SLEEP_BETWEEN}s before retry..."
        sleep "$SLEEP_BETWEEN"
    fi
done

echo ""
echo "=== Max retries (${MAX_RETRIES}) exhausted. Giving up. ==="
exit 1
