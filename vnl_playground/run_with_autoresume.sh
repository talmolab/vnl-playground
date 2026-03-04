#!/usr/bin/env bash
# run_with_autoresume.sh — Auto-resume training on kill/crash/OOM.
#
# Usage:
#   ./vnl_playground/run_with_autoresume.sh [--config-name CONFIG] [HYDRA_OVERRIDES...]
#
# Examples:
#   ./vnl_playground/run_with_autoresume.sh --config-name=rodent_run_gap/vision_task_obs_transfer
#   ./vnl_playground/run_with_autoresume.sh --config-name=rodent_run_gap/vision_task_obs_transfer \
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

# Collect all arguments to forward to the training script
EXTRA_ARGS=("$@")

# --- Auto-detect MODEL_PATH from config YAML ---
# Extract --config-name value from args (or use the Hydra default) to locate
# the YAML file, then read logging_config.model_path.  Falls back to
# "highlvl_checkpoints" if the config cannot be found or parsed.
# Can always be overridden with MODEL_PATH env var.
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
    # Hydra default when no --config-name is provided
    _CONFIG_NAME="${_CONFIG_NAME:-rodent_run_gap/vision_task_obs_transfer}"

    _CONFIG_FILE="${_CONFIG_DIR}/${_CONFIG_NAME}.yaml"
    if [[ -f "$_CONFIG_FILE" ]]; then
        MODEL_PATH=$(grep -Po '^\s*model_path:\s*\K\S+' "$_CONFIG_FILE" | head -1) || true
    fi
    MODEL_PATH="${MODEL_PATH:-highlvl_checkpoints}"
fi

# Derive a job tag for unique state/log files when running multiple jobs in the same dir.
# Priority: explicit JOB_TAG env var > CUDA_VISIBLE_DEVICES > "default"
# Hostname is always prepended to avoid conflicts on shared NAS mounts.
_HOST="$(hostname -s 2>/dev/null || echo local)"
if [[ -n "${JOB_TAG:-}" ]]; then
    _TAG="${_HOST}_${JOB_TAG}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Replace commas with dashes so "0,1" becomes "0-1" (safe for filenames)
    _TAG="${_HOST}_gpu${CUDA_VISIBLE_DEVICES//,/-}"
else
    _TAG="${_HOST}_default"
fi

# State file to persist run_id across restarts (unique per job tag)
STATE_FILE=".autoresume_state_${_TAG}"

echo "=== Auto-Resume Training Wrapper ==="
echo "Job tag: ${_TAG}"
echo "Model path: ${MODEL_PATH}"
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

_MONITOR_PID=""
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

    ATTEMPT_LOG="training_attempt_${attempt}_${_TAG}.log"

    # Background monitor: eagerly write state file as soon as run_id appears in log
    if [[ -z "$RESUME_RUN_ID" ]]; then
        (
            for _ in $(seq 1 60); do [[ -f "$ATTEMPT_LOG" ]] && break; sleep 0.5; done
            if [[ -f "$ATTEMPT_LOG" ]]; then
                tail -f "$ATTEMPT_LOG" 2>/dev/null | while IFS= read -r line; do
                    if [[ "$line" == *"NEW run_id: "* ]]; then
                        _rid="${line##*NEW run_id: }"
                        _rid="${_rid%% *}"
                        _rid="${_rid%%$'\r'}"
                        echo "$_rid" > "$STATE_FILE"
                        break
                    fi
                done
            fi
        ) &
        _MONITOR_PID=$!
    fi

    # Run training, capture exit code
    set +e
    "${CMD[@]}" 2>&1 | tee "$ATTEMPT_LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    # Keep errexit disabled through the entire retry-handling section.
    # Previously, set -e was re-enabled here, which caused silent script
    # exits when kill (on an already-dead monitor) returned non-zero,
    # preventing the retry loop from ever executing.

    if [[ -n "${_MONITOR_PID:-}" ]]; then
        kill "$_MONITOR_PID" 2>/dev/null || true
        wait "$_MONITOR_PID" 2>/dev/null || true
        _MONITOR_PID=""
    fi

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "=== Training completed successfully! ==="
        rm -f "$STATE_FILE"
        exit 0
    fi

    echo ""
    echo "=== Training exited with code ${EXIT_CODE} ==="

    if [[ -z "$RESUME_RUN_ID" ]]; then
        # Check if background monitor already wrote state file
        if [[ -f "$STATE_FILE" ]]; then
            RESUME_RUN_ID=$(cat "$STATE_FILE")
            if [[ -d "${MODEL_PATH}/${RESUME_RUN_ID}" ]]; then
                echo "Captured run_id=${RESUME_RUN_ID} for resume (from eager monitor)"
            else
                echo "WARNING: Eager state references ${RESUME_RUN_ID} but dir missing."
                RESUME_RUN_ID=""
                rm -f "$STATE_FILE"
            fi
        fi
    fi

    # Fallback: grep the per-instance training log
    if [[ -z "$RESUME_RUN_ID" ]]; then
        DETECTED_ID=$(grep -oP 'NEW run_id: \K\S+' "$ATTEMPT_LOG" 2>/dev/null | head -1) || true
        if [[ -n "$DETECTED_ID" && -d "${MODEL_PATH}/${DETECTED_ID}" ]]; then
            RESUME_RUN_ID="$DETECTED_ID"
            echo "$RESUME_RUN_ID" > "$STATE_FILE"
            echo "Captured run_id=${RESUME_RUN_ID} for resume (from training log)"
        elif [[ -n "$DETECTED_ID" ]]; then
            echo "WARNING: run_id=${DETECTED_ID} found in log but checkpoint dir missing (OOM before first save?)."
            echo "Starting fresh on next attempt."
            RESUME_RUN_ID=""
            rm -f "$STATE_FILE"
        else
            echo "ERROR: Could not detect run_id from ${ATTEMPT_LOG}"
            echo "Cannot resume. Exiting."
            exit 1
        fi
    fi

    # Check if there are actually checkpoints to resume from
    CKPT_FILES=( "${MODEL_PATH}/${RESUME_RUN_ID}"/PPONetwork_* )
    if [[ -e "${CKPT_FILES[0]}" ]]; then
        CKPT_COUNT=${#CKPT_FILES[@]}
    else
        CKPT_COUNT=0
    fi
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

    set -e
done

echo ""
echo "=== Max retries (${MAX_RETRIES}) exhausted. Giving up. ==="
exit 1
