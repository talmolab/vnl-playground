#!/bin/bash
# Training job queue runner with autoresume integration.
#
# Each job is run through run_with_autoresume.sh which automatically restarts
# training from the latest checkpoint on crashes/OOM (up to MAX_RETRIES times).
#
# Usage:
#   tmux new -s train-queue "bash run_queue.sh"
#
# Add jobs (config name + optional hydra overrides, space-separated):
#   echo "rodent_run_gap/my_config" >> job_queue.txt
#   echo "rodent_run_gap/my_config train.num_timesteps=1_000_000 exp_name=foo" >> job_queue.txt
#
# Skip current job (kills autoresume + python, moves to next):
#   touch skip_current
#
# View queue:     cat job_queue.txt
# View completed: cat job_done.txt
#
# Env vars passed to autoresume:
#   MAX_RETRIES=100   (default)
#   SLEEP_BETWEEN_RETRIES=10 (default)

set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
QUEUE_FILE="$DIR/job_queue.txt"
DONE_FILE="$DIR/job_done.txt"
SKIP_FILE="$DIR/skip_current"
AUTORESUME="$DIR/vnl_playground/run_with_autoresume.sh"
POLL_INTERVAL=10

# JAX memory management — prevent preallocation of entire GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# Source venv so autoresume and python work
source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate
cd "$DIR"

echo "[queue] Runner started. Queue: $QUEUE_FILE"
echo "[queue] Autoresume: $AUTORESUME"
echo "[queue] Add jobs:     echo 'config overrides...' >> $QUEUE_FILE"
echo "[queue] Skip current:  touch $SKIP_FILE"
echo ""

if [ ! -x "$AUTORESUME" ] && [ ! -f "$AUTORESUME" ]; then
    echo "[queue] ERROR: autoresume script not found at $AUTORESUME"
    exit 1
fi

# Wait for any existing training to finish
while pgrep -f "vnl_playground.train_highlvl" > /dev/null 2>&1; do
    echo "[queue] Waiting for existing training to finish..."
    sleep 30
done

rm -f "$SKIP_FILE"

while true; do
    JOB=""
    if [ -f "$QUEUE_FILE" ]; then
        JOB=$(grep -m1 '.' "$QUEUE_FILE" 2>/dev/null || true)
    fi

    if [ -z "$JOB" ]; then
        echo "[queue] No jobs. Waiting... (add with: echo 'config overrides...' >> $QUEUE_FILE)"
        sleep $POLL_INTERVAL
        continue
    fi

    echo ""
    echo "============================================"
    echo "[queue] Starting: $JOB"
    echo "[queue] $(date)"
    echo "[queue] Skip with: touch $SKIP_FILE"
    echo "============================================"

    rm -f "$SKIP_FILE"

    # Lines starting with "!" are raw shell commands (no autoresume wrapper).
    # Useful for non-vnl_playground training scripts that have their own resume.
    # Example: !cd /path/to/repo && python -m module.train --config-name=foo
    if [[ "$JOB" == "!"* ]]; then
        RAW_CMD="${JOB#!}"
        echo "[queue] RAW command mode"
        echo "[queue] Command: $RAW_CMD"

        # Simple retry loop (the training script handles its own resume)
        RAW_MAX_RETRIES="${RAW_MAX_RETRIES:-5}"
        RAW_ATTEMPT=0
        RAW_EXIT=1
        setsid --wait bash -c "$RAW_CMD" &
        AUTORESUME_PID=$!
    else
        # Standard vnl_playground mode via autoresume
        CONFIG=$(echo "$JOB" | awk '{print $1}')
        OVERRIDES=$(echo "$JOB" | awk '{$1=""; print $0}' | sed 's/^ //')
        echo "[queue] Config: $CONFIG"
        [ -n "$OVERRIDES" ] && echo "[queue] Overrides: $OVERRIDES"

        # Unique JOB_TAG per queue entry so autoresume keeps separate state files
        SAFE_TAG=$(echo "$JOB" | tr '/ =.,' '_____' | cut -c1-60)
        export JOB_TAG="queue_${SAFE_TAG}"

        # Launch autoresume in its own process group (setsid)
        setsid bash "$AUTORESUME" --fresh --config-name="$CONFIG" $OVERRIDES &
        AUTORESUME_PID=$!
    fi

    # Monitor: check for skip signal or natural completion
    SKIPPED=0
    while kill -0 "$AUTORESUME_PID" 2>/dev/null; do
        if [ -f "$SKIP_FILE" ]; then
            echo "[queue] Skip signal received. Killing process group $AUTORESUME_PID..."
            # Kill the whole process group (negative PID). Try SIGTERM first,
            # then SIGKILL after a short grace period.
            kill -TERM -"$AUTORESUME_PID" 2>/dev/null || true
            sleep 5
            kill -KILL -"$AUTORESUME_PID" 2>/dev/null || true
            wait "$AUTORESUME_PID" 2>/dev/null || true
            rm -f "$SKIP_FILE"
            SKIPPED=1
            break
        fi
        sleep 5
    done

    wait "$AUTORESUME_PID" 2>/dev/null || true
    EXIT_CODE=$?

    # For raw commands: retry on failure (the script handles its own resume)
    if [[ "$JOB" == "!"* && "$SKIPPED" -eq 0 && "$EXIT_CODE" -ne 0 ]]; then
        RAW_ATTEMPT=1
        while [[ "$RAW_ATTEMPT" -lt "$RAW_MAX_RETRIES" ]]; do
            RAW_ATTEMPT=$((RAW_ATTEMPT + 1))
            echo "[queue] RAW retry ${RAW_ATTEMPT}/${RAW_MAX_RETRIES} (exit=$EXIT_CODE)"
            sleep 10

            # Check for skip before retrying
            if [ -f "$SKIP_FILE" ]; then
                rm -f "$SKIP_FILE"
                SKIPPED=1
                break
            fi

            setsid --wait bash -c "$RAW_CMD" &
            AUTORESUME_PID=$!

            while kill -0 "$AUTORESUME_PID" 2>/dev/null; do
                if [ -f "$SKIP_FILE" ]; then
                    kill -TERM -"$AUTORESUME_PID" 2>/dev/null || true
                    sleep 5
                    kill -KILL -"$AUTORESUME_PID" 2>/dev/null || true
                    wait "$AUTORESUME_PID" 2>/dev/null || true
                    rm -f "$SKIP_FILE"
                    SKIPPED=1
                    break
                fi
                sleep 5
            done

            wait "$AUTORESUME_PID" 2>/dev/null || true
            EXIT_CODE=$?
            [[ "$SKIPPED" -eq 1 || "$EXIT_CODE" -eq 0 ]] && break
        done
    fi

    # Remove first line of queue (this job)
    sed -i '1d' "$QUEUE_FILE"

    if [ "$SKIPPED" -eq 1 ]; then
        echo "[queue] SKIPPED: $JOB"
        echo "$(date) | SKIPPED | $JOB" >> "$DONE_FILE"
    elif [ "$EXIT_CODE" -eq 0 ]; then
        echo "[queue] Completed: $JOB"
        echo "$(date) | exit=0 | $JOB" >> "$DONE_FILE"
    else
        echo "[queue] FAILED (exit=$EXIT_CODE): $JOB"
        echo "$(date) | exit=$EXIT_CODE | $JOB" >> "$DONE_FILE"
    fi

    # Brief pause to let GPU/resources settle
    sleep 5
done
