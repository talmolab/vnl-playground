#!/usr/bin/env bash
# run_flat_arena_validation.sh
#
# Runs 4 no-vision flat-arena go-to-target experiments sequentially
# to validate reward design before adding vision/gap complexity.
# Each runs 100M steps (~fast with MLP, no vision rendering).
#
# Experiments:
#   1. Speed only        - directional velocity toward target (RunGap pattern)
#   2. Proximity + speed - adds positional pull for reorientation
#   3. Proximity only    - can proximity alone drive navigation?
#   4. Waypoint loop     - 4-point rectangle, tests direction changes
#
# Usage (in screen session):
#   bash vnl_playground/run_flat_arena_validation.sh 2>&1 | tee flat_validation.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Flat Arena Go-To-Target Validation Suite"
echo "  No vision, MLP only, 100M steps each"
echo "  $(date)"
echo "============================================"
echo ""

# --- Experiment 1: Directional speed only ---
echo ">>> [1/4] Speed only (RunGap pattern)"
JOB_TAG=speed_only bash "${SCRIPT_DIR}/run_with_autoresume.sh" \
    --config-name=go_to_target/flat_speed_only
echo ">>> [1/4] Complete."
echo ""

# --- Experiment 2: Proximity + speed ---
echo ">>> [2/4] Proximity + directional speed"
JOB_TAG=proximity_speed bash "${SCRIPT_DIR}/run_with_autoresume.sh" \
    --config-name=go_to_target/flat_proximity_speed
echo ">>> [2/4] Complete."
echo ""

# --- Experiment 3: Proximity only ---
echo ">>> [3/4] Proximity only"
JOB_TAG=proximity_only bash "${SCRIPT_DIR}/run_with_autoresume.sh" \
    --config-name=go_to_target/flat_proximity_only
echo ">>> [3/4] Complete."
echo ""

# --- Experiment 4: Waypoint loop (proximity + speed) ---
echo ">>> [4/4] Waypoint loop (4-point rectangle)"
JOB_TAG=waypoint_loop bash "${SCRIPT_DIR}/run_with_autoresume.sh" \
    --config-name=go_to_target/flat_waypoint_loop
echo ">>> [4/4] Complete."
echo ""

echo "============================================"
echo "  All 4 experiments complete! $(date)"
echo "============================================"
