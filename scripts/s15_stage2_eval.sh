#!/bin/bash
# Stage 2 of s15-ms: re-evaluate 8 frontier checkpoints under the new EMG
# metrics at --emg-norm-percentile 98 (legacy) and 100 (new default).
# Outputs per-(checkpoint, muscle, norm_pct) rows to
# plots/2026-04-23-s15-stage2/eval_matrix.csv.
#
# Total: 8 ckpts × 2 percentiles = 16 eval-replay runs, ~20 min each.
# Wall-clock ~3 h on one GPU.
#
# Spec: docs/superpowers/specs/2026-04-23-s15-ms-design.md
# Plan: docs/superpowers/plans/2026-04-23-s15-ms-implementation.md (Task 10)

set -o pipefail
cd /root/vast/eric/vnl-playground
source /root/vast/eric/track-mjx/.venv/bin/activate

OUT_DIR="plots/2026-04-23-s15-stage2"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/eval_matrix.csv"

# --- 8 CHECKPOINTS (resolved 2026-04-23) ---
CKPTS=(
    "checkpoints/s13-ms-armM-anchorA-fs1p1-s2-20260421-043700"
    "checkpoints/s13-ms-armM-anchorA-fs1p4-20260421-042506"
    "checkpoints/s13-ms-armM-anchorC-fs1p3-20260421-042708"
    "checkpoints/s14-ms-anchorA-C7-t1p4b1p4-s1-20260422-094925"
    "checkpoints/s14-ms-anchorA-C4-t1p1b1p1-s1-20260422-044930"
    "checkpoints/s12-ms-armA-d1em6-fs1p0-cc0p025-cdc0p025-20260420-090043"
    "checkpoints/s11-ms-R3-fs1p0-d5em7-cc0p05-cdc0p1-s2-20260419-172437"
    "checkpoints/s10-bridge-fs03-C-s1-20260415-070324"
)

# CSV header (one row per checkpoint × muscle × norm_pct)
echo "checkpoint,norm_pct,muscle,mean_corr,mean_mae,trial_corr_mean,trial_corr_median,trial_mae,lagged_corr_max,phase_lag_steps,phase_lag_ms,lagged_corr_fwhm_steps,lagged_corr_edge_saturated,per_trial_lagged_corr_mean,per_trial_lagged_corr_median,per_trial_phase_lag_mean_ms,per_trial_phase_lag_std_ms" > "${CSV}"

TOTAL=$(( ${#CKPTS[@]} * 2 ))
I=0
for CKPT in "${CKPTS[@]}"; do
    if [ ! -d "${CKPT}" ]; then
        echo "[SKIP] ${CKPT} does not exist — skipping"
        continue
    fi
    CKPT_BASE=$(basename "${CKPT}")
    for PCT in 98 100; do
        I=$(( I + 1 ))
        LOG="${OUT_DIR}/eval_${CKPT_BASE}_p${PCT}.log"
        JSON="${OUT_DIR}/eval_${CKPT_BASE}_p${PCT}.json"
        echo "================================================================"
        echo "[Stage2 ${I}/${TOTAL}] ${CKPT_BASE} @ p${PCT}"
        echo "================================================================"

        if python scripts/emg_comparison.py \
                --checkpoint "${CKPT}" \
                --emg-norm-percentile "${PCT}" \
                --output-json "${JSON}" \
                2>&1 | tee "${LOG}"; then
            echo "[OK] ${CKPT_BASE} @ p${PCT}"
        else
            echo "[FAIL] ${CKPT_BASE} @ p${PCT} — see ${LOG}"
            continue
        fi

        # Parse the JSON and append one CSV row per muscle.
        python - <<PY >> "${CSV}"
import json, os
try:
    J = json.load(open("${JSON}"))
except Exception as e:
    print(f"# failed to parse ${JSON}: {e}", file=__import__('sys').stderr)
    raise SystemExit(0)
for muscle, m in J.get("metrics_by_muscle", {}).items():
    row = [
        "${CKPT_BASE}", "${PCT}", muscle,
        m.get("mean_corr"), m.get("mean_mae"),
        m.get("trial_corr_mean"), m.get("trial_corr_median"), m.get("trial_mae"),
        m.get("lagged_corr_max"), m.get("phase_lag_steps"), m.get("phase_lag_ms"),
        m.get("lagged_corr_fwhm_steps"), m.get("lagged_corr_edge_saturated"),
        m.get("per_trial_lagged_corr_mean"), m.get("per_trial_lagged_corr_median"),
        m.get("per_trial_phase_lag_mean_ms"), m.get("per_trial_phase_lag_std_ms"),
    ]
    print(",".join("" if v is None else f"{v}" for v in row))
PY
    done
done

echo ""
echo "================================================================"
echo "=== Stage 2 complete. CSV at ${CSV} ==="
echo "    Rows: $(tail -n +2 "${CSV}" | wc -l)"
echo "    JSONs: $(ls ${OUT_DIR}/*.json 2>/dev/null | wc -l)"
echo "================================================================"
