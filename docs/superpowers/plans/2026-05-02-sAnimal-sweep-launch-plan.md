# sAnimal Sweep Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 6 launch-ready sweep scripts and an `SANIMAL_LAUNCH.md` for the per-animal hyperparameter sweep specified in `2026-05-02-sAnimal-per-animal-hyperparameter-sweep-design.md`. 45 runs total (9 cells × 5 animals), single seed, balanced 5/8/8/8/8/8 across 6 GPUs.

**Architecture:** Pure shell scripts following the s17/s19 pattern: each script sources `BASE_ARGS`, defines a `run_cell()` function with budget-aware skip, and ends with a status report. No trainer changes (s17 already shipped `--train-animals`, `--emg-animals`, `--emg-norm-method`, `--reference-data-path`).

**Tech Stack:** Bash, `train_mouse_janelia_sigmoid_moving_shoulder.py` (pre-existing), wandb (for run grouping).

**Spec:** `docs/superpowers/specs/2026-05-02-sAnimal-per-animal-hyperparameter-sweep-design.md`

---

## Cell parameters (referenced by every script)

`shoulder_damping=6e-7` and `seed=0` for all cells.

| Cell | --force-scale | --joint-damping | --control-cost | --control-diff-cost |
|------|---|---|---|---|
| **C0** | 1.1  | 1.5e-6 | 0.025 | 0.025 |
| **F1** | 1.0  | 1e-6   | 0.0   | 0.0   |
| **F2** | 1.2  | 1e-6   | 0.0   | 0.05  |
| **F3** | 1.0  | 2e-6   | 0.0   | 0.05  |
| **F4** | 1.2  | 2e-6   | 0.0   | 0.0   |
| **F5** | 1.0  | 1e-6   | 0.05  | 0.05  |
| **F6** | 1.2  | 1e-6   | 0.05  | 0.0   |
| **F7** | 1.0  | 2e-6   | 0.05  | 0.0   |
| **F8** | 1.2  | 2e-6   | 0.05  | 0.05  |

Animals: `A36-1, AT006, AT009, AT012, AT013`. Tag pattern: `<animal>-<cell>` (e.g., `A36-1-C0`).

## File Structure

**Created:**
- `sweep_sAnimal_1.sh` — A36-1: C0, F1, F2, F3, F4 (5 cells)
- `sweep_sAnimal_2.sh` — A36-1: F5, F6, F7, F8 + AT006: C0, F1, F2, F3 (8 cells)
- `sweep_sAnimal_3.sh` — AT006: F4, F5, F6, F7, F8 + AT009: C0, F1, F2 (8 cells)
- `sweep_sAnimal_4.sh` — AT009: F3, F4, F5, F6, F7, F8 + AT012: C0, F1 (8 cells)
- `sweep_sAnimal_5.sh` — AT012: F2, F3, F4, F5, F6, F7, F8 + AT013: C0 (8 cells)
- `sweep_sAnimal_6.sh` — AT013: F1, F2, F3, F4, F5, F6, F7, F8 (8 cells)
- `SANIMAL_LAUNCH.md` — pre-launch checklist + per-job nohup commands

**Reused (read-only):**
- `train_mouse_janelia_sigmoid_moving_shoulder.py`
- `vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/`

---

## Task 1: Pre-flight smoke verification

Confirm the trainer accepts the BASE_ARGS template + a single cell's flags before producing 6 scripts that all share the same template.

**Files:** none (operational task; no commits in this task).

- [ ] **Step 1: Verify reference clip dir exists**

```
ls vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/ | wc -l
```
Expected: integer ≥ 270 (s17 spec says 278 clips).

- [ ] **Step 2: Verify no concurrent sweep is using GPUs**

```
pgrep -af 'sweep_s1[789]_ms_[0-9]\.sh'
nvidia-smi --query-gpu=index,memory.used --format=csv
```
Expected: pgrep prints empty (or only stale shell PIDs); nvidia-smi memory.used per GPU < 1000 MiB. **If GPUs are busy, STOP and wait — sAnimal is not designed to share GPUs with another sweep.**

- [ ] **Step 3: Run the 5M-timestep preflight (10 min)**

```
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 timeout 900 python train_mouse_janelia_sigmoid_moving_shoulder.py \
  --reference-data-path /root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals \
  --train-animals A36-1 \
  --emg-animals A36-1 AT006 AT009 AT012 AT013 \
  --emg-norm-method z_baseline_x2 \
  --ctrl-dt 0.0025 --sim-dt 0.00125 --episode-length 100 \
  --qvel-init zeros --joint-armature 4e-10 \
  --joints-weight 5.0 --joints-vel-weight 0.5 \
  --wrist-pos-weight 0.1 --bodies-pos-weight 0.1 \
  --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
  --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
  --num-timesteps 5000000 --num-evals 1 \
  --no-wandb --tag sAnimal-preflight --run-name sAnimal-preflight-$(date +%s) \
  2>&1 | tail -40
```

Expected: training completes within 10 min; final lines include `eval/episode_reward` for the single eval and one `eval/emg_*` block per animal. **No traceback.** If a flag is rejected (e.g., `--train-animals: error: unrecognized arguments`), the trainer hasn't been updated for s17 yet — DO NOT proceed; flag this to the user.

- [ ] **Step 4: Manual confirmation gate**

Confirm out loud (or in the task tracker): "preflight passed, all 5 animals' EMG eval logged." Do not proceed to Task 2 until this is confirmed.

---

## Task 2: Write `sweep_sAnimal_1.sh` (A36-1: 5 cells)

**Files:**
- Create: `sweep_sAnimal_1.sh`

This script is the template every other script follows. The structure is identical to `sweep_s17_ms_1.sh` (BASE_ARGS array, run_cell function with budget skip, list of run_cell calls, status report at the end). Subsequent tasks reuse this template and only change the cell list.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# SWEEP sAnimal part 1/6 — Job1 GPU0 (lightest script, runs first as canary)
# 5 cells: A36-1 × {C0, F1, F2, F3, F4}.
# Spec: docs/superpowers/specs/2026-05-02-sAnimal-per-animal-hyperparameter-sweep-design.md
set -o pipefail

cd /root/vast/eric/vnl-playground
eval "$(conda shell.bash hook)"
conda activate track_mjx

WANDB_GROUP="sAnimal-part1"
BUDGET_SECONDS=$(( ${BUDGET_HOURS:-30} * 3600 ))
ESTIMATED_RUN_SECONDS=${ESTIMATED_RUN_SECONDS:-14400}

REF_DATA=/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals

BASE_ARGS=(
    --reference-data-path "${REF_DATA}"
    --emg-animals A36-1 AT006 AT009 AT012 AT013
    --emg-norm-method z_baseline_x2
    --ctrl-dt 0.0025
    --sim-dt 0.00125
    --episode-length 100
    --qvel-init zeros
    --joint-armature 4e-10
    --joints-weight 5.0
    --joints-vel-weight 0.5
    --wrist-pos-weight 0.1
    --bodies-pos-weight 0.1
    --num-timesteps 800000000
    --num-evals 8
    --wandb-group "${WANDB_GROUP}"
)

START_TIME=$(date +%s)
CRASHED=()
OK=()
SKIPPED=()
TOTAL=5
CELL=0

run_cell() {
    local TAG="$1"; shift
    local NOW=$(date +%s)
    local REMAINING=$(( BUDGET_SECONDS - (NOW - START_TIME) ))
    CELL=$((CELL + 1))
    if (( REMAINING < ESTIMATED_RUN_SECONDS )); then
        echo "----------------------------------------------------------------"
        echo "[sAnimal-1 ${CELL}/${TOTAL}] ${TAG} — SKIPPED (budget ${REMAINING}s < ${ESTIMATED_RUN_SECONDS}s est)"
        echo "----------------------------------------------------------------"
        SKIPPED+=("${TAG}")
        return
    fi
    local RUN_NAME="sAnimal-${TAG}-$(date +%Y%m%d-%H%M%S)"
    local LOG="/tmp/sweep_${RUN_NAME}.log"
    echo "----------------------------------------------------------------"
    echo "[sAnimal-1 ${CELL}/${TOTAL}] ${RUN_NAME} (${REMAINING}s remaining)"
    echo "  $@"
    echo "----------------------------------------------------------------"
    if python train_mouse_janelia_sigmoid_moving_shoulder.py "${BASE_ARGS[@]}" \
        --tag "${TAG}" --run-name "${RUN_NAME}" "$@" 2>&1 | tee "${LOG}"; then
        OK+=("${RUN_NAME}"); echo "[OK] ${RUN_NAME}"
    else
        CRASHED+=("${RUN_NAME}"); echo "[CRASHED] ${RUN_NAME} (see ${LOG})"
    fi
    echo
}

# A36-1 × C0 (center, s16 leader)
run_cell "A36-1-C0" \
    --train-animals A36-1 \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
    --wandb-tags sAnimal A36-1 C0 center

# A36-1 × F1
run_cell "A36-1-F1" \
    --train-animals A36-1 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal A36-1 F1 fractional

# A36-1 × F2
run_cell "A36-1-F2" \
    --train-animals A36-1 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal A36-1 F2 fractional

# A36-1 × F3
run_cell "A36-1-F3" \
    --train-animals A36-1 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal A36-1 F3 fractional

# A36-1 × F4
run_cell "A36-1-F4" \
    --train-animals A36-1 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal A36-1 F4 fractional

echo "================================================================"
echo "=== sAnimal part 1/6 complete ==="
echo "  OK      (${#OK[@]}):";      for R in "${OK[@]}"; do echo "    OK  ${R}"; done
echo "  CRASHED (${#CRASHED[@]}):"; for R in "${CRASHED[@]}"; do echo "    BAD ${R}"; done
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[@]}"
echo "================================================================"
```

- [ ] **Step 2: Bash syntax check**

```
bash -n sweep_sAnimal_1.sh && echo OK
```
Expected: prints `OK`, no syntax errors.

- [ ] **Step 3: Verify cell count via grep**

```
grep -c '^run_cell ' sweep_sAnimal_1.sh
```
Expected: `5`.

- [ ] **Step 4: Verify TOTAL matches grep count**

```
grep '^TOTAL=' sweep_sAnimal_1.sh
```
Expected: `TOTAL=5`.

- [ ] **Step 5: Commit**

```
git add sweep_sAnimal_1.sh
git commit -m "add sAnimal sweep script 1: A36-1 C0+F1-F4 (5 cells)"
```

---

## Task 3: Write `sweep_sAnimal_2.sh` (A36-1 F5-F8 + AT006 C0,F1-F3)

**Files:**
- Create: `sweep_sAnimal_2.sh`

Same template as Task 2. Header and `WANDB_GROUP` change to `part2`; `TOTAL=8`; cell list contains 8 entries.

- [ ] **Step 1: Write the script**

The header through the `run_cell()` function definition is identical to `sweep_sAnimal_1.sh` except for these substitutions:
- Line 2 comment: `# SWEEP sAnimal part 2/6 — Job1 GPU1`, body: `# 8 cells: A36-1 × {F5, F6, F7, F8} + AT006 × {C0, F1, F2, F3}.`
- `WANDB_GROUP="sAnimal-part2"`
- `TOTAL=8`
- All `[sAnimal-1 ...]` echo prefixes become `[sAnimal-2 ...]`
- Final banner becomes `=== sAnimal part 2/6 complete ===`

The cell list (replaces the 5 run_cell calls in Task 2's script):
```bash
# A36-1 × F5
run_cell "A36-1-F5" \
    --train-animals A36-1 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal A36-1 F5 fractional

# A36-1 × F6
run_cell "A36-1-F6" \
    --train-animals A36-1 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal A36-1 F6 fractional

# A36-1 × F7
run_cell "A36-1-F7" \
    --train-animals A36-1 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal A36-1 F7 fractional

# A36-1 × F8
run_cell "A36-1-F8" \
    --train-animals A36-1 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal A36-1 F8 fractional

# AT006 × C0
run_cell "AT006-C0" \
    --train-animals AT006 \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
    --wandb-tags sAnimal AT006 C0 center

# AT006 × F1
run_cell "AT006-F1" \
    --train-animals AT006 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT006 F1 fractional

# AT006 × F2
run_cell "AT006-F2" \
    --train-animals AT006 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT006 F2 fractional

# AT006 × F3
run_cell "AT006-F3" \
    --train-animals AT006 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT006 F3 fractional
```

The fastest implementation path: copy `sweep_sAnimal_1.sh`, apply the header substitutions, replace the 5 cell calls with the 8 above. Verify by hand that the total line count is similar to script 1 plus 3 cells × 5 lines.

- [ ] **Step 2: Bash syntax check**

```
bash -n sweep_sAnimal_2.sh && echo OK
```
Expected: `OK`.

- [ ] **Step 3: Verify cell count and TOTAL**

```
[ "$(grep -c '^run_cell ' sweep_sAnimal_2.sh)" = "8" ] && grep '^TOTAL=' sweep_sAnimal_2.sh
```
Expected: prints `TOTAL=8` (and exits 0, meaning the count matched).

- [ ] **Step 4: Verify the WANDB_GROUP and banner labels updated**

```
grep -E 'WANDB_GROUP=|sAnimal-2|sAnimal-1' sweep_sAnimal_2.sh
```
Expected: lines mention `sAnimal-part2` and `sAnimal-2`; **no** lines mention `sAnimal-part1` or `sAnimal-1`.

- [ ] **Step 5: Commit**

```
git add sweep_sAnimal_2.sh
git commit -m "add sAnimal sweep script 2: A36-1 F5-F8 + AT006 C0+F1-F3 (8 cells)"
```

---

## Task 4: Write `sweep_sAnimal_3.sh` (AT006 F4-F8 + AT009 C0,F1,F2)

**Files:**
- Create: `sweep_sAnimal_3.sh`

Same template as Tasks 2/3. Substitutions: `part3`, `[sAnimal-3 …]`, `TOTAL=8`, `=== sAnimal part 3/6 complete ===`. Cell list:

- [ ] **Step 1: Write the script — cell list**

```bash
# AT006 × F4
run_cell "AT006-F4" \
    --train-animals AT006 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT006 F4 fractional

# AT006 × F5
run_cell "AT006-F5" \
    --train-animals AT006 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT006 F5 fractional

# AT006 × F6
run_cell "AT006-F6" \
    --train-animals AT006 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT006 F6 fractional

# AT006 × F7
run_cell "AT006-F7" \
    --train-animals AT006 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT006 F7 fractional

# AT006 × F8
run_cell "AT006-F8" \
    --train-animals AT006 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT006 F8 fractional

# AT009 × C0
run_cell "AT009-C0" \
    --train-animals AT009 \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
    --wandb-tags sAnimal AT009 C0 center

# AT009 × F1
run_cell "AT009-F1" \
    --train-animals AT009 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT009 F1 fractional

# AT009 × F2
run_cell "AT009-F2" \
    --train-animals AT009 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT009 F2 fractional
```

- [ ] **Step 2: Verify**

```
bash -n sweep_sAnimal_3.sh && echo OK
[ "$(grep -c '^run_cell ' sweep_sAnimal_3.sh)" = "8" ] && grep '^TOTAL=' sweep_sAnimal_3.sh
grep -E 'WANDB_GROUP=' sweep_sAnimal_3.sh   # must show sAnimal-part3
```
Expected: `OK`, `TOTAL=8`, `WANDB_GROUP="sAnimal-part3"`.

- [ ] **Step 3: Commit**

```
git add sweep_sAnimal_3.sh
git commit -m "add sAnimal sweep script 3: AT006 F4-F8 + AT009 C0+F1+F2 (8 cells)"
```

---

## Task 5: Write `sweep_sAnimal_4.sh` (AT009 F3-F8 + AT012 C0,F1)

**Files:**
- Create: `sweep_sAnimal_4.sh`

Substitutions: `part4`, `[sAnimal-4 …]`, `TOTAL=8`, `=== sAnimal part 4/6 complete ===`.

- [ ] **Step 1: Cell list**

```bash
# AT009 × F3
run_cell "AT009-F3" \
    --train-animals AT009 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT009 F3 fractional

# AT009 × F4
run_cell "AT009-F4" \
    --train-animals AT009 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT009 F4 fractional

# AT009 × F5
run_cell "AT009-F5" \
    --train-animals AT009 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT009 F5 fractional

# AT009 × F6
run_cell "AT009-F6" \
    --train-animals AT009 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT009 F6 fractional

# AT009 × F7
run_cell "AT009-F7" \
    --train-animals AT009 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT009 F7 fractional

# AT009 × F8
run_cell "AT009-F8" \
    --train-animals AT009 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT009 F8 fractional

# AT012 × C0
run_cell "AT012-C0" \
    --train-animals AT012 \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
    --wandb-tags sAnimal AT012 C0 center

# AT012 × F1
run_cell "AT012-F1" \
    --train-animals AT012 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT012 F1 fractional
```

- [ ] **Step 2: Verify**

```
bash -n sweep_sAnimal_4.sh && echo OK
[ "$(grep -c '^run_cell ' sweep_sAnimal_4.sh)" = "8" ] && grep '^TOTAL=' sweep_sAnimal_4.sh
grep -E 'WANDB_GROUP=' sweep_sAnimal_4.sh
```
Expected: `OK`, `TOTAL=8`, `WANDB_GROUP="sAnimal-part4"`.

- [ ] **Step 3: Commit**

```
git add sweep_sAnimal_4.sh
git commit -m "add sAnimal sweep script 4: AT009 F3-F8 + AT012 C0+F1 (8 cells)"
```

---

## Task 6: Write `sweep_sAnimal_5.sh` (AT012 F2-F8 + AT013 C0)

**Files:**
- Create: `sweep_sAnimal_5.sh`

Substitutions: `part5`, `[sAnimal-5 …]`, `TOTAL=8`, `=== sAnimal part 5/6 complete ===`.

- [ ] **Step 1: Cell list**

```bash
# AT012 × F2
run_cell "AT012-F2" \
    --train-animals AT012 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT012 F2 fractional

# AT012 × F3
run_cell "AT012-F3" \
    --train-animals AT012 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT012 F3 fractional

# AT012 × F4
run_cell "AT012-F4" \
    --train-animals AT012 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT012 F4 fractional

# AT012 × F5
run_cell "AT012-F5" \
    --train-animals AT012 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT012 F5 fractional

# AT012 × F6
run_cell "AT012-F6" \
    --train-animals AT012 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT012 F6 fractional

# AT012 × F7
run_cell "AT012-F7" \
    --train-animals AT012 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT012 F7 fractional

# AT012 × F8
run_cell "AT012-F8" \
    --train-animals AT012 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT012 F8 fractional

# AT013 × C0
run_cell "AT013-C0" \
    --train-animals AT013 \
    --force-scale 1.1 --joint-damping 1.5e-6 --shoulder-damping 6e-7 \
    --control-cost 0.025 --control-diff-cost 0.025 --seed 0 \
    --wandb-tags sAnimal AT013 C0 center
```

- [ ] **Step 2: Verify**

```
bash -n sweep_sAnimal_5.sh && echo OK
[ "$(grep -c '^run_cell ' sweep_sAnimal_5.sh)" = "8" ] && grep '^TOTAL=' sweep_sAnimal_5.sh
grep -E 'WANDB_GROUP=' sweep_sAnimal_5.sh
```
Expected: `OK`, `TOTAL=8`, `WANDB_GROUP="sAnimal-part5"`.

- [ ] **Step 3: Commit**

```
git add sweep_sAnimal_5.sh
git commit -m "add sAnimal sweep script 5: AT012 F2-F8 + AT013 C0 (8 cells)"
```

---

## Task 7: Write `sweep_sAnimal_6.sh` (AT013 F1-F8)

**Files:**
- Create: `sweep_sAnimal_6.sh`

Substitutions: `part6`, `[sAnimal-6 …]`, `TOTAL=8`, `=== sAnimal part 6/6 complete ===`.

- [ ] **Step 1: Cell list**

```bash
# AT013 × F1
run_cell "AT013-F1" \
    --train-animals AT013 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT013 F1 fractional

# AT013 × F2
run_cell "AT013-F2" \
    --train-animals AT013 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT013 F2 fractional

# AT013 × F3
run_cell "AT013-F3" \
    --train-animals AT013 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT013 F3 fractional

# AT013 × F4
run_cell "AT013-F4" \
    --train-animals AT013 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.0 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT013 F4 fractional

# AT013 × F5
run_cell "AT013-F5" \
    --train-animals AT013 \
    --force-scale 1.0 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT013 F5 fractional

# AT013 × F6
run_cell "AT013-F6" \
    --train-animals AT013 \
    --force-scale 1.2 --joint-damping 1e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT013 F6 fractional

# AT013 × F7
run_cell "AT013-F7" \
    --train-animals AT013 \
    --force-scale 1.0 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.0 --seed 0 \
    --wandb-tags sAnimal AT013 F7 fractional

# AT013 × F8
run_cell "AT013-F8" \
    --train-animals AT013 \
    --force-scale 1.2 --joint-damping 2e-6 --shoulder-damping 6e-7 \
    --control-cost 0.05 --control-diff-cost 0.05 --seed 0 \
    --wandb-tags sAnimal AT013 F8 fractional
```

- [ ] **Step 2: Verify**

```
bash -n sweep_sAnimal_6.sh && echo OK
[ "$(grep -c '^run_cell ' sweep_sAnimal_6.sh)" = "8" ] && grep '^TOTAL=' sweep_sAnimal_6.sh
grep -E 'WANDB_GROUP=' sweep_sAnimal_6.sh
```
Expected: `OK`, `TOTAL=8`, `WANDB_GROUP="sAnimal-part6"`.

- [ ] **Step 3: Commit**

```
git add sweep_sAnimal_6.sh
git commit -m "add sAnimal sweep script 6: AT013 F1-F8 (8 cells)"
```

---

## Task 8: Cross-script verification

Confirm all 6 scripts together cover exactly the 45 expected (animal, cell) pairs with no duplicates and no omissions.

**Files:** none (verification only).

- [ ] **Step 1: Total cell count**

```
grep -h '^run_cell ' sweep_sAnimal_*.sh | wc -l
```
Expected: `45`.

- [ ] **Step 2: Unique (animal, cell) pairs**

```
grep -h '^run_cell ' sweep_sAnimal_*.sh | awk '{print $2}' | sort -u | wc -l
```
Expected: `45` — every pair must be unique.

- [ ] **Step 3: All 5 animals × 9 cells appear**

```
grep -h '^run_cell ' sweep_sAnimal_*.sh | awk '{print $2}' | sort | uniq -c | awk '{print $1}' | sort -u
```
Expected: a single line `1` — each tag appears exactly once.

- [ ] **Step 4: Per-animal coverage = 9 cells**

```
for A in A36-1 AT006 AT009 AT012 AT013; do
  echo -n "$A: "
  grep -h '^run_cell ' sweep_sAnimal_*.sh | grep -c "${A}-"
done
```
Expected: each line ends in `: 9`.

- [ ] **Step 5: Per-cell coverage = 5 animals**

```
for C in C0 F1 F2 F3 F4 F5 F6 F7 F8; do
  echo -n "$C: "
  grep -h '^run_cell ' sweep_sAnimal_*.sh | grep -cE "\"[A-Z0-9-]+-${C}\""
done
```
Expected: each line ends in `: 5`.

If any of Steps 1–5 fails, identify the offending script(s) by re-running the same `grep` per file. Fix the script, re-run Tasks 2–7's verifies, then re-run Task 8.

---

## Task 9: Write `SANIMAL_LAUNCH.md`

**Files:**
- Create: `SANIMAL_LAUNCH.md`

- [ ] **Step 1: Write the launch doc**

```markdown
# sAnimal Launch Commands — per-animal hyperparameter sweep (2^(4-1) fractional + center)

**45 runs** across **6 scripts**, **6 GPUs**, partitioned as 2× 2-GPU jobs + 2× 1-GPU jobs (matches s17/s19 layout). **Time-box: ~28 h wall** at `BUDGET_HOURS=30`.

Goal: train one specialist policy per (animal, cell) on the animal's kinematics only, then characterize per-animal hyperparameter optima. Outputs feed the per-mouse Bayesian posterior in `2026-05-02-hierarchical-bayesian-emg-population-design.md`.

Spec & rationale: `docs/superpowers/specs/2026-05-02-sAnimal-per-animal-hyperparameter-sweep-design.md`.

## Pre-launch checklist

- [ ] No s17/s18/s19 sweeps still running (`pgrep -af 'sweep_s1[789]_ms_[0-9]\.sh'` is empty).
- [ ] All 6 GPUs idle (`nvidia-smi --query-gpu=index,memory.used --format=csv` shows < 1000 MiB used per GPU).
- [ ] Reference clip dir present (`ls vnl_playground/tasks/mouse/reference_data_moving_shoulder_v16_5animals/ | wc -l` ≥ 270).
- [ ] Preflight (Task 1 of plan) was run within the past 24h and passed.
- [ ] 6 sweep scripts present and syntactically valid (`for f in sweep_sAnimal_*.sh; do bash -n $f && echo "$f OK"; done` prints 6 OK lines).

## Interactive Job 1 — 2 GPUs (machine #1)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_sAnimal_1.sh > /tmp/sweep_sAnimal_1_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_sAnimal_2.sh > /tmp/sweep_sAnimal_2_master.log 2>&1 &
```

## Interactive Job 2 — 2 GPUs (machine #2)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_sAnimal_3.sh > /tmp/sweep_sAnimal_3_master.log 2>&1 &
```

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=1 nohup bash sweep_sAnimal_4.sh > /tmp/sweep_sAnimal_4_master.log 2>&1 &
```

## Interactive Job 3 — 1 GPU (machine #3)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_sAnimal_5.sh > /tmp/sweep_sAnimal_5_master.log 2>&1 &
```

## Interactive Job 4 — 1 GPU (machine #4)

```bash
cd /root/vast/eric/vnl-playground && CUDA_VISIBLE_DEVICES=0 nohup bash sweep_sAnimal_6.sh > /tmp/sweep_sAnimal_6_master.log 2>&1 &
```

## Script summary

| GPU | Script | Cells | Cell list |
|---|---|---:|---|
| Job1 GPU0 | `sweep_sAnimal_1.sh` | 5  | A36-1: C0, F1, F2, F3, F4 |
| Job1 GPU1 | `sweep_sAnimal_2.sh` | 8  | A36-1: F5–F8 + AT006: C0, F1, F2, F3 |
| Job2 GPU0 | `sweep_sAnimal_3.sh` | 8  | AT006: F4–F8 + AT009: C0, F1, F2 |
| Job2 GPU1 | `sweep_sAnimal_4.sh` | 8  | AT009: F3–F8 + AT012: C0, F1 |
| Job3 GPU0 | `sweep_sAnimal_5.sh` | 8  | AT012: F2–F8 + AT013: C0 |
| Job4 GPU0 | `sweep_sAnimal_6.sh` | 8  | AT013: F1–F8 |
| **Total** |  | **45** |  |

## Cell parameters (held identical across animals)

| Cell | --force-scale | --joint-damping | --control-cost | --control-diff-cost |
|------|---|---|---|---|
| C0 | 1.1 | 1.5e-6 | 0.025 | 0.025 |
| F1 | 1.0 | 1e-6   | 0.0   | 0.0   |
| F2 | 1.2 | 1e-6   | 0.0   | 0.05  |
| F3 | 1.0 | 2e-6   | 0.0   | 0.05  |
| F4 | 1.2 | 2e-6   | 0.0   | 0.0   |
| F5 | 1.0 | 1e-6   | 0.05  | 0.05  |
| F6 | 1.2 | 1e-6   | 0.05  | 0.0   |
| F7 | 1.0 | 2e-6   | 0.05  | 0.0   |
| F8 | 1.2 | 2e-6   | 0.05  | 0.05  |

`--shoulder-damping 6e-7` and `--seed 0` are held fixed for every cell. `--train-animals` is the per-animal filter.

## Monitoring

Check progress per script:
```bash
for f in /tmp/sweep_sAnimal_{1..6}_master.log; do
  echo "=== $f ==="
  tail -5 "$f" 2>/dev/null
  echo
done
```

Check per-cell logs (one file per cell):
```bash
ls /tmp/sweep_sAnimal-*-*.log | wc -l   # should grow over time toward 45
```

## Stop conditions

- A script crashes early (CRASHED line for a cell): leave the script running — subsequent cells will still be attempted. Investigate the failing run via its `/tmp/sweep_sAnimal-<tag>-*.log`.
- All scripts running but no progress in wandb after 30 min from launch: kill via `pkill -f sweep_sAnimal_` and inspect `/tmp/sweep_sAnimal_*_master.log`.

## Post-sweep

Per the spec, build the per-animal heatmap (9 cells × 5 animals) of cohort-mean correlation, identify each animal's best cell, and feed the 45 checkpoints into the Bayesian framework's per-mouse posterior cache.
```

- [ ] **Step 2: Verify markdown structure**

```
grep -c '^## ' SANIMAL_LAUNCH.md
```
Expected: ≥ 7 (Pre-launch, Job 1–4, Script summary, Cell parameters, Monitoring, Stop conditions, Post-sweep).

- [ ] **Step 3: Verify all 6 scripts referenced**

```
for n in 1 2 3 4 5 6; do
  grep -q "sweep_sAnimal_${n}.sh" SANIMAL_LAUNCH.md && echo "script $n OK" || echo "script $n MISSING"
done
```
Expected: 6 `OK` lines.

- [ ] **Step 4: Commit**

```
git add SANIMAL_LAUNCH.md
git commit -m "add SANIMAL_LAUNCH doc with per-job nohup commands and script summary"
```

---

## Task 10: Final cross-cutting verification before handoff

**Files:** none.

- [ ] **Step 1: All 6 scripts pass `bash -n`**

```
for f in sweep_sAnimal_*.sh; do
  bash -n "$f" && echo "$f OK" || echo "$f FAIL"
done
```
Expected: 6 `OK` lines.

- [ ] **Step 2: Re-run Task 8 verifications as a final check**

```
echo "Total: $(grep -h '^run_cell ' sweep_sAnimal_*.sh | wc -l)"   # 45
echo "Unique: $(grep -h '^run_cell ' sweep_sAnimal_*.sh | awk '{print $2}' | sort -u | wc -l)"   # 45
```
Expected: both `45`.

- [ ] **Step 3: Confirm spec + plan + scripts + launch doc are all committed**

```
git status --short
```
Expected: empty (or only files unrelated to sAnimal). `git log --oneline -10` should show the spec commit, the rebalance commit, and one commit per script (Tasks 2–7) plus the launch doc commit.

- [ ] **Step 4: Print handoff summary**

```
echo "sAnimal sweep ready to launch. Files:"
ls sweep_sAnimal_*.sh SANIMAL_LAUNCH.md
echo "Spec: docs/superpowers/specs/2026-05-02-sAnimal-per-animal-hyperparameter-sweep-design.md"
echo "Plan: docs/superpowers/plans/2026-05-02-sAnimal-sweep-launch-plan.md"
echo "Launch: see SANIMAL_LAUNCH.md"
```

This is the handoff point. The user runs the launch commands from `SANIMAL_LAUNCH.md` themselves.

---

## Self-review checklist (run before handoff)

- [x] **Spec coverage:** 45 runs (Tasks 2–7 produce 5+8+8+8+8+8=45 ✓), 9 cells × 5 animals (Task 8 verifies), shoulder_damping fixed at 6e-7 (every cell call ✓), seed=0 (every cell ✓), z_baseline_x2 EMG norm (BASE_ARGS in Task 2 template ✓), 6-script GPU partition (Task 9 launch doc ✓), preflight gate (Task 1 ✓), per-animal cross-eval against all 5 animals (`--emg-animals A36-1 AT006 AT009 AT012 AT013` in BASE_ARGS ✓).
- [x] **Placeholder scan:** no TBD/TODO. The "machine #1/#2/#3/#4" labels in the launch doc are templated to the user's hostnames at launch time — same pattern as S17_MS_LAUNCH.md and S19_MS_LAUNCH.md.
- [x] **Type consistency:** every cell tag is `<animal>-<cell>` with `<cell>` in `{C0, F1..F8}` and `<animal>` in `{A36-1, AT006, AT009, AT012, AT013}`. Cell parameters are pinned identically across animals — Task 8's grep counts enforce uniqueness and completeness.
- [x] **Real APIs:** `--train-animals`, `--force-scale`, `--joint-damping`, `--shoulder-damping`, `--control-cost`, `--control-diff-cost`, `--emg-animals`, `--emg-norm-method`, `--reference-data-path`, `--num-timesteps`, `--num-evals`, `--seed`, `--tag`, `--run-name`, `--wandb-group`, `--wandb-tags`, `--no-wandb` all match `train_mouse_janelia_sigmoid_moving_shoulder.py` flags as inspected.
