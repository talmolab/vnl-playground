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
