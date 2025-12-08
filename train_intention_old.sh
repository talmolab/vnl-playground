#!/bin/bash
#SBATCH --job-name=train_intention_old
#SBATCH --account=kempner_hms
#SBATCH --partition=kempner
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00
#SBATCH --mem=256G
#SBATCH --output=output/train_intention_%A_%a.out
#SBATCH --error=error/train_intention_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=daniel_sprague@fas.harvard.edu

# Load modules
module load python

# Activate conda environment
source activate vnl

# List of config names
CONFIGS=("flat_arena_transfer" "flat_arena_transfer_nofreeze" "flat_arena_basic")

# Pick the config for this array task
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running config: $CFG"

# Run
MUJOCO_GL=egl python vnl_mjx/train_intention_old.py "$CFG"