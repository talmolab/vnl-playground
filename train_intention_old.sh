#!/bin/bash
#SBATCH --job-name=train_intention_old
#SBATCH --account=kempner_hms
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00
#SBATCH --mem=256G
#SBATCH --output=output/train_intention_%A.out
#SBATCH --error=error/train_intention_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=daniel_sprague@fas.harvard.edu

# Load modules
module load python

# Activate conda environment
source activate vnl

# Run with correct Hydra flag
MUJOCO_GL=egl python vnl_mjx/train_intention_old.py --config-name "flat_arena_transfer_nofreeze"