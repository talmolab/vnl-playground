#!/bin/bash
#SBATCH --job-name=train_flat_arena
#SBATCH --account=kempner_hms
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00
#SBATCH --mem=256G
#SBATCH --output=output/train_output.out
#SBATCH --error=error/train_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=daniel_sprague@fas.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
source activate vnl

# Run the job
MUJOCO_GL=egl python vnl_mjx/train.py