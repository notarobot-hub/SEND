#!/bin/bash
#SBATCH --job-name=train_SAEs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=0-15:00
#SBATCH --mail-user=
#SBATCH --output=logs/halu/%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --array=107-143:18       # Define job array for revisions

module load python/3.10
module load cuda/12.3.2
source env/bin/activate
source .env
wandb login

# Set revision number based on Slurm array task ID
REVISION=$((SLURM_ARRAY_TASK_ID))

# 107, 125, 143
python ./SAEs/sparse_coding-main/big_sweep_experiments.py --revision $REVISION