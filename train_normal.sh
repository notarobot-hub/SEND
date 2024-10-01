#!/bin/bash
#SBATCH --job-name=continual-normal
#SBATCH --partition=short-unkillable
#SBATCH --tasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=128G
#SBATCH --time=0-03:00:00  # 3 hours
#SBATCH --mail-user= 
#SBATCH --output=logs/halu/%A_%a_1bcontinual.out
#SBATCH --mail-type=ALL

module load python/3.10
module load cuda/12.3.2
source env/bin/activate
source .env
wandb login

# Repeat the command 15 times
for i in {1..5}
do
    python desend/train_normal.py
done

for i in {1..5}
do
    python desend/train_no_meta.py
done