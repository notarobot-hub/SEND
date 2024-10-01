#!/bin/bash
#SBATCH --job-name=medical-continual-30%dropout
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=24G
#SBATCH --time=00-06:00:00  # 2 days
#SBATCH --partition=main
#SBATCH --mail-user= 
#SBATCH --output=logs/halu/%A_%a_1bcontinual_30%_medical.out
#SBATCH --mail-type=ALL

module load python/3.10
module load cuda/12.3.2
source env/bin/activate
source .env
wandb login

# Repeat the command 15 times
for i in {1..4}
do
    python desend/train_no_meta_medical.py
done