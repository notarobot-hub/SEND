#!/bin/bash
#SBATCH --job-name=extend_pretraining
#SBATCH --nodes=1
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-user= 
#SBATCH --output=logs/halu/%A_%a_extend.out
#SBATCH --mail-type=ALL

module load python/3.10
module load cuda/12.3.2
source env/bin/activate
source .env
wandb login

python desend/train.py