#!/bin/bash
#SBATCH --job-name=halu_data_generation_6.9B
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:2
#SBATCH --time=0-10:00
#SBATCH --mail-user= 
#SBATCH --output=logs/halu/%A_%a_6.9B.out
#SBATCH --mail-type=ALL

module load python/3.10
module load cuda/12.3.2
source env/bin/activate
source .env
wandb login

model_names=("6.9B")

for model_name in "${model_names[@]}"
do
    python ./sensitivity/pipeline_sliding_window.py --model_name $model_name --function prepare_data_dir
done