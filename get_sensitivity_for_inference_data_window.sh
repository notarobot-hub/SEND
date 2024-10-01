#!/bin/bash
#SBATCH --job-name=halu_get_data_sensitivity_2.8B
#SBATCH --nodes=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=24:00:00
#SBATCH --mail-user= 
#SBATCH --output=logs/halu/%A_%a_2.8B.out
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

module load python/3.10
module load cuda/12.3.2
source env/bin/activate
source .env
wandb login

model_name="2.8B"
data_point_ids=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

# Calculate data point range based on SLURM_ARRAY_TASK_ID
data_point_start=$(( SLURM_ARRAY_TASK_ID * 5 ))
data_point_end=$(( data_point_start + 4 ))

for data_point_id in $(seq $data_point_start $data_point_end)
do
    python ./sensitivity/pipeline_sliding_window.py --model_name $model_name --data_id $data_point_id --function get_sensitivity_for_single_point
done