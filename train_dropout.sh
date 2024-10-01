#!/bin/bash
#SBATCH --job-name=final_mitigation
#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=128G
#SBATCH --time=0-03:00:00  # 3 hours
#SBATCH --mail-user= 
#SBATCH --output=logs/halu/%A_%a_1bcontinual_mitigation.out
#SBATCH --mail-type=ALL

module load python/3.10
module load cuda/12.3.2
source env/bin/activate
source .env
wandb login

# Define an array of model names

# Function to run the pipeline for a given model and GPU
run_pipeline() {
    local model_name=$1
    local gpu_id=$2
    echo "Running pipeline for model: $model_name on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python rag_experiments/pipeline.py --model_name "$model_name" &
}

# Run each model in parallel on a different GPU
for i in "${!models[@]}"; do
    run_pipeline "${models[$i]}" "$i"
done

# Wait for all background processes to complete
wait

echo "All models have been processed."