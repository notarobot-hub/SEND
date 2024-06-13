#!/bin/bash
#SBATCH --job-name=halu_data_generation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:2
#SBATCH --time=0-00:30
#SBATCH --mail-user=shahrad_m@icloud.com
#SBATCH --output=logs/halu/%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --array=107-125:18        # Define job array for revisions

module load python/3.10
source env/bin/activate
source .env

# Set revision number based on Slurm array task ID
REVISION=$((SLURM_ARRAY_TASK_ID))

srun python ./MIND/generate_data.py --step_num $REVISION 