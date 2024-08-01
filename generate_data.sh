#!/bin/bash
#SBATCH --job-name=halu_data_generation
#SBATCH --nodes=1
#SBATCH --array=143       # Define job array for revisions

module load python/3.10
source env/bin/activate
source $HOME/hallucinations_venv/bin/activate

# Set revision number based on Slurm array task ID
REVISION=$((SLURM_ARRAY_TASK_ID))

srun python ./MIND/generate_data.py --step_num $REVISION 