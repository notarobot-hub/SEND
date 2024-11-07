#!/bin/bash
#SBATCH --job-name=halu_data_generation
#SBATCH --nodes=1

module load python/3.10
source .env
source $HOME/hallucinations_venv/bin/activate

for REVISION in $(seq 81 130)
do
    srun python ./MIND/generate_data.py --step_num $REVISION 
done