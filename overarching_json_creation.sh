#!/bin/bash
#SBATCH --job-name=json_creation_multiple_checks
#SBATCH --nodes=1
#SBATCH --mail-user=juan.guerra@mila.quebec
#SBATCH --output=logs/halu/%A_%a.out
#SBATCH --mail-type=ALL

module load python/3.10
source $HOME/hallucinations_venv/bin/activate
source .env

python ./MIND/utils/gather_results.py --model_checkpoints 21 82 143
python ./MIND/generate_hd_chunk.py --model_checkpoints 21 82 143

python ./MIND/utils/gather_results.py --model_checkpoints 107 110 114
python ./MIND/generate_hd_chunk.py --model_checkpoints 107 110 114

python ./MIND/utils/gather_results.py --model_checkpoints 107 125 143
python ./MIND/generate_hd_chunk.py --model_checkpoints 107 125 143

python ./MIND/utils/gather_results.py --model_checkpoints 124 125 126
python ./MIND/generate_hd_chunk.py --model_checkpoints 124 125 126

python ./MIND/utils/gather_results.py --model_checkpoints 110 125 143
python ./MIND/generate_hd_chunk.py --model_checkpoints 110 125 143