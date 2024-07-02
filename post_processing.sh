#!/bin/bash
module load python/3.10
source $HOME/hallucinations_venv/bin/activate
source .env

python ./MIND/utils/gather_results.py --model_checkpoints 21 82 143
python ./MIND/generate_hd_chunk.py

echo "Done post processing"