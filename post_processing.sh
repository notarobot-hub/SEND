#!/bin/bash
module load python/3.10
source $HOME/hallucinations_venv/bin/activate
source .env

python ./MIND/utils/gather_results.py --model_checkpoints 130 131 132 134 135 136 137 138 139 140 141 142 143
python ./MIND/generate_hd_chunk.py --model_checkpoints 130 131 132 134 135 136 137 138 139 140 141 142 143