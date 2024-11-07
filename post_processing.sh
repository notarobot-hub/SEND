#!/bin/bash

module load python/3.10
source .env
source $HOME/hallucinations_venv/bin/activate

python ./MIND/utils/gather_results.py --model_checkpoints 130 131 132 133 134 135 136 137 138 139 140 141 142 143 --hallu_type 0
python ./MIND/generate_hd_chunk.py