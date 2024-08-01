#!/bin/bash

module load python/3.10
source .env
source $HOME/hallucinations_venv/bin/activate

python ./MIND/utils/gather_results.py
python ./MIND/generate_hd_chunk.py