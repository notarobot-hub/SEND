module load python/3.10
<<<<<<< HEAD
source $HOME/hallucinations_venv/bin/activate
source .env

python ./MIND/utils/gather_results.py --model_checkpoints 107 125 143
=======
source env/bin/activate
source .env

python ./MIND/utils/gather_results.py
>>>>>>> 6a247e9eaff057ebb6f917bb7b17a31ba9ceccd3
python ./MIND/generate_hd_chunk.py