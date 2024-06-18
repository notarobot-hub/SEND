module load python/3.10
source env/bin/activate
source .env

python ./MIND/utils/gather_results.py
python ./MIND/generate_hd_chunk.py