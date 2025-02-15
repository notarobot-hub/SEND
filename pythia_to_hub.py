import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define variables.
SCRATCH_DIR = os.environ.get("SCRATCH")
dataset_names = ["helm", "legalbench", "codesearchnet", "alpaca"]

for dataset_name in dataset_names:

    model_name = f"pythia_send_1b_{dataset_name}"
    repo_id = f"Shahradmz/pythia_send_{model_name}"
    model_dir = f"{SCRATCH_DIR}/pythia_send/deduped-{dataset_name}"

    # Load model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    hf_token = ""

    # Use the 'token' parameter instead of 'use_auth_token' as recommended.
    model.push_to_hub(repo_id, token=hf_token, exist_ok=True)
    tokenizer.push_to_hub(repo_id, token=hf_token, exist_ok=True)