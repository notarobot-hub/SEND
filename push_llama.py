from transformers import AutoModel, AutoTokenizer, AutoConfig
import os

# Set your Hugging Face Hub token
hf_token = ""

# Set the model directory and repository name
model_dir = os.path.expandvars("$SCRATCH/llama_8b_send")
print(model_dir)
repo_name = "Shahradmz/llama-8b-send"

# Load the model, tokenizer, and config
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Push the model, tokenizer, and config to the Hugging Face Hub
model.push_to_hub(repo_name, use_auth_token=hf_token)
tokenizer.push_to_hub(repo_name, use_auth_token=hf_token)

print(f"Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_name}")