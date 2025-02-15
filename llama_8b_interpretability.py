from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Path to your local 8B Llama model
model_path = "/network/scratch/s/shahrad.mohammadzadeh/llama_8b_send"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('Shahradmz/send-tuned-pythia-1b-helm')
SCRATCH_DIR = os.getenv("SCRATCH")

# Load the model with automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    'Shahradmz/send-tuned-pythia-1b-helm',
    cache_dir=SCRATCH_DIR,
    device_map="auto",               # Automatically maps model layers to available GPUs
    torch_dtype=torch.bfloat16,      # Use half-precision for faster inference and reduced memory usage
)

# Initialize the text generation pipeline without specifying the device
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer
)

print("Output by send trained model")

# Define your input prompt
prompt = """def crosstab(index, columns, values=None, rownames=None, colnames=None,
             aggfunc=None, margins=False, margins_name='All', dropna=True,
             normalize=False):"""

# Number of generations
num_generations = 5


for i in range(1, num_generations + 1):
    # Generate text with specified temperature
    output = generator(prompt, max_length=100, num_return_sequences=1)
    
    # Display the generated text
    print(f"\n--- Generation {i} ---")
    print(output[0]['generated_text'])