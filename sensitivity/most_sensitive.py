import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from accelerate import Accelerator
import numpy as np
import warnings

# Suppress specific PyTorch warning about TypedStorage
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def load_model(model_name, revision: int | None=None):
    if revision is not None:
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=f"step{revision}", device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=f"step{revision}")
    else:
        model = GPTNeoXForCausalLM.from_pretrained(model_name, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def penultimate_layer_output(accelerator, model, tokenizer, input_text, middle=True):
    model, tokenizer = accelerator.prepare(model, tokenizer)
    with torch.no_grad():
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
        input_ids = accelerator.prepare(input_ids)
        output = model(input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states
        if middle:
            penultimate_layer_output = hidden_states[len(hidden_states) // 2]
        else:
            penultimate_layer_output = hidden_states[-2]

    return penultimate_layer_output

def extract_embeddings(accelerator, model_name, input_text):
    CHECKPOINTS = range(133000, 144000, 1000)
    penultimate_outputs = []
    for checkpoint in CHECKPOINTS:
        model, tokenizer = load_model(model_name, checkpoint)
        penultimate_layer = penultimate_layer_output(accelerator, model, tokenizer, input_text)
        penultimate_outputs.append(penultimate_layer)

    # make sure the outputs are of numpy arrays of shape (num_checkpoints, sequence_length, hidden_size)
    embeddings = np.concatenate([output.cpu() for output in penultimate_outputs], axis=0)
    return embeddings

"""
Contextual Embedding Extraction
"""

def get_SLT_embedding(embeddings): 
    # get the embeddings of the SLT token
    SLT_embedding = embeddings[:, -2, :]
    return SLT_embedding

def get_pooled_embedding(embeddings):
    # get the pooled embedding of the sequence
    pooled_embedding = embeddings.mean(axis=1)
    return pooled_embedding

def get_last_token_embedding(embeddings):
    # get the embedding of the last token in the sequence
    last_token_embedding = embeddings[:, -1, :]
    return last_token_embedding

"""
Change of neurons during training metrics
"""

def net_change(embeddings: np.ndarray):
    # given array of size (num_checkpoints, hidden_size) where each hidden column is a neuron, sum the absolute difference between each neuron across checkpoints
    return np.sum(np.abs(np.diff(embeddings, axis=0)), axis=0)

def net_variability(embeddings: np.ndarray):
    variability = np.var(embeddings, axis=0)
    return variability

def combined_sensitivity(embeddings: np.ndarray):
    net_change_comp = net_change(embeddings)
    net_variability_comp = net_variability(embeddings)
    # combine the net change and net variability to get the sensitivity
    return net_change_comp * net_variability_comp

def top_10_sensitive(net_change: np.ndarray):
    # get the top 10 most sensitive neurons given an array of size (nd, )
    return np.argsort(net_change)[::-1][:10]

def top_k_percent_sensitive(net_change: np.ndarray, k: float):
    # Calculate the number of top elements to include
    num_elements = int(k * len(net_change))
    
    # Get indices of the sorted array in descending order
    sorted_indices = np.argsort(net_change)[::-1]
    
    # Select the top k% indices
    top_indices = sorted_indices[:num_elements]
    
    # Create a dictionary of indices and their corresponding net change values
    top_changes = {index: net_change[index] for index in top_indices}
    
    return top_changes

if __name__ == '__main__':
    model_name = "EleutherAI/pythia-70m"
    accelerator = Accelerator()
    full_emb = extract_embeddings(accelerator, model_name, "what do you think about einstein?")
    SLT_emb = get_SLT_embedding(full_emb)
    net = combined_sensitivity(SLT_emb)
    top_10 = top_10_sensitive(net)

