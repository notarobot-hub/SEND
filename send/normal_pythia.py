import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import wandb
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from sensitivity.efficient_eigenscore.efficient import *
import os


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        return input_ids, attention_mask, labels


def pad_embeddings(embeddings, target_dim=512):
    num_datapoints, num_dim = embeddings.shape
    if num_dim < target_dim:
        padding = np.zeros((num_datapoints, target_dim - num_dim))
        padded_embeddings = np.concatenate((embeddings, padding), axis=1)
    else:
        padded_embeddings = embeddings
    return padded_embeddings


def net_change(embeddings: np.ndarray):
    return np.sum(np.abs(np.diff(embeddings, axis=0)), axis=0)


def net_variability(embeddings: np.ndarray):
    return np.var(embeddings, axis=0)


def combined_sensitivity(embeddings: np.ndarray):
    net_change_comp = net_change(embeddings)
    net_variability_comp = net_variability(embeddings)
    return net_change_comp * net_variability_comp


def top_k_percent_sensitive(net_change: np.ndarray, k: float):
    net_change = net_change[0]
    num_elements = int(k * len(net_change))
    sorted_indices = np.argsort(net_change)[::-1]
    top_indices = sorted_indices[:num_elements]
    top_changes = {index: net_change[index] for index in top_indices}
    return top_changes


def get_sensitive_neurons(embeddings_list, k=0.3):
    embeddings_array = np.array(embeddings_list)
    avg_embeddings = np.mean(embeddings_array, axis=0)
    sensitivity = combined_sensitivity(avg_embeddings)
    top_sensitive_neurons = top_k_percent_sensitive(sensitivity, k)
    print(top_sensitive_neurons)
    return top_sensitive_neurons


def compute_eigenscore(X, limit=20, alpha=0.001):
    X = X[:, :, 0] if X.ndim == 3 else X
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean
    X_std = np.std(X_centered, axis=0, keepdims=True)
    X_normalized = X_centered / X_std
    sigma_max = power_method(X_normalized)
    X_normalized /= sigma_max
    c_m_values = np.array([compute_c_m(m) for m in range(1, limit + 1)])
    K = X.shape[1]
    Nz = 20
    Z = np.random.randn(K, Nz)
    T_prime_m_values = compute_T_prime_m(X_normalized, limit, Z)
    d_m_values = np.array([compute_dm(T_prime_m_values, Z, m) for m in range(1, limit + 1)])
    n = X.shape[0]
    _, S, _ = np.linalg.svd(X_normalized, full_matrices=False)
    lambda_reg = (S**2 / (n - 1)) + alpha
    eigenscore = np.sum(d_m_values * c_m_values)
    eigenscore += np.sum(np.log(lambda_reg))
    return eigenscore / n


def get_embeddings(model, tokenizer, test_texts, num_passes=10, device='cuda'):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for _ in range(num_passes):
            for text in test_texts:
                inputs = tokenizer(text, return_tensors='pt').to(device)
                outputs = model(**inputs)
                hidden_states = outputs.hidden_states
                SLT_embedding = hidden_states[-2][:, -2, :].float()
                embeddings.append(SLT_embedding.cpu().numpy())
    return np.array(embeddings)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", type=str, default="alpaca")
    argument_parser.add_argument("--model_version", type=str, default="1b")
    argument_parser.add_argument("--model_type", type=str, default="8B")
    argument_parser.add_argument("--iter", type=int, default=0)
    args = argument_parser.parse_args()

    data_path = args.data
    model_type = args.model_type
    model_version = args.model_version    
    iteration = args.iter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use EleutherAI's Pythia naming convention.
    MODEL_NAME = f"EleutherAI/pythia-{model_version}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    SCRATCH_DIR = os.getenv("SCRATCH")
    # Load the model with GPTNeoXForCausalLM since Pythia is built on GPT-NeoX.
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=SCRATCH_DIR,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True
    ).to(device)
    tokenizer.pad_token = tokenizer.eos_token

    print("starting training")

    data = pd.read_csv(f"send/data/{data_path}.csv")
    if 'helm' in data_path.lower():
        all_data = data['texts'].tolist()[:150]
        total_data = len(all_data)
        eval_data = data['texts'].tolist()[150:]
    else: 
        all_data = data['texts'].tolist()[:300]
        total_data = len(all_data)
        eval_data = data['texts'].tolist()[300:350]
    large_texts_end = int(0.8 * total_data)
    tracking_texts_end = int(0.9 * total_data)

    large_texts = all_data[:large_texts_end]
    tracking_texts = all_data[large_texts_end:tracking_texts_end]
    test_texts = all_data[tracking_texts_end:]
    eval_texts = eval_data

    large_dataset = TextDataset(large_texts, tokenizer)
    tracking_dataset = TextDataset(tracking_texts, tokenizer)

    large_dataloader = DataLoader(large_dataset, batch_size=1, shuffle=True)
    tracking_dataloader = DataLoader(tracking_dataset, batch_size=1, shuffle=False)

    LR = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    num_epochs = 4

    wandb.init(
        project=f"pythia_normal_{model_version}-{data_path.split('/')[-1].split('.')[0]}",
        config={
            "model_name": MODEL_NAME,
            "epochs": num_epochs,
            "batch_size": 1,
            "learning_rate": LR,
            "dataset_size": len(large_dataset) + len(tracking_dataset),
            "device": device.type,
            "data_path": data_path,
            "eigenscore_type": "EES"
        },
    )
    wandb.watch(model, log="all")

    embeddings_list = []

    # Freeze the first 24 layers (or as many as available) using the GPT-NeoX attribute.
    freeze_layers = min(24, len(model.gpt_neox.layers))
    print(f"Freezing first {freeze_layers} layers")
    for i in range(freeze_layers):
        for param in model.gpt_neox.layers[i].parameters():
            param.requires_grad = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        for batch in tqdm(large_dataloader, desc="Training batches", leave=False):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({
                "epoch": epoch + 1,
                "batch_loss": loss.item()
            })

        epoch_loss = running_loss / len(large_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss}')
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": epoch_loss
        })

        model.eval()
        epoch_embeddings = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(tracking_dataloader, desc="Generating embeddings", leave=False):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                embeddings = pad_embeddings(hidden_states[-2][:, -2, :].cpu().float().numpy())
                epoch_embeddings.append(embeddings)

        embeddings_list.append(epoch_embeddings)

        if (epoch + 1) % 2 == 0:
            embeddings_list.pop(0)
        embeddings = get_embeddings(model, tokenizer, test_texts, device=device.type)
        average_eigenscore = compute_eigenscore(embeddings)
        wandb.log({"average_eigenscore": average_eigenscore})

        eval_embeddings = get_embeddings(model, tokenizer, eval_texts, device=device.type)
        eval_eigenscore = compute_eigenscore(eval_embeddings)
        wandb.log({"eval_eigenscore": eval_eigenscore})

    wandb.finish()
    os.makedirs(f"{SCRATCH_DIR}/pythia_normal", exist_ok=True)
    model_name = f"pythia_normal_{model_version}-{data_path.split('/')[-1].split('.')[0]}_{iteration}"
    model.save_pretrained(f"{SCRATCH_DIR}/pythia_normal/{model_version}-{data_path.split('/')[-1].split('.')[0]}")
    model.push_to_hub(f"YourUsername/{model_name}")
    print("Training complete!")


if __name__ == "__main__":
    main()
