import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from sensitivity.efficient_eigenscore.efficient import *
import os
import json
from datasets import load_dataset
import re


class MultiDatasetTextDataset(Dataset):
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
    print(f"Found {len(top_sensitive_neurons)} sensitive neurons")
    return top_sensitive_neurons


def drop_sensitive_neurons(model, sensitive_neurons, model_family):
    """Drop sensitive neurons based on model architecture"""
    with torch.no_grad():
        if "llama" in model_family.lower():
            # Llama architecture
            for neuron in sensitive_neurons:
                model.model.layers[-2].mlp.gate_proj.weight[neuron, :] = 0
                model.model.layers[-2].mlp.up_proj.weight[neuron] = 0
                model.model.layers[-2].mlp.down_proj.weight[:, neuron] = 0
        elif "vicuna" in model_family.lower():
            # Vicuna uses Llama architecture
            for neuron in sensitive_neurons:
                model.model.layers[-2].mlp.gate_proj.weight[neuron, :] = 0
                model.model.layers[-2].mlp.up_proj.weight[neuron] = 0
                model.model.layers[-2].mlp.down_proj.weight[:, neuron] = 0
        else:
            # Generic transformer architecture
            for neuron in sensitive_neurons:
                if hasattr(model.model.layers[-2], 'mlp'):
                    if hasattr(model.model.layers[-2].mlp, 'gate_proj'):
                        model.model.layers[-2].mlp.gate_proj.weight[neuron, :] = 0
                    if hasattr(model.model.layers[-2].mlp, 'up_proj'):
                        model.model.layers[-2].mlp.up_proj.weight[neuron] = 0
                    if hasattr(model.model.layers[-2].mlp, 'down_proj'):
                        model.model.layers[-2].mlp.down_proj.weight[:, neuron] = 0


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


def load_dataset_texts(dataset_name, split='train', max_samples=500):
    """Load different datasets and extract text for training"""
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name.lower() == "triviaqa":
        dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)
        texts = []
        for item in dataset[:max_samples]:
            question = item['question']
            answer = item['answer']['value']
            text = f"Question: {question}\nAnswer: {answer}"
            texts.append(text)
    
    elif dataset_name.lower() == "truthfulqa":
        dataset = load_dataset("truthful_qa", "generation", split=split)
        texts = []
        for item in dataset[:max_samples]:
            question = item['question']
            answer = item['best_answer']
            text = f"Question: {question}\nAnswer: {answer}"
            texts.append(text)
    
    elif dataset_name.lower() == "coqa":
        dataset = load_dataset("coqa", split=split)
        texts = []
        for item in dataset[:max_samples]:
            question = item['question']
            answer = item['answer']
            text = f"Question: {question}\nAnswer: {answer}"
            texts.append(text)
    
    elif dataset_name.lower() == "tydiqa":
        dataset = load_dataset("tydiqa", "primary_task", split=split)
        texts = []
        for item in dataset[:max_samples]:
            question = item['question_text']
            answer = item['annotations']['minimal_answers'][0]['text'] if item['annotations']['minimal_answers'] else "No answer provided"
            text = f"Question: {question}\nAnswer: {answer}"
            texts.append(text)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Loaded {len(texts)} texts from {dataset_name}")
    return texts


def get_model_config(model_name):
    """Get model configuration based on model name"""
    model_configs = {
        "llama2-7b": {
            "model_path": "meta-llama/Llama-2-7b-chat-hf",
            "family": "llama",
            "freeze_layers": 13,
            "max_length": 512
        },
        "llama3.1-8b": {
            "model_path": "meta-llama/Llama-3.1-8B-Instruct",
            "family": "llama",
            "freeze_layers": 13,
            "max_length": 512
        },
        "vicuna-7b": {
            "model_path": "lmsys/vicuna-7b-v1.5",
            "family": "vicuna",
            "freeze_layers": 13,
            "max_length": 512
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(model_configs.keys())}")
    
    return model_configs[model_name]


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--dataset", type=str, default="triviaqa", 
                               choices=["triviaqa", "truthfulqa", "coqa", "tydiqa"])
    argument_parser.add_argument("--model_name", type=str, default="llama2-7b",
                               choices=["llama2-7b", "llama3.1-8b", "vicuna-7b"])
    argument_parser.add_argument("--max_samples", type=int, default=500)
    argument_parser.add_argument("--num_epochs", type=int, default=4)
    argument_parser.add_argument("--learning_rate", type=float, default=1e-4)
    argument_parser.add_argument("--sensitivity_threshold", type=float, default=0.3)
    argument_parser.add_argument("--batch_size", type=int, default=1)
    argument_parser.add_argument("--max_length", type=int, default=512)
    argument_parser.add_argument("--device", type=str, default="auto")
    argument_parser.add_argument("--use_wandb", action="store_true", default=False)
    argument_parser.add_argument("--output_dir", type=str, default="./output")
    
    args = argument_parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Get model configuration
    model_config = get_model_config(args.model_name)
    model_path = model_config["model_path"]
    model_family = model_config["family"]
    freeze_layers = model_config["freeze_layers"]
    
    print(f"Loading model: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        output_hidden_states=True,
        device_map="auto" if device.type == 'cuda' else None
    )
    
    if device.type == 'cpu':
        model = model.to(device)
    
    print("Model loaded successfully!")
    
    # Load dataset
    try:
        all_texts = load_dataset_texts(args.dataset, max_samples=args.max_samples)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to sample texts...")
        all_texts = [
            "What is the capital of France? Answer: Paris",
            "Who wrote Romeo and Juliet? Answer: William Shakespeare",
            "What is 2 + 2? Answer: 4",
            "What is the largest planet in our solar system? Answer: Jupiter",
            "Who painted the Mona Lisa? Answer: Leonardo da Vinci"
        ] * (args.max_samples // 5)
    
    # Split data
    total_data = len(all_texts)
    large_texts_end = int(0.8 * total_data)
    tracking_texts_end = int(0.9 * total_data)
    
    large_texts = all_texts[:large_texts_end]
    tracking_texts = all_texts[large_texts_end:tracking_texts_end]
    test_texts = all_texts[tracking_texts_end:]
    
    print(f"Data split: {len(large_texts)} train, {len(tracking_texts)} tracking, {len(test_texts)} test")
    
    # Create datasets
    large_dataset = MultiDatasetTextDataset(large_texts, tokenizer, args.max_length)
    tracking_dataset = MultiDatasetTextDataset(tracking_texts, tokenizer, args.max_length)
    
    # Create dataloaders
    large_dataloader = DataLoader(large_dataset, batch_size=args.batch_size, shuffle=True)
    tracking_dataloader = DataLoader(tracking_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=f"SEND-{args.model_name}-{args.dataset}",
            config={
                "model_name": args.model_name,
                "dataset": args.dataset,
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "sensitivity_threshold": args.sensitivity_threshold,
                "max_samples": args.max_samples,
                "device": device.type,
                "eigenscore_type": "EES"
            },
        )
        wandb.watch(model, log="all")
    
    # Freeze early layers
    print(f"Freezing first {freeze_layers} layers")
    for i in range(freeze_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = False
    
    # Training loop
    embeddings_list = []
    current_sensitive_neurons = None
    dropout_applied_epochs = 0
    
    print("Starting training...")
    
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        # Apply sensitive neuron dropout if available
        if current_sensitive_neurons and dropout_applied_epochs < 3:
            drop_sensitive_neurons(model, current_sensitive_neurons, model_family)
            dropout_applied_epochs += 1
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "num_sensitive_neurons": len(current_sensitive_neurons),
                    "dropout_epochs_remaining": 3 - dropout_applied_epochs
                })
        else:
            current_sensitive_neurons = None
            dropout_applied_epochs = 0
        
        # Training phase
        for batch in tqdm(large_dataloader, desc=f"Epoch {epoch+1} Training", leave=False):
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
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "batch_loss": loss.item()
                })
        
        epoch_loss = running_loss / len(large_dataloader)
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {epoch_loss:.4f}')
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "training_loss": epoch_loss
            })
        
        # Generate embeddings for sensitivity analysis
        model.eval()
        epoch_embeddings = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(tracking_dataloader, desc="Generating embeddings", leave=False):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                embeddings = pad_embeddings(hidden_states[-2][:, -2, :].cpu().float().numpy())
                epoch_embeddings.append(embeddings)
        
        embeddings_list.append(epoch_embeddings)
        
        # Keep only last 3 epochs for sensitivity analysis
        if (epoch + 1) % 2 == 0:
            embeddings_list.pop(0)
        
        # Compute sensitive neurons every 2 epochs
        if (epoch + 1) % 2 == 0 and len(embeddings_list) >= 3:
            current_sensitive_neurons = get_sensitive_neurons(embeddings_list[-3:], args.sensitivity_threshold)
            dropout_applied_epochs = 0
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "num_sensitive_neurons": len(current_sensitive_neurons)
                })
        
        # Compute eigenscore on test set
        if len(test_texts) > 0:
            test_embeddings = get_embeddings(model, tokenizer, test_texts, device=device.type)
            test_eigenscore = compute_eigenscore(test_embeddings)
            print(f"Test eigenscore: {test_eigenscore:.4f}")
            
            if args.use_wandb:
                wandb.log({"test_eigenscore": test_eigenscore})
    
    # Save the trained model
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset}_SEND")
    
    print(f"Saving model to {model_save_path}")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save training configuration
    config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "sensitivity_threshold": args.sensitivity_threshold,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": device.type
    }
    
    with open(os.path.join(model_save_path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    if args.use_wandb:
        wandb.finish()
    
    print("Training complete!")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
