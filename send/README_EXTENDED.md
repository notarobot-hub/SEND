# Extended SEND Implementation

This directory contains an extended implementation of the SEND (Sensitive Neuron Dropout) method that supports multiple datasets and language models.

## Overview

SEND is a method for analyzing and modifying language models by identifying and dropping sensitive neurons during training. This extended implementation supports:

### Supported Models
- **Llama 2 7B** (`meta-llama/Llama-2-7b-chat-hf`)
- **Llama 3.1 8B** (`meta-llama/Llama-3.1-8B-Instruct`)
- **Vicuna 7B** (`lmsys/vicuna-7b-v1.5`)

### Supported Datasets
- **TriviaQA**: Question-answering dataset with trivia questions
- **TruthfulQA**: Dataset for measuring truthfulness in language models
- **CoQA**: Conversational question-answering dataset
- **TyDiQA**: Typologically diverse question-answering dataset

## Files

- `extended_send_multi_dataset.py`: Main training script for SEND on multiple datasets/models
- `generate_responses.py`: Script to generate responses using trained SEND models
- `run_send_batch.py`: Batch script to run SEND training on multiple model-dataset combinations
- `requirements_extended.txt`: Dependencies for the extended implementation

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_extended.txt
```

2. Ensure you have access to the models (you may need to request access to Llama models on Hugging Face)

## Usage

### Single Model-Dataset Training

Train a single model on a specific dataset:

```bash
python send/extended_send_multi_dataset.py \
    --dataset triviaqa \
    --model_name llama2-7b \
    --max_samples 500 \
    --num_epochs 4 \
    --learning_rate 1e-4 \
    --sensitivity_threshold 0.3 \
    --batch_size 1 \
    --max_length 512 \
    --output_dir ./output
```

### Batch Training

Run SEND training on multiple model-dataset combinations:

```bash
# Dry run to see what commands will be executed
python send/run_send_batch.py --dry_run

# Execute all combinations
python send/run_send_batch.py

# Custom selection of models and datasets
python send/run_send_batch.py \
    --models llama2-7b vicuna-7b \
    --datasets triviaqa truthfulqa \
    --max_samples 300 \
    --num_epochs 3
```

### Generating Responses

Generate responses using a trained SEND model:

```bash
python send/generate_responses.py \
    --model_path ./output/llama2-7b_triviaqa_SEND \
    --dataset triviaqa \
    --num_samples 10 \
    --max_length 100 \
    --temperature 0.7 \
    --output_file results.json
```

## Command Line Arguments

### Training Script (`extended_send_multi_dataset.py`)

- `--dataset`: Dataset to use (triviaqa, truthfulqa, coqa, tydiqa)
- `--model_name`: Model to train (llama2-7b, llama3.1-8b, vicuna-7b)
- `--max_samples`: Maximum number of samples to use from dataset
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimization
- `--sensitivity_threshold`: Threshold for identifying sensitive neurons (0.0-1.0)
- `--batch_size`: Training batch size
- `--max_length`: Maximum sequence length
- `--device`: Device to use (auto, cuda, cpu)
- `--use_wandb`: Enable wandb logging
- `--output_dir`: Directory to save trained models

### Generation Script (`generate_responses.py`)

- `--model_path`: Path to trained SEND model
- `--dataset`: Dataset name for loading test questions
- `--num_samples`: Number of test questions to generate responses for
- `--max_length`: Maximum length of generated response
- `--temperature`: Temperature for generation (0.0-1.0)
- `--top_p`: Top-p sampling parameter (0.0-1.0)
- `--output_file`: File to save results (optional)

### Batch Script (`run_send_batch.py`)

- `--models`: List of models to train
- `--datasets`: List of datasets to use
- `--dry_run`: Print commands without executing
- All other training parameters are passed through to the training script

## How SEND Works

1. **Training Phase**: The model is trained on the selected dataset while monitoring hidden state activations
2. **Sensitivity Analysis**: Every 2 epochs, the system identifies sensitive neurons using the efficient eigenscore method
3. **Neuron Dropout**: Sensitive neurons are dropped (weights set to 0) for the next 3 epochs
4. **Iterative Process**: This cycle continues throughout training, allowing the model to adapt to the dataset while reducing sensitivity

## Output Structure

Trained models are saved with the following structure:
```
output/
├── llama2-7b_triviaqa_SEND/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_config.json
├── llama2-7b_truthfulqa_SEND/
│   └── ...
└── ...
```

## Example Workflow

1. **Train SEND models**:
```bash
python send/run_send_batch.py --models llama2-7b --datasets triviaqa
```

2. **Generate responses**:
```bash
python send/generate_responses.py \
    --model_path ./output/llama2-7b_triviaqa_SEND \
    --dataset triviaqa \
    --num_samples 20 \
    --output_file triviaqa_responses.json
```

3. **Analyze results**: Check the generated JSON file for question-answer pairs and generated responses

## Tips and Best Practices

1. **Start small**: Begin with smaller datasets and fewer epochs to test the setup
2. **Monitor resources**: SEND training can be memory-intensive; monitor GPU memory usage
3. **Adjust sensitivity**: Lower sensitivity thresholds (0.1-0.2) may work better for some datasets
4. **Use wandb**: Enable wandb logging to track training progress and metrics
5. **Checkpoint models**: The script automatically saves models after training

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, max_length, or max_samples
2. **Model Loading Errors**: Ensure you have access to the model on Hugging Face
3. **Dataset Loading Issues**: Check internet connection and dataset availability
4. **CUDA Errors**: Verify CUDA installation and GPU compatibility

### Performance Optimization

- Use `device_map="auto"` for automatic GPU memory management
- Reduce `max_samples` for faster training
- Use smaller models for initial testing
- Enable gradient checkpointing for memory efficiency

## Citation

If you use this implementation, please cite the original SEND paper and this extended implementation.

## License

This implementation follows the same license as the original SEND codebase.
