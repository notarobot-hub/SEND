#!/usr/bin/env python3
"""
Batch script to run SEND training on multiple models and datasets
"""

import subprocess
import os
import time
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"✅ {description} completed successfully in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run SEND training on multiple models and datasets")
    parser.add_argument("--models", nargs="+", 
                       default=["llama2-7b", "llama3.1-8b", "vicuna-7b"],
                       help="Models to train")
    parser.add_argument("--datasets", nargs="+",
                       default=["triviaqa", "truthfulqa", "coqa", "tydiqa"],
                       help="Datasets to use")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum number of samples per dataset")
    parser.add_argument("--num_epochs", type=int, default=4,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--sensitivity_threshold", type=float, default=0.3,
                       help="Sensitivity threshold for neuron dropout")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for models")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing them")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate commands for each model-dataset combination
    commands = []
    
    for model in args.models:
        for dataset in args.datasets:
            output_path = os.path.join(args.output_dir, f"{model}_{dataset}_SEND")
            
            cmd = [
                "python", "send/extended_send_multi_dataset.py",
                "--dataset", dataset,
                "--model_name", model,
                "--max_samples", str(args.max_samples),
                "--num_epochs", str(args.num_epochs),
                "--learning_rate", str(args.learning_rate),
                "--sensitivity_threshold", str(args.sensitivity_threshold),
                "--batch_size", str(args.batch_size),
                "--max_length", str(args.max_length),
                "--output_dir", args.output_dir
            ]
            
            if args.use_wandb:
                cmd.append("--use_wandb")
            
            commands.append({
                "command": " ".join(cmd),
                "description": f"SEND training on {model} with {dataset} dataset",
                "output_path": output_path
            })
    
    print(f"Generated {len(commands)} commands:")
    for i, cmd_info in enumerate(commands):
        print(f"{i+1}. {cmd_info['description']}")
        print(f"   Output: {cmd_info['output_path']}")
    
    if args.dry_run:
        print("\nDry run mode - no commands will be executed")
        return
    
    # Ask for confirmation
    response = input(f"\nProceed with executing {len(commands)} commands? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Execute commands
    successful_runs = 0
    failed_runs = 0
    
    for i, cmd_info in enumerate(commands):
        print(f"\nProgress: {i+1}/{len(commands)}")
        
        if run_command(cmd_info["command"], cmd_info["description"]):
            successful_runs += 1
        else:
            failed_runs += 1
        
        # Check if output was created
        if os.path.exists(cmd_info["output_path"]):
            print(f"✅ Model saved to: {cmd_info['output_path']}")
        else:
            print(f"⚠️  Warning: Expected output not found at {cmd_info['output_path']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total commands: {len(commands)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Success rate: {successful_runs/len(commands)*100:.1f}%")
    
    if successful_runs > 0:
        print(f"\nTrained models saved to: {args.output_dir}")
        print("\nTo generate responses with a trained model, use:")
        print("python send/generate_responses.py --model_path <model_path> --dataset <dataset_name>")


if __name__ == "__main__":
    main()
