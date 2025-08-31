#!/usr/bin/env python3
"""
Example usage of the extended SEND implementation
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and print the result"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Command executed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error code {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        return False

def main():
    print("Extended SEND Implementation - Example Usage")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("send/extended_send_multi_dataset.py"):
        print("❌ Error: Please run this script from the project root directory")
        print("   (where the 'send' folder is located)")
        sys.exit(1)
    
    print("✅ Found SEND implementation files")
    
    # Example 1: Train a single model on TriviaQA
    print("\n" + "="*50)
    print("EXAMPLE 1: Training Llama 2 7B on TriviaQA")
    print("="*50)
    
    cmd1 = """python send/extended_send_multi_dataset.py \\
        --dataset triviaqa \\
        --model_name llama2-7b \\
        --max_samples 100 \\
        --num_epochs 2 \\
        --learning_rate 1e-4 \\
        --sensitivity_threshold 0.3 \\
        --batch_size 1 \\
        --max_length 256 \\
        --output_dir ./example_output"""
    
    print("This command will:")
    print("- Load Llama 2 7B model")
    print("- Load TriviaQA dataset (100 samples)")
    print("- Train for 2 epochs with SEND method")
    print("- Save the trained model to ./example_output")
    
    response = input("\nWould you like to run this example? (y/N): ")
    if response.lower() == 'y':
        run_command(cmd1, "Training Llama 2 7B on TriviaQA")
    
    # Example 2: Batch training on multiple models
    print("\n" + "="*50)
    print("EXAMPLE 2: Batch Training on Multiple Models")
    print("="*50)
    
    cmd2 = """python send/run_send_batch.py \\
        --models llama2-7b vicuna-7b \\
        --datasets triviaqa truthfulqa \\
        --max_samples 50 \\
        --num_epochs 2 \\
        --dry_run"""
    
    print("This command will:")
    print("- Show what commands would be executed")
    print("- Train Llama 2 7B and Vicuna 7B")
    print("- On TriviaQA and TruthfulQA datasets")
    print("- Using 50 samples and 2 epochs each")
    
    response = input("\nWould you like to run this example? (y/N): ")
    if response.lower() == 'y':
        run_command(cmd2, "Batch training dry run")
    
    # Example 3: Generate responses
    print("\n" + "="*50)
    print("EXAMPLE 3: Generating Responses")
    print("="*50)
    
    # Check if we have a trained model
    model_path = "./example_output/llama2-7b_triviaqa_SEND"
    if os.path.exists(model_path):
        cmd3 = f"""python send/generate_responses.py \\
            --model_path {model_path} \\
            --dataset triviaqa \\
            --num_samples 5 \\
            --max_length 50 \\
            --temperature 0.7 \\
            --output_file example_responses.json"""
        
        print("This command will:")
        print("- Load the trained SEND model")
        print("- Generate responses for 5 TriviaQA questions")
        print("- Save results to example_responses.json")
        
        response = input("\nWould you like to run this example? (y/N): ")
        if response.lower() == 'y':
            run_command(cmd3, "Generating responses with trained model")
    else:
        print("⚠️  No trained model found at {model_path}")
        print("   Please run Example 1 first to train a model")
    
    # Example 4: Custom training parameters
    print("\n" + "="*50)
    print("EXAMPLE 4: Custom Training Parameters")
    print("="*50)
    
    cmd4 = """python send/extended_send_multi_dataset.py \\
        --dataset truthfulqa \\
        --model_name vicuna-7b \\
        --max_samples 200 \\
        --num_epochs 3 \\
        --learning_rate 5e-5 \\
        --sensitivity_threshold 0.2 \\
        --batch_size 1 \\
        --max_length 512 \\
        --use_wandb \\
        --output_dir ./custom_output"""
    
    print("This command demonstrates:")
    print("- Custom learning rate (5e-5)")
    print("- Lower sensitivity threshold (0.2)")
    print("- More samples (200) and epochs (3)")
    print("- Wandb logging enabled")
    print("- Custom output directory")
    
    response = input("\nWould you like to run this example? (y/N): ")
    if response.lower() == 'y':
        run_command(cmd4, "Custom training parameters")
    
    print("\n" + "="*50)
    print("Example usage complete!")
    print("="*50)
    print("\nFor more information, see:")
    print("- send/README_EXTENDED.md")
    print("- send/extended_send_multi_dataset.py --help")
    print("- send/run_send_batch.py --help")
    print("- send/generate_responses.py --help")

if __name__ == "__main__":
    main()
