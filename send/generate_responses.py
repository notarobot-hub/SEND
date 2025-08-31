import argparse
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd


def load_model_and_tokenizer(model_path):
    """Load a trained SEND model and tokenizer"""
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to('cpu')
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """Generate a response using the SEND model"""
    inputs = tokenizer(prompt, return_tensors='pt')
    
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(prompt):].strip()
    return response


def load_test_questions(dataset_name, num_samples=10):
    """Load test questions from different datasets"""
    from datasets import load_dataset
    
    print(f"Loading test questions from {dataset_name}")
    
    if dataset_name.lower() == "triviaqa":
        dataset = load_dataset("trivia_qa", "rc.nocontext", split='validation')
        questions = []
        for item in dataset[:num_samples]:
            questions.append({
                'question': item['question'],
                'reference_answer': item['answer']['value']
            })
    
    elif dataset_name.lower() == "truthfulqa":
        dataset = load_dataset("truthful_qa", "generation", split='validation')
        questions = []
        for item in dataset[:num_samples]:
            questions.append({
                'question': item['question'],
                'reference_answer': item['best_answer']
            })
    
    elif dataset_name.lower() == "coqa":
        dataset = load_dataset("coqa", split='validation')
        questions = []
        for item in dataset[:num_samples]:
            questions.append({
                'question': item['question'],
                'reference_answer': item['answer']
            })
    
    elif dataset_name.lower() == "tydiqa":
        dataset = load_dataset("tydiqa", "primary_task", split='validation')
        questions = []
        for item in dataset[:num_samples]:
            questions.append({
                'question': item['question_text'],
                'reference_answer': item['annotations']['minimal_answers'][0]['text'] if item['annotations']['minimal_answers'] else "No answer provided"
            })
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Loaded {len(questions)} test questions")
    return questions


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_path", type=str, required=True,
                               help="Path to the trained SEND model")
    argument_parser.add_argument("--dataset", type=str, default="triviaqa",
                               choices=["triviaqa", "truthfulqa", "coqa", "tydiqa"])
    argument_parser.add_argument("--num_samples", type=int, default=10,
                               help="Number of test questions to generate responses for")
    argument_parser.add_argument("--max_length", type=int, default=100,
                               help="Maximum length of generated response")
    argument_parser.add_argument("--temperature", type=float, default=0.7,
                               help="Temperature for generation")
    argument_parser.add_argument("--top_p", type=float, default=0.9,
                               help="Top-p for generation")
    argument_parser.add_argument("--output_file", type=str, default=None,
                               help="Output file to save results (optional)")
    
    args = argument_parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load test questions
    try:
        test_questions = load_test_questions(args.dataset, args.num_samples)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using sample questions...")
        test_questions = [
            {'question': 'What is the capital of France?', 'reference_answer': 'Paris'},
            {'question': 'Who wrote Romeo and Juliet?', 'reference_answer': 'William Shakespeare'},
            {'question': 'What is 2 + 2?', 'reference_answer': '4'},
            {'question': 'What is the largest planet in our solar system?', 'reference_answer': 'Jupiter'},
            {'question': 'Who painted the Mona Lisa?', 'reference_answer': 'Leonardo da Vinci'}
        ]
    
    # Generate responses
    results = []
    print("\nGenerating responses...")
    
    for i, qa in enumerate(tqdm(test_questions, desc="Generating responses")):
        question = qa['question']
        reference_answer = qa['reference_answer']
        
        # Create prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Generate response
        try:
            response = generate_response(
                model, tokenizer, prompt, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
        except Exception as e:
            print(f"Error generating response for question {i+1}: {e}")
            response = "Error generating response"
        
        result = {
            'question_id': i + 1,
            'question': question,
            'reference_answer': reference_answer,
            'generated_response': response,
            'prompt': prompt
        }
        results.append(result)
        
        # Print results
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {question}")
        print(f"Reference A: {reference_answer}")
        print(f"Generated A: {response}")
        print("-" * 50)
    
    # Save results if output file specified
    if args.output_file:
        output_path = args.output_file
        if not output_path.endswith('.json'):
            output_path += '.json'
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'model_path': args.model_path,
                'dataset': args.dataset,
                'generation_params': {
                    'max_length': args.max_length,
                    'temperature': args.temperature,
                    'top_p': args.top_p
                },
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
