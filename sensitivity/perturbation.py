import cupy as cp
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import argparse
import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

"""
GPU requirements for the script: CUDA-enabled GPU with CuPy installed 80GB of memory
CPU cores: The more the better. Start with around 32 cores and increase if needed.
"""

def compute_eigenscore(embedding_matrix, alpha=0.001):
    # mean center the data
    embedding_matrix = embedding_matrix - cp.mean(embedding_matrix, axis=0)
    n = embedding_matrix.shape[0]
    _, S, _ = cp.linalg.svd(embedding_matrix, full_matrices=False)  # main source of complexity
    lambda_reg = (S**2 / (n - 1)) + alpha
    eigenscore = cp.sum(cp.log(lambda_reg))
    return eigenscore / n

def perturb_and_evaluate(embeddings, feature_index, num_perturbations=20):
    original_value = embeddings[:, feature_index].copy()
    total_change = 0.0
    original_score = compute_eigenscore(embeddings)
    
    for _ in range(num_perturbations):

        # just set the column to zero and see the effect
        perturbation = cp.zeros(embeddings.shape[0])
        
        embeddings[:, feature_index] = perturbation
        perturbed_score = compute_eigenscore(embeddings)
        
        # Calculate change in eigenscore
        change_in_score = original_score - perturbed_score
        total_change += change_in_score
        
        embeddings[:, feature_index] = original_value
        break

    return total_change

def process_task(args):
    embeddings, feature_indices, original_score = args
    effects = []
    # Convert range to list for tqdm description (efficient alternative)
    first_index = min(feature_indices)
    last_index = max(feature_indices)
    for feature_index in tqdm(feature_indices, desc=f"Process {first_index}-{last_index}"):
        perturbed_score = perturb_and_evaluate(embeddings, feature_index)
        effect = perturbed_score - original_score
        effects.append((feature_index, effect))
    return effects

def analyze_feature_effects_parallel(embeddings_np, max_gpu_processes=60):
    embeddings_gpu = cp.asarray(embeddings_np)
    original_score = compute_eigenscore(embeddings_gpu)

    # Limit the number of processes to the lesser of CPU cores or a predefined maximum
    # num_processes = min(cpu_count(), embeddings_np.shape[1], max_gpu_processes)
    num_processes = max_gpu_processes
    total_features = embeddings_np.shape[1]
    features_per_process = total_features // num_processes
    extra_features = total_features % num_processes

    feature_indices = []
    start = 0
    for i in range(num_processes):
        end = start + features_per_process + (1 if i < extra_features else 0)
        if start < end:  # Ensure the range is not empty
            feature_indices.append(range(start, end))
        start = end

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_task, [(embeddings_gpu.copy(), indices, original_score) for indices in feature_indices if indices])

    effects = [effect for sublist in results for effect in sublist]
    effects.sort(key=lambda x: x[0])  # Sort by feature index
    del embeddings_gpu, original_score
    effects_dict = {effect[0]: abs(effect[1]) for effect in effects}  # Use absolute value of change
    return effects_dict

def argsort_effect(effects):
    cp_effects = cp.array(effects)
    cp_abs_effects = cp.abs(cp_effects)
    sorted_indices = cp.argsort(cp_abs_effects)[::-1]
    np_sorted_indices = cp.asnumpy(sorted_indices)
    del cp_effects, cp_abs_effects, sorted_indices
    return np_sorted_indices

def explain_features_multiprocess(embeddings: np.ndarray) -> dict:
    """
    Analyzes the effect of perturbing each feature in the embeddings on the eigenscore of the covariance matrix.

    This function initializes a multiprocessing environment to parallelize the computation of the effect of perturbations
    on each feature across multiple processes. It is designed to work with GPU arrays for efficient computation.

    Parameters:
    - embeddings (np.ndarray): A NumPy array of shape (N, M) where N is the number of samples and M is the number of features.
                                This array contains the embeddings that will be analyzed.

    Returns:
    - dict: A dictionary containing the effectiveness of each neuron given its index in the sentence embedding.

    Note:
    - The multiprocessing start method is set to 'spawn' to ensure compatibility with CUDA operations in child processes.
    """
    set_start_method('spawn', force=True)
    effects_dict = analyze_feature_effects_parallel(embeddings)
    return effects_dict

def main(embeddings=None):
    # Example usage
    if embeddings is None:
        embeddings = np.random.rand(10, 20).astype(np.float32)
    sorted_indices = explain_features_multiprocess(embeddings)
    print(f"Sorted effectivenesses {sorted_indices}")


if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--embeddings", type=str, help="Path to embeddings file", required=False)
    args = arguments.parse_args()
    embeddings = np.load(args.embeddings)
    main()

    