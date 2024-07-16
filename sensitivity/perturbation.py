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

def compute_eigenscore(covariance_matrix, alpha=0.001):
    K = covariance_matrix.shape[0]
    regularized_covariance_matrix = covariance_matrix + alpha * cp.eye(K) 
    eigenvalues = cp.linalg.eigvalsh(regularized_covariance_matrix)
    eigenscore = (1 / K) * cp.sum(cp.log(eigenvalues))
    del regularized_covariance_matrix, eigenvalues
    return eigenscore

def perturb_and_evaluate(embeddings, feature_index):
    original_value = embeddings[:, feature_index].copy()
    perturbation = cp.random.normal(0, 1, size=embeddings.shape[0])

    embeddings[:, feature_index] += perturbation
    perturbed_score = compute_eigenscore(cp.cov(embeddings, rowvar=False))

    embeddings[:, feature_index] = original_value

    del perturbation

    return perturbed_score

def process_task(args):
    embeddings, feature_indices, original_score, covariance_matrix = args
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
    covariance_matrix = cp.cov(embeddings_gpu, rowvar=False)
    original_score = compute_eigenscore(covariance_matrix)

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
        results = pool.map(process_task, [(embeddings_gpu.copy(), indices, original_score, covariance_matrix) for indices in feature_indices if indices])

    effects = [effect for sublist in results for effect in sublist]
    effects.sort(key=lambda x: x[0])  # Sort by feature index
    del embeddings_gpu, covariance_matrix, original_score
    return [effect[1] for effect in effects]

def argsort_effect(effects):
    cp_effects = cp.array(effects)
    cp_abs_effects = cp.abs(cp_effects)
    sorted_indices = cp.argsort(cp_abs_effects)[::-1]
    np_sorted_indices = cp.asnumpy(sorted_indices)
    del cp_effects, cp_abs_effects, sorted_indices
    return np_sorted_indices

def explain_features_multiprocess(embeddings: np.ndarray) -> np.ndarray:
    """
    Analyzes the effect of perturbing each feature in the embeddings on the eigenscore of the covariance matrix.

    This function initializes a multiprocessing environment to parallelize the computation of the effect of perturbations
    on each feature across multiple processes. It is designed to work with GPU arrays for efficient computation.

    Parameters:
    - embeddings (np.ndarray): A NumPy array of shape (N, M) where N is the number of samples and M is the number of features.
                                This array contains the embeddings that will be analyzed.

    Returns:
    - np.ndarray: Incrementally sorted array of feature indices based on the effect of perturbing each feature on the eigenscore.

    Note:
    - The multiprocessing start method is set to 'spawn' to ensure compatibility with CUDA operations in child processes.
    """
    set_start_method('spawn', force=True)
    effects = analyze_feature_effects_parallel(embeddings)
    return argsort_effect(effects)

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

    