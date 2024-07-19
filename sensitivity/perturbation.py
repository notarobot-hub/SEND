import random
import cupy as cp
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
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

def drop_sensitive_neurons(embeddings, sensitive_dict, top_k_num=10):
    """
    Drops the top k sensitive neurons and returns the change in eigenscore.

    Parameters:
    - embeddings (cp.array): The original embeddings array.
    - sensitive_dict (dict): A dictionary mapping neuron indices to their sensitivity values.
    - top_k_num (int, optional): Number of top sensitive neurons to drop. Defaults to 10.

    Returns:
    - float: The change in eigenscore after dropping the top k sensitive neurons
    """
    modified_embeddings = embeddings.copy()
    top_k_indices = cp.array(list(sensitive_dict.keys())[:top_k_num])
    for index in top_k_indices:
        modified_embeddings[:, index] = 0

    original_score = compute_eigenscore(embeddings)
    modified_score = compute_eigenscore(modified_embeddings)
    modified_specific = modified_score - original_score

    return modified_specific

def drop_compare_to_average(embeddings, neuron_indices, num_perturbations=20):
    """
    Drops random neurons and compares the average change in eigenscore to the specific perturbation.

    Parameters:
    - embeddings (cp.array): The original embeddings array.
    - neuron_indices (list): Indices of neurons to drop in the specific perturbation.
    - num_perturbations (int, optional): Number of random perturbations for calculating the average change. Defaults to 10.

    Returns:
    - float: The average change in eigenscore after dropping random neurons.
    """
    original_score = compute_eigenscore(embeddings)

    modified_scores = []
    for _ in range(num_perturbations):
        all_indices = cp.array((range(embeddings.shape[1])))
        random_indices = cp.random.choice(all_indices, size=len(neuron_indices), replace=False)
        modified_embeddings = cp.array(embeddings, copy=True)
        for index in random_indices:
            modified_embeddings[:, index] = 0

        modified_score = compute_eigenscore(modified_embeddings)
        modified_scores.append(modified_score - original_score)

    modified_scores = cp.array(modified_scores)
    modified_average = cp.mean(modified_scores)
    return modified_average

def drop_comparison_task(args):
    embeddings, sensitive_dict, top_k_num = args
    # given the dictionary of sensitive neurons mapping index to value, get the top k sensitive neurons
    sensitive_diff = drop_sensitive_neurons(embeddings, sensitive_dict, top_k_num)
    random_diff = drop_compare_to_average(embeddings, list(sensitive_dict.keys()), num_perturbations=20)
    return sensitive_diff, random_diff

def analyze_sensitive_drop_parallel(embeddings, sensitive_dict, top_k_num=10, max_gpu_processes=60):

    
    # Preparing data for parallel processing
    embeddings_gpu = cp.array(embeddings)
    sensitive_drop_effect = drop_sensitive_neurons(embeddings_gpu, sensitive_dict, top_k_num)
    num_processes = min(max_gpu_processes, embeddings.shape[1])
    total_features = embeddings.shape[1]
    features_per_process = total_features // num_processes
    extra_features = total_features % num_processes

    feature_indices = []
    start = 0
    for i in range(num_processes):
        end = start + features_per_process + (1 if i < extra_features else 0)
        if start < end:  # Ensure the range is not empty
            feature_indices.append((embeddings_gpu.copy(), list(range(start, end)), top_k_num))
        start = end

    # Step 2: In parallel, call drop_compare_to_average for each set of feature indices
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(drop_compare_to_average, feature_indices)

    average_of_averages = cp.mean(cp.array(results))
    
    # Cleanup
    del embeddings_gpu
    
    # Return the effect of dropping sensitive neurons and the average of average effects
    return sensitive_drop_effect, average_of_averages

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

def explain_sensitive_vs_random(embeddings: np.ndarray, sensitive_dict: dict, top_k_num: int = 10) -> tuple:
    set_start_method('spawn', force=True)
    sensitive_diff, random_diff = analyze_sensitive_drop_parallel(embeddings, sensitive_dict, top_k_num)
    return sensitive_diff, random_diff

def main(embeddings=None):
    set_start_method('spawn', force=True)

    # Example usage
    if embeddings is None:
        embeddings = np.random.rand(10,128).astype(np.float32)

    # create a generic dict of 128 neurons and their sensitivity values between 0 and 1
    sensitive_dict = {i: random.random() for i in range(128)}
    top_k_num = 10

    # Drop the top k sensitive neurons and compare to random neuron drop
    sensitive_diff, random_diff = analyze_sensitive_drop_parallel(embeddings, sensitive_dict, top_k_num)
    print(f"Sensitive drop effect: {sensitive_diff}")
    print(f"Random drop effect: {random_diff}")


if __name__ == '__main__':
    main()

    