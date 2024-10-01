import argparse
from cProfile import label
import os
import json
from scipy.stats import pearsonr
from efficient import *

def main():
    # Read the model type
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_points', type=int)   
    args = parser.parse_args()
    num_points = args.num_points
    model_type = args.model

    # Set the input data directory
    input_data_dir = f'MIND/auto-labeled/output/pythia_{model_type}_143000'
    input_data_dir = os.path.abspath(input_data_dir)

    # Load the data
    with open(f'{input_data_dir}/data_test.json', 'r') as f:
        data = json.load(f)

    # Separate data into hallucinating and non-hallucinating
    new_entities = []
    non_new_entities = []
    for sample in data:
        if (len(sample['new_entities']) > 0) and (len(new_entities) < num_points):
            new_entities.append(sample['original_text'])
        elif (len(non_new_entities) < num_points):
            non_new_entities.append(sample['original_text'])

    # Combine the data and create labels
    all_data = new_entities + non_new_entities
    labels = [1] * len(new_entities) + [0] * len(non_new_entities)
    print(labels)

    from sklearn.metrics import roc_auc_score

    # Get ES scores for all data
    # es_scores = []
    # non_efficient_es_scores = []
    # for prompt in all_data:
    #     es_score, non_efficient_es_score = llm_experiment(prompt, model_type)
    #     es_scores.append(es_score)
    #     non_efficient_es_scores.append(non_efficient_es_score)

    # # Calculate AUROC
    # es_auroc = roc_auc_score(labels, es_scores)
    # non_efficient_es_auroc = roc_auc_score(labels, non_efficient_es_scores)

    # # Print the results
    # print(f'AUROC for ES scores: {es_auroc}')
    # print(f'AUROC for non-efficient ES scores: {non_efficient_es_auroc}')

    # # Determine which one attains a higher AUROC
    # if es_auroc > non_efficient_es_auroc:
    #     print('ES scores have a higher AUROC with the true labels.')
    # else:
    #     print('Non-efficient ES scores have a higher AUROC with the true labels.')

    corr = compute_correlation(all_data, model_type)
    print(corr, "is the correlation between the two methods")

if __name__ == "__main__":
    main()