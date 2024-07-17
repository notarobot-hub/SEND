"""
Use perturbation.py and most_sensitive.py to analyze sensitivity.
"""
from run_with_temperature import *
from perturbation import *
import json

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")
    GUIDANCE = "You are my completion assistant. Continue the following prompt and complete it based on facts:  "
    INPUT = "Obesity hypoventilation syndrome ( also known as Pickwickian syndrome ) is a condition in which severely overweight people fail to breathe rapidly enough or deeply enough , resulting in low blood oxygen levels and high blood carbon dioxide ( CO2 ) levels . Many people with this condition also frequently stop breathing altogether for short periods of time during sleep ( obstructive sleep apnea ) , resulting in many partial awakenings during the night , which leads to continual sleepiness during "

    INPUT = GUIDANCE + INPUT

    # Load the model and tokenizer
    accelerator = Accelerator()
    model_name = "EleutherAI/pythia-70m"
    # Run 10 inferences on the fully trained model
    embeddings = run_with_temperature(accelerator, model_name, INPUT)
    # start a new process for explain_features_multiprocess function with input embeddings
    most_effective_indices_on_eigenscore = explain_features_multiprocess(embeddings)    # get the neuron effectiveness scores dict

    full_emb = extract_embeddings(accelerator, model_name, INPUT)
    SLT_emb = get_SLT_embedding(full_emb)
    net = combined_sensitivity(SLT_emb)
    top_k_percent = top_k_percent_sensitive(net, 1)  # get the top 5% most sensitive neurons as a dictionary
    
    print(f"Most effective neurons on eigenscore: {most_effective_indices_on_eigenscore}")
    print(f"Top 5% most sensitive neurons: {top_k_percent}")

    # Find global min and max across both dictionaries
    all_values = list(most_effective_indices_on_eigenscore.values()) + list(top_k_percent.values())
    global_min = float(min(all_values))
    global_max = float(max(all_values))

    # Function to normalize values
    def normalize(value, min_val, max_val):
        return 1 + ((value - min_val) * (10)) / (max_val - min_val)

    # Normalize and prepare the first dictionary
    most_effective_indices_on_eigenscore_serializable = {
        key: normalize(float(value.item()), global_min, global_max)
        for key, value in most_effective_indices_on_eigenscore.items()
    }

    # Save the first dictionary
    with open('most_effective_indices_on_eigenscore.json', 'w') as f:
        json.dump(most_effective_indices_on_eigenscore_serializable, f)

    # Normalize and prepare the second dictionary
    top_k_percent_serializable = {
        str(int(key)): normalize(float(value), global_min, global_max) if isinstance(value, (np.float32, np.float64)) else normalize(value, global_min, global_max)
        for key, value in top_k_percent.items()
    }

    # Save the second dictionary
    with open('most_sensitive_indices_in_training.json', 'w') as f:
        json.dump(top_k_percent_serializable, f)

    print("Done!")
