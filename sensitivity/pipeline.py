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
    model_name = "EleutherAI/pythia-410m"
    # Run 10 inferences on the fully trained model
    embeddings = run_with_temperature(accelerator, model_name, INPUT)
    # start a new process for explain_features_multiprocess function with input embeddings
    # most_effective_indices_on_eigenscore = explain_features_multiprocess(embeddings)    # get the neuron effectiveness scores dict

    full_emb = extract_embeddings(accelerator, model_name, INPUT)
    SLT_emb = get_SLT_embedding(full_emb)
    net = combined_sensitivity(SLT_emb)
    top_k_percent = top_k_percent_sensitive(net, 1)  # get the top 5% most sensitive neurons as a dictionary

    most_effective_indices_on_eigenscore = explain_sensitive_vs_random(embeddings, top_k_percent)
    
    print(f"Most effective neurons on eigenscore: {most_effective_indices_on_eigenscore}")
    print(f"Most sensitive neurons: {top_k_percent}")

    # Define the softmax function
    # def softmax(values):
    #     e_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
    #     return e_values / e_values.sum()

    # # Function to apply softmax and prepare dictionary for serialization
    # def apply_softmax_and_serialize(input_dict, output_file):
    #     values = []
        
    #     for value in input_dict.values():
    #         # Convert CuPy array to NumPy array if necessary
    #         if isinstance(value, cp.ndarray):
    #             values.append(value.get())  # Use .get() to convert to NumPy
    #         else:
    #             values.append(float(value))  # Ensure it's a float
        
    #     values = np.array(values, dtype=float)
    #     softmax_values = softmax(values)
        
    #     serialized_dict = {
    #         str(key): float(value)
    #         for key, value in zip(input_dict.keys(), softmax_values)
    #     }
        
    #     with open(output_file, 'w') as f:
    #         json.dump(serialized_dict, f)

    # # Apply softmax and serialize each dictionary
    # apply_softmax_and_serialize(most_effective_indices_on_eigenscore, 'most_effective_indices_on_eigenscore_softmax.json')
    # apply_softmax_and_serialize(top_k_percent, 'most_sensitive_indices_in_training_softmax.json')


    print("Done!")
