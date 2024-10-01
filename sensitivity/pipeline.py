"""
Use perturbation.py and most_sensitive.py to analyze sensitivity.
"""
from multiprocessing.spawn import prepare
from run_with_temperature import *
from perturbation import *
import json
import os
import argparse

GUIDANCE = "You are my completion assistant. Continue the following prompt and complete it based on facts:  "

# create a pipeline from input text to output json
def get_sensitivity_for_single_point(id: int, input_text: str, mode:str, model_name: str = "410m", GUIDANCE: str=GUIDANCE):
    """
    Calculate and save the sensitivity of a single point in a neural network model.

    This function takes an input text and processes it through a specified model to calculate the sensitivity of neurons.
    The results are saved in a JSON file with the format: {"id": id, "original_ES": orig, "sensitive_drop": sens_drop, "random_drop_avg": random_drop_avg}.

    Parameters:
    id (int): The identifier for the data point.
    input_text (str): The input text to be processed.
    mode (str): The mode in which the function is operating (used in the filename).
    GUIDANCE (str, optional): Guidance text to prepend to the input text. Defaults to GUIDANCE.
    model_name (str, optional): The name of the model to use. Defaults to "410m".

    Returns:
    None
    """
    accelerator = Accelerator()
    input = GUIDANCE + input_text
    model_name = f"EleutherAI/pythia-{model_name}"
    embeddings = run_with_temperature(accelerator, model_name, input)
    full_emb = extract_embeddings(accelerator, model_name, input)
    SLT_emb = get_SLT_embedding(full_emb)
    net = combined_sensitivity(SLT_emb)
    top_k_percent = top_k_percent_sensitive(net, 1) # get the neurons as a dictionary with their sensitivity values
    most_effective_indices_on_eigenscore = explain_sensitive_vs_random(embeddings, top_k_percent) # get the most effective neurons on eigenscore
    orig, sens_drop, random_drop_avg = most_effective_indices_on_eigenscore

    if mode == "0":
        mode = "nonhallu"
    else: 
        mode = "hallu"

    directory = f"data/sensitivity/{model_name}"
    os.makedirs(directory, exist_ok=True)

    # print the types for debugging
    print(f"ID: {id}, orig: {type(orig)}, sens_drop: {type(sens_drop)}, random_drop_avg: {type(random_drop_avg)}")
    print(f"ID: {id}, orig: {orig}, sens_drop: {sens_drop}, random_drop_avg: {random_drop_avg}")

    # Convert NumPy arrays to floats if they contain a single element
    def to_float_if_single_element(array):
        if isinstance(array, cp.ndarray) and array.size == 1:
            return array.item()
        return array

    data = {
        "id": id,
        "original_ES": to_float_if_single_element(orig),
        "sensitive_drop": to_float_if_single_element(sens_drop),
        "random_drop_avg": to_float_if_single_element(random_drop_avg),
        "mode": mode
    }

    # Write the file
    with open(f"{directory}/{mode}_{id}_sensitivity.json", 'w') as f:
        json.dump(data, f)

    return

def get_metadata_single_point(id: int, mode: str, model_name: str):
    """
    Retrieve metadata for a single data point.

    Reads a JSON file to extract the text associated with the given id, mode, and model name.

    Parameters:
    id (int): The identifier for the data point.
    mode (str): The mode in which the function is operating.
    model_name (str): The name of the model.

    Returns:
    tuple: A tuple containing id, text, mode, and model_name.
    """
    with open(f"data/sensitivity/{model_name}/{mode}_{id}.json", 'r') as f:
        data = json.load(f)
        text = data["truncated_text"]
    
    return id, text, mode, model_name

def prepare_data_dir(mode: str, model_name: str):
    """
    Prepares the data directory by filtering and selecting data points based on the mode and model name. Adds 20 points per mode as json.

    Args:
        mode (str): "hallu" to select points with non-empty "texts", otherwise selects points with empty "texts".
        model_name (str): The name of the model to be used.

    Returns:
        None
    
    json file created is in data/sensitivity/{model_name}/{mode}_{id}.json with truncated_text as the input to the LLM
    """
    if mode == "hallu":
        mode = "1"
    else:
        mode = "0"

    output_dir = f"./MIND/auto-labeled/output/pythia_{model_name}_143000"
    if not os.path.exists(output_dir):
        os.system(f"python ./MIND/generate_data.py --model_type {model_name} --model_family pythia --step_num 143")

    with open(f"MIND/auto-labeled/output/pythia_{model_name}_143000/data_test.json", 'r') as f:
        test_data = json.load(f)
    with open(f"MIND/auto-labeled/output/pythia_{model_name}_143000/data_train.json", 'r') as f:
        train_data = json.load(f)
    with open(f"MIND/auto-labeled/output/pythia_{model_name}_143000/data_valid.json", 'r') as f:
        valid_data = json.load(f)

    # Combine all data
    data = test_data + train_data + valid_data

    del test_data, train_data, valid_data

    # Filter data points where the feature matches the mode
    filtered_data =[]
    if mode == "0":
        filtered_data = [point for point in data if len(point["texts"]) == 0]

    else:
        filtered_data = [point for point in data if len(point["texts"]) > 0]

    # Randomly select 50 points from the filtered data
    selected_data = random.sample(filtered_data, min(20, len(filtered_data)))

    # Assign unique IDs to the selected data points
    for i, point in enumerate(selected_data):
        point["id"] = i

    print(f"Number of data points for mode: {mode} and model: {model_name} is : {len(data)} but selected {len(selected_data)} points that have hallu: {mode}")

    # Ensure the directory exists
    output_dir = f"data/sensitivity/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Write the selected data points to files
    for point in selected_data:
        with open(f"{output_dir}/{mode}_{point['id']}.json", 'w') as f:
            json.dump(point, f)

    return


if __name__ == '__main__':

    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")
    parser = argparse.ArgumentParser(description="Run the sensitivity pipeline.")
    parser.add_argument("--model_name", type=str, help="The identifier for the model size.")
    parser.add_argument("--function", type=str, help="The function to run.", required=False)
    parser.add_argument("--data_id", type=int, help="The identifier for the data point. Starting from 0", required=False)

    args = parser.parse_args()
    model="EleutherAI"
    function=args.function
    model_name = args.model_name
    data_id = args.data_id
    if (function is not None):
        if function == "prepare_data_dir":
            prepare_data_dir("hallu", model_name)
            prepare_data_dir("nonhallu", model_name)
            print("Done preparing the data for model: ", model_name)
            exit(0)
        elif (function == "get_sensitivity_for_single_point"):
            id, text, mode, model_name = get_metadata_single_point(data_id, "0", model_name)
            get_sensitivity_for_single_point(id, text, mode, model_name)

            id, text, mode, model_name = get_metadata_single_point(data_id, "1", model_name)
            get_sensitivity_for_single_point(id, text, mode, model_name)

            print(f"Done calculating sensitivity for data point: {data_id} on model {model_name}")
            exit(0)
        
    else:
        print("Invalid function. Exiting.")
        exit(1)
    

    # INPUT = "Obesity hypoventilation syndrome ( also known as Pickwickian syndrome ) is a condition in which severely overweight people fail to breathe rapidly enough or deeply enough , resulting in low blood oxygen levels and high blood carbon dioxide ( CO2 ) levels . Many people with this condition also frequently stop breathing altogether for short periods of time during sleep ( obstructive sleep apnea ) , resulting in many partial awakenings during the night , which leads to continual sleepiness during "

    # INPUT = GUIDANCE + INPUT

    # Load the model and tokenizer
    # accelerator = Accelerator()
    # model_name = "EleutherAI/pythia-410m"
    # Run 10 inferences on the fully trained model
    # embeddings = run_with_temperature(accelerator, model_name, INPUT)
    # # start a new process for explain_features_multiprocess function with input embeddings
    # # most_effective_indices_on_eigenscore = explain_features_multiprocess(embeddings)    # get the neuron effectiveness scores dict

    # full_emb = extract_embeddings(accelerator, model_name, INPUT)
    # SLT_emb = get_SLT_embedding(full_emb)
    # net = combined_sensitivity(SLT_emb)
    # top_k_percent = top_k_percent_sensitive(net, 1)  # get the top 5% most sensitive neurons as a dictionary

    # most_effective_indices_on_eigenscore = explain_sensitive_vs_random(embeddings, top_k_percent)
    
    # print(f"Most effective neurons on eigenscore: {most_effective_indices_on_eigenscore}")
    # print(f"Most sensitive neurons: {top_k_percent}")

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
