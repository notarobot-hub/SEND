from pipeline import *
from sliding_window_sensitivity import *

def prepare_data_dir_window(mode: str, model_name: str, checkpoints: list[int]):
    if mode == "hallu":
        mode = "1"
    else:
        mode = "0"

    output_dirs = [f"./MIND/auto-labeled/output/pythia_{model_name}_{checkpoint}000" for checkpoint in checkpoints]

    for output_dir in output_dirs:
        checkpoint = output_dir.split('_')[-1][:-3]

        if not os.path.exists(output_dir):
            os.system(f"python ./MIND/generate_data.py --model_type {model_name} --model_family pythia --step_num {checkpoint}")

        with open(f"{output_dir}/data_test.json", 'r') as f:
            test_data = json.load(f)
        with open(f"{output_dir}/data_train.json", 'r') as f:
            train_data = json.load(f)
        with open(f"{output_dir}/data_valid.json", 'r') as f:
            valid_data = json.load(f)

        # Combine all data
        data = test_data + train_data + valid_data

        del test_data, train_data, valid_data

        # Filter data points where the feature matches the mode
        filtered_data = []
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
        output_dir = f"data/sensitivity/{model_name}/{checkpoint}"
        os.makedirs(output_dir, exist_ok=True)

        # Write the selected data points to files
        for point in selected_data:
            with open(f"{output_dir}/{mode}_{point['id']}.json", 'w') as f:
                json.dump(point, f)

    return

def get_metadata_single_point_window(id: int, mode: str, model_name: str, checkpoint: int):
    """
    Retrieve metadata for a single data point across all checkpoints.

    Reads JSON files to extract the text associated with the given id, mode, and model name.

    Parameters:
    id (int): The identifier for the data point.
    mode (str): The mode in which the function is operating.
    model_name (str): The name of the model.
    checkpoints (int):  Checkpoint to search for.

    Returns:
    tuple: A tuple containing id, text, mode, model_name, and checkpoint.
    """

    file_path = f"data/sensitivity/{model_name}/{checkpoint}/{mode}_{id}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
        text = data["truncated_text"]
    return id, text, mode, model_name, checkpoint


def get_sensitivity_for_single_point_window(id: int, input_text: str, mode: str, model_name: str = "410m", window_start: int = 130000, window_end: int = 143000, GUIDANCE: str = GUIDANCE):
    """
    Calculate and save the sensitivity of a single point in a neural network model across multiple checkpoints.

    This function takes an input text and processes it through specified model checkpoints to calculate the sensitivity of neurons.
    The results are saved in JSON files with the format: {"id": id, "original_ES": orig, "sensitive_drop": sens_drop, "random_drop_avg": random_drop_avg}.

    Parameters:
    id (int): The identifier for the data point.
    input_text (str): The input text to be processed.
    mode (str): The mode in which the function is operating (used in the filename).
    GUIDANCE (str, optional): Guidance text to prepend to the input text. Defaults to GUIDANCE.
    model_name (str, optional): The name of the model to use. Defaults to "410m".
    window_start (int, optional): The starting checkpoint. Defaults to 130000.
    window_end (int, optional): The ending checkpoint. Defaults to 143000.

    Returns:
    None
    """
    accelerator = Accelerator()
    input = GUIDANCE + input_text
    model_name_base = f"EleutherAI/pythia-{model_name}"

    # Extract embeddings using the sliding window approach
    embeddings = run_with_temperature_window(accelerator, model_name_base, input, window_start, window_end)
    full_emb = extract_window_embeddings(accelerator, model_name_base, input, window_start, window_end)
    SLT_emb = get_SLT_embedding(full_emb)
    net = combined_sensitivity(SLT_emb)
    top_k_percent = top_k_percent_sensitive(net, 1)  # get the neurons as a dictionary with their sensitivity values
    most_effective_indices_on_eigenscore = explain_sensitive_vs_random(embeddings, top_k_percent)  # get the most effective neurons on eigenscore
    orig, sens_drop, random_drop_avg = most_effective_indices_on_eigenscore

    if mode == "0":
        mode = "nonhallu"
    else:
        mode = "hallu"

    directory = f"data/sensitivity/{model_name_base}/{window_start}-{window_end}"
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

    if function is not None:
        if function == "prepare_data_dir":
            checkpoints = [131, 135, 139, 143]
            prepare_data_dir_window("hallu", model_name, checkpoints)
            prepare_data_dir_window("nonhallu", model_name, checkpoints)
            print("Done preparing the data for model: ", model_name)
            exit(0)
        elif function == "get_sensitivity_for_single_point":
            # Define the window range
            window_start = 131000
            window_end = 143000
            window_size = (window_end - window_start) // 3

            # Divide the window into 3 sub-windows
            sub_windows = [(window_start + i * window_size, window_start + (i + 1) * window_size) for i in range(3)]

            for start, end in sub_windows:
                # take the last 000 out of the start and end variables
                start = int(start / 1000)
                end = int(end / 1000)
                id, text, mode, model_name, checkpoint = get_metadata_single_point_window(data_id, "0", model_name, end)
                get_sensitivity_for_single_point_window(id, text, mode, model_name, start, end)

                id, text, mode, model_name, checkpoint = get_metadata_single_point_window(data_id, "1", model_name, end)
                get_sensitivity_for_single_point_window(id, text, mode, model_name, start, end)

            print(f"Done calculating sensitivity for data point: {data_id} on model {model_name}")
            exit(0)
    else:
        print("Invalid function. Exiting.")
        exit(1)