from most_sensitive import *
from run_with_temperature import *

def run_with_temperature_window(accelerator, model_name, input_text, window_start, window_end, temperature=0.5, contextual_embedding_extraction_form="last_token", middle=False):
    """
    Generates the embedding matrix of 10 different outputs of inference with temperature for a sliding window of the input text.

    Parameters:
    - accelerator (str): The accelerator to use.
    - model_name (str): The model to use.
    - input_text (str): The input text.
    - window_size (int, optional): The size of the sliding window. Defaults to 5.
    - window_start (int, optional): The starting checkpoint. Defaults to 130000.
    - window_end (int, optional): The ending checkpoint. Defaults to 143000. MAX 143000. 
    - temperature (float, optional): The temperature. Defaults to 0.5.
    - contextual_embedding_extraction_form (str, optional): The form of contextual embedding extraction. Defaults to "last_token".
    - middle (bool, optional): Whether to use the middle. Defaults to True.

    Returns:
    - np.array: The embeddings for the sliding window.
    """

    return run_with_temperature(accelerator, model_name, input_text, revision=window_end*1000, temperature=temperature, contextual_embedding_extraction_form=contextual_embedding_extraction_form, middle=middle)

def extract_window_embeddings(accelerator, model_name: str, input_text: str, window_start: int=130000, window_end: int=143000):
    CHECKPOINTS = range(window_start * 1000, window_end * 1000 + 1000, 1000)

    penultimate_outputs = []
    for checkpoint in CHECKPOINTS:
        model, tokenizer = load_model(model_name, checkpoint)
        penultimate_layer = penultimate_layer_output(accelerator, model, tokenizer, input_text)
        penultimate_outputs.append(penultimate_layer)

    # make sure the outputs are of numpy arrays of shape (num_checkpoints, sequence_length, hidden_size)
    embeddings = np.concatenate([output.cpu() for output in penultimate_outputs], axis=0)
    return embeddings



