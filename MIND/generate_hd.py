import os
import argparse
import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer
from tqdm import tqdm
import json

def prompt_chat(title):
    return [{"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer: "}]


def get_tokenized_ids(otext, tokenizer, model_family, title=None):
    text = otext.replace("@", "").replace("  ", " ").replace("  ", " ")
    text = tokenizer.decode(tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]).replace("<s>", "").replace("</s>", "")
    if model_family == "vicuna":
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: {text}"
    if "chat" in model_family:
        return chat_change_with_answer(prompt_chat(title), text.strip(), tokenizer)
    return tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()


def get_middle_layer_hd(model, text, tokenizer, model_family, title=None):
    # print(f"Text: {text}")
    # Tokenize the input text
    ids = get_tokenized_ids(text, tokenizer, title)
    # print(f"ids: {len(ids[0])}")
    # print(f"ids shape: {len(ids)}")
    hd = model(torch.tensor(ids).to(model.device), output_hidden_states=True).hidden_states

    # Get the residual dimension from the model configuration
    residual_dim = model.config.hidden_size

    # select the penultimate layer of ther LLM
    middle_layer_index = len(hd) - 2
    hds = hd[middle_layer_index].clone().detach()  # Shape: [batch_size, seq_length, hidden_size]

    # Calculate the sum of all token's activation vectors in the middle layer
    sum_hds = torch.sum(hds, dim=1)  # Shape: [batch_size, hidden_size]

    # Get the last token's activation vector
    h_n = hds[:, -1, :]  # Shape: [batch_size, hidden_size]

    # Calculate the final representation as per the formula
    final_representation = 0.5 * ((1 / hds.shape[1]) * sum_hds + h_n)  # Shape: [batch_size, hidden_size]

    # print(f"Final representation shape: {final_representation.shape}")
    # print(f"Final representation: {final_representation}")

    return final_representation.tolist()
            
if __name__ == "__main__":

    # --------------------------------------------- #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    argument_parser = argparse.ArgumentParser() 
    argument_parser.add_argument("--model_type", type=str, default="1b")
    argument_parser.add_argument("--model_family", type=str, default="pythia")
    argument_parser.add_argument("--step_num", type=int, default=1)
    argument_parser.add_argument("--wiki_path", type=str, default="./MIND/auto-labeled/wiki")
    argument_parser.add_argument("--output_path", type=str, default="./MIND/auto-labeled/output")

    args = argument_parser.parse_args()
    model_type = args.model_type
    model_family = args.model_family
    step_num = args.step_num
    wiki_path = args.wiki_path
    result_path = args.output_path + f"/{model_family}_{model_type}_{step_num}000"
    model_checkpoint = f"step{step_num}000"
    # --------------------------------------------- #
    model, tokenizer, generation_config, at_id = get_model(model_type, model_family, max_new_tokens=1, model_checkpoint=model_checkpoint)
    
    # middle layer activation for the original text
    for data_type in ["train", "valid", "test"]:
        # Load the dataset
        data = json.load(open(f"{result_path}/data_{data_type}.json", encoding='utf-8'))

        # Initialize the results list
        results = []

        for k in tqdm(data):
            if len(k['texts']) == 0:
                hallu = False
            else:
                hallu = True

            hds_normalized = get_middle_layer_hd(model, k["truncated_text"], tokenizer, model_family, k["title"])

            # Append the title, activation and hallu to the results list
            results.append({
                "title": k["title"],
                "activation": hds_normalized,
                "hallu": hallu,
            })

        # Ensure the directory exists
        os.makedirs(os.path.dirname(f"{result_path}/hidden/"), exist_ok=True)

        # Save the results to a file
        with open(f"{result_path}/hidden/normalized_hidden_states_{data_type}.json", "w+") as f:
            json.dump(results, f)