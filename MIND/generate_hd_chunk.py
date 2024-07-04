# Given the diff_results csv file, only generate the hidden layer activations for the given data

import os
import json
from generate_hd import *
from tqdm import tqdm
import pandas as pd

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--model_family", type=str, default="pythia")
argument_parser.add_argument("--model_name", type=str, default="1b")
# model checkpoints to be a list of strings separated by space
argument_parser.add_argument("--model_checkpoints", type=str, nargs="+")
args = argument_parser.parse_args()
model_family = args.model_family
model_name = args.model_name
model_checkpoints = args.model_checkpoints

# load the diff results
diff_results = pd.read_csv("./data/diff_results.csv", index_col=0)

titles = diff_results["title"].tolist()
models = diff_results.columns[1:-1].tolist()

# iterate over titles and for each model, generate hidden layer activations

data_map = {}
for model_name in models:
    f = json.load(open(f"./MIND/auto-labeled/output/{model_name}/data_train.json", "r"))
    f2 = json.load(open(f"./MIND/auto-labeled/output/{model_name}/data_valid.json", "r"))
    f3 = json.load(open(f"./MIND/auto-labeled/output/{model_name}/data_test.json", "r"))

    data_map[model_name] = f + f2 + f3        

data_dict = {data['title']: data for model_data in data_map.values() for data in model_data}

# Load models before entering the loop
models_dict = {}
for model_name in models:
    model_family, model_type, model_checkpoint = model_name.split("_")
    model, tokenizer, generation_config, at_id = get_model(model_type, model_family, max_new_tokens=1, model_checkpoint="step"+model_checkpoint)
    models_dict[model_name] = (model, tokenizer)

results = []
for index, title in tqdm(enumerate(titles), total=len(titles), desc="Processing titles"):
    one_instance = {}
    for model_index, model_name in enumerate(models):
        model, tokenizer = models_dict[model_name]  # Retrieve model and tokenizer from dictionary
        data = data_dict.get(title)
        if data:
            hidden = get_penultimate_layer_hd(model, data["truncated_text"], tokenizer, model_family, data["title"])
            if model_index == 0:
                one_instance["title"] = title
                one_instance["text"] = data["truncated_text"]
            one_instance[model_name] = hidden
    results.append(one_instance)

name = model_name.split("_")

with open(f"./data/pen_hidden_layer_activations_model_{name[0]}_{name[1]}_{model_checkpoints[0]}_{model_checkpoints[1]}_{model_checkpoints[2]}(101).json", "w") as f:

    json.dump(results, f)



