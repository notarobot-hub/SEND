import argparse
import json
import pandas as pd
import os

INPUT_DIR = "./MIND/auto-labeled/output/"
DICTIONARY_PATH = "./MIND/auto-labeled/wiki/wiki_titles.csv"
DICTIONARY = pd.read_csv(DICTIONARY_PATH)    # True means hallucinates on that data point, and False means it does not

def gather_train_test_valid(model_family, model_name, model_checkpoint) -> None:
    global DICTIONARY
    path = INPUT_DIR + model_family + "_" + model_name + "_" + model_checkpoint + "/"
    train = []
    valid = []
    test = []
    with open(path + "data_train.json", "r") as f:
        train = json.load(f)
    
    with open(path + "data_valid.json", "r") as f:
        valid = json.load(f)
    
    with open(path + "data_test.json", "r") as f:
        test = json.load(f)
    
    # merge train, valid, test
    data = train + valid + test
    
    # in the data list only take jsons with "texts" key list, length greater than zero
    data = [d for d in data if "texts" in d and len(d["texts"]) > 0]

    DICTIONARY[model_family + "_" + model_name + "_" + model_checkpoint] = DICTIONARY["title"].apply(lambda x: x in [d["title"] for d in data]).astype(int)

    subset = DICTIONARY[[model_family + "_" + model_name + "_" + model_checkpoint, "title"]]
    subset.to_csv(path + "tabular_results.csv")

def gather_all_101(model_family, model_name, *args) -> None:
    global DICTIONARY
    # iterate over args which is a list of all checkpoints and gather results
    for model_checkpoint in args:
        gather_train_test_valid(model_family, model_name, model_checkpoint+"000")

    # add a last column which is the sum of all the columns from column 1 to the last column
    DICTIONARY["sum"] = DICTIONARY.iloc[:, 1:].sum(axis=1)

    # only keep rows where sum is not 0 or len(args)
    # DICTIONARY = DICTIONARY[(DICTIONARY["sum"] != 0) & (DICTIONARY["sum"] != len(args))]
    # only keep rows where the first feature is 1 second 0 and third 1
    DICTIONARY = DICTIONARY[(DICTIONARY.iloc[:, 1] == 1) & (DICTIONARY.iloc[:, 2] == 0) & (DICTIONARY.iloc[:, 3] == 1)]

def gather_all_100(model_family, model_name, *args) -> None:
    global DICTIONARY
    # iterate over args which is a list of all checkpoints and gather results
    for model_checkpoint in args:
        gather_train_test_valid(model_family, model_name, model_checkpoint+"000")

    # add a last column which is the sum of all the columns from column 1 to the last column
    DICTIONARY["sum"] = DICTIONARY.iloc[:, 1:].sum(axis=1)

    # only keep rows where sum is not 0 or len(args)
    # DICTIONARY = DICTIONARY[(DICTIONARY["sum"] != 0) & (DICTIONARY["sum"] != len(args))]
    # only keep rows where the first feature is 1 second 0 and third 0
    DICTIONARY = DICTIONARY[(DICTIONARY.iloc[:, 1] == 1) & (DICTIONARY.iloc[:, 2] == 0) & (DICTIONARY.iloc[:, 3] == 0)]

def gather_all_001(model_family, model_name, *args) -> None:
    global DICTIONARY
    # iterate over args which is a list of all checkpoints and gather results
    for model_checkpoint in args:
        gather_train_test_valid(model_family, model_name, model_checkpoint+"000")

    # add a last column which is the sum of all the columns from column 1 to the last column
    DICTIONARY["sum"] = DICTIONARY.iloc[:, 1:].sum(axis=1)

    # only keep rows where sum is not 0 or len(args)
    # DICTIONARY = DICTIONARY[(DICTIONARY["sum"] != 0) & (DICTIONARY["sum"] != len(args))]
    # only keep rows where the first feature is 0 second 0 and third 1
    DICTIONARY = DICTIONARY[(DICTIONARY.iloc[:, 1] == 0) & (DICTIONARY.iloc[:, 2] == 0) & (DICTIONARY.iloc[:, 3] == 1)]

def gather_all_000(model_family, model_name, *args) -> None:
    global DICTIONARY
    # iterate over args which is a list of all checkpoints and gather results
    for model_checkpoint in args:
        gather_train_test_valid(model_family, model_name, model_checkpoint+"000")

    # add a last column which is the sum of all the columns from column 1 to the last column
    DICTIONARY["sum"] = DICTIONARY.iloc[:, 1:].sum(axis=1)

    # only keep rows where sum is not 0 or len(args)
    # DICTIONARY = DICTIONARY[(DICTIONARY["sum"] != 0) & (DICTIONARY["sum"] != len(args))]
    # only keep rows where the first feature is 0 second 0 and third 1
    DICTIONARY = DICTIONARY[(DICTIONARY.iloc[:, 1] == 0) & (DICTIONARY.iloc[:, 2] == 0) & (DICTIONARY.iloc[:, 3] == 0)]


def gather_all(model_family, model_name, *args) -> None:
    global DICTIONARY
    # iterate over args which is a list of all checkpoints and gather results
    for model_checkpoint in args:
        gather_train_test_valid(model_family, model_name, model_checkpoint+"000")

    # add a last column which is the sum of all the columns from column 1 to the last column
    DICTIONARY["sum"] = DICTIONARY.iloc[:, 1:].sum(axis=1)

    # only keep rows where sum is not 0 or len(args)
    # DICTIONARY = DICTIONARY[(DICTIONARY["sum"] != 0) & (DICTIONARY["sum"] != len(args))]
    # only keep rows where the first feature is 0 second 0 and third 1

def gather_all_0(model_family, model_name, *args) -> None:
    global DICTIONARY
    # iterate over args which is a list of all checkpoints and gather results
    for model_checkpoint in args:
        gather_train_test_valid(model_family, model_name, model_checkpoint+"000")

    # add a last column which is the sum of all the columns from column 1 to the last column
    DICTIONARY["sum"] = DICTIONARY.iloc[:, 1:].sum(axis=1)

    # only keep rows where sum is not 0 or len(args)
    # DICTIONARY = DICTIONARY[(DICTIONARY["sum"] != 0) & (DICTIONARY["sum"] != len(args))]
    # only keep rows where the first feature is 0 second 0 and third 1

    DICTIONARY = DICTIONARY[DICTIONARY.iloc[:, -1] ==0] # Ensure that all the things are non hallucinating

def gather_all_1(model_family, model_name, *args) -> None:
    global DICTIONARY
    # iterate over args which is a list of all checkpoints and gather results
    for model_checkpoint in args:
        gather_train_test_valid(model_family, model_name, model_checkpoint+"000")

    # add a last column which is the sum of all the columns from column 1 to the last column
    DICTIONARY["sum"] = DICTIONARY.iloc[:, 1:].sum(axis=1)

    # only keep rows where sum is not 0 or len(args)
    # DICTIONARY = DICTIONARY[(DICTIONARY["sum"] != 0) & (DICTIONARY["sum"] != len(args))]
    # only keep rows where the first feature is 0 second 0 and third 1

    DICTIONARY = DICTIONARY[DICTIONARY.iloc[:, -1] == len(DICTIONARY.iloc[0, 1:-1])] # Ensure that all the things are non hallucinating

    

def main():

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_family", type=str, default="pythia")
    argument_parser.add_argument("--model_name", type=str, default="1b")
    # model checkpoints to be a list of strings separated by space
    argument_parser.add_argument("--model_checkpoints", type=str, nargs="+")
    argument_parser.add_argument("--hallu_type", type=str, default="9")
    args = argument_parser.parse_args()
    model_family = args.model_family
    model_name = args.model_name
    model_checkpoints = args.model_checkpoints
    hallu_type = args.hallu_type
    if hallu_type == "101":
        gather_all_101(model_family, model_name, *model_checkpoints)
    elif hallu_type == "100":
        gather_all_100(model_family, model_name, *model_checkpoints)
    elif hallu_type == "001":
        gather_all_001(model_family, model_name, *model_checkpoints)
    elif hallu_type == "000":
        gather_all_000(model_family, model_name, *model_checkpoints)
    elif hallu_type == "9":
        gather_all(model_family, model_name, *model_checkpoints)
    elif hallu_type == "0":
        gather_all_0(model_family, model_name, *model_checkpoints)
    elif hallu_type == "1":
        gather_all_1(model_family, model_name, *model_checkpoints)
        
    os.makedirs("./data", exist_ok=True)
    DICTIONARY.to_csv("./data/diff_results.csv")

if __name__ == "__main__":
    main()