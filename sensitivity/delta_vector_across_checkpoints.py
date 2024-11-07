from logging import StrFormatStyle
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats
from matplotlib import pyplot as plt

def efficient_extract_sensitive(model_name):

    with open(f'.././data/hidden_layer_activations_model_30_to_43_{model_name}_hallu.json') as f:
        data = json.load(f)

        df = pd.DataFrame(data)

        # Scaling the vectors/normalizing them
        avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

        # Compute the scaling factor
        scale = 2048 / avg_norm

        # Scale each vector by the scaling factor
        for col in df.columns[2:]:
            df[col] = df[col].apply(lambda x: scale * np.array(x))

        checkpoints = df.columns[2:]

        values = df[checkpoints]
        

        # Flatten each vector and concatenate them into a single numpy array
        all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
        size = all_vectors.size

        allv2 = all_vectors.reshape(int(size/(2048*len(checkpoints))), len(checkpoints), 2048) # there are 14 models
        
        variability = np.sum(np.abs(np.diff(allv2, axis=0)), axis=0) 
        var_2 = np.var(allv2, axis=0)
        all_var = variability * var_2
    
        mean_var = np.mean(all_var, axis=0)
        args = np.argsort(mean_var)
        args = np.flip(args)
        values = mean_var[args]


        index_dict = {}

        for i, index in enumerate(args):
            index_dict[index] = values[i]
        
    return index_dict

def efficient_extract_sensitive_with_df(model_name, df):
    
    df = df.copy(deep=True)

    # Scaling the vectors/normalizing them
    avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

    # Compute the scaling factor
    scale = 2048 / avg_norm

    # Scale each vector by the scaling factor
    for col in df.columns:
        df[col] = df[col].apply(lambda x: scale * np.array(x))


    values = df
    

    # Flatten each vector and concatenate them into a single numpy array
    all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
    size = all_vectors.size

    allv2 = all_vectors.reshape(int(size/(2048*len(checkpoints))), len(checkpoints), 2048) # there are 14 models
    
    variability = np.sum(np.abs(np.diff(allv2, axis=0)), axis=0) 
    var_2 = np.var(allv2, axis=0)
    all_var = variability * var_2

    mean_var = np.mean(all_var, axis=0)
    args = np.argsort(mean_var)
    args = np.flip(args)
    values = mean_var[args]


    index_dict = {}

    for i, index in enumerate(args):
            index_dict[index] = values[i]
        
    return index_dict

def efficient_extract_sensitive_window(df, win_index, win_size):
    df = df.copy(deep=True)
     # Scaling the vectors/normalizing them
    avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

    # Compute the scaling factor
    scale = 2048 / avg_norm

    # Scale each vector by the scaling factor
    for col in df.columns[2:]:
        df[col] = df[col].apply(lambda x: scale * np.array(x))
    
    checkpoints = df.columns[win_index + 2:win_index+ 7]

    values = df[checkpoints]
    

    # Flatten each vector and concatenate them into a single numpy array
    all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
    size = all_vectors.size

    allv2 = all_vectors.reshape(int(size/(2048*len(checkpoints))), len(checkpoints), 2048) # there are 14 models
    
    variability = np.sum(np.abs(np.diff(allv2, axis=0)), axis=0) 
    var_2 = np.var(allv2, axis=0)
    all_var = variability * var_2

    mean_var = np.mean(all_var, axis=0)
    args = np.argsort(mean_var)
    args = np.flip(args)
    values = mean_var[args]

    print(f'\nWindow Starting at {win_index} best mean value {values[0]} ')
    index_dict = {}

    for i, index in enumerate(args):
            index_dict[index] = values[i]
        
    return index_dict


if __name__ == '__main__':
    
    with open(f'/home/mila/j/juan.guerra/pythia-hallucination/data/hidden_layer_activations_model_30_to_43_pythia-1b_hallu.json') as f:
        data = json.load(f)

        df = pd.DataFrame(data)

        index = efficient_extract_sensitive_window(df, 0, 5)
        print(index)