import json
import numpy as np
import pandas as pd

with open(f'/home/mila/j/juan.guerra/pythia-hallucination/data/hidden_layer_activations_model_30_to_43_pythia-1b_hallu.json') as f:
    data = json.load(f)

    df = pd.DataFrame(data)

    # Scaling the vectors/normalizing them
    avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

    # Compute the scaling factor
    scale = 2048 / avg_norm

    # Scale each vector by the scaling factor
    for col in df.columns[2:]:
        df[col] = df[col].apply(lambda x: scale * np.array(x))

    for w in range(2, len(df.columns) - 5):
        checkpoints = df.columns[w:w+5]

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
        print(f'\nWindow Starting at {w} best mean value {values[0]} ')
    