from logging import StrFormatStyle
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats
from matplotlib import pyplot as plt

with open(f'./data/hidden_layer_activations_model_30_to_43.json') as f:
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
    
    variability = np.zeros(2048)
    for point in range(len(allv2)):
        for i in range(len(allv2[point]) - 1): # Want to only go until the penultimate one since we are calculating deltas
            diff = np.abs(allv2[point][i] - allv2[point][i+1])

            diff = stats.zscore(diff, axis=0)
            variability += diff
            """for i in range(len(diff)):
                if diff[i] > 2 or diff[i] < -2:
                    variability[i] += 1"""
        
        args = np.argsort(variability)[-5:]
        args = np.flip(args)
        values = variability[args]
        print(f'\nMaximum indices: {args}')
        print(f'Maximum values: {values}\n')