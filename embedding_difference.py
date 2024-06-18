import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
print("hello")

with open('./data/hidden_layer_activations.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Scaling the vectors/normalizing them
avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

# Compute the scaling factor
scale = 2048 / avg_norm

# Scale each vector by the scaling factor
for col in df.columns[2:]:
    df[col] = df[col].apply(lambda x: scale * np.array(x))

values = df[['pythia_1b_107000', 'pythia_1b_125000', 'pythia_1b_143000']]

print(values.shape)
# Flatten each vector and concatenate them into a single numpy array
all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
print(all_vectors.shape)
