import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio

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

# Flatten each vector and concatenate them into a single numpy array
all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
all_vectors = all_vectors.reshape(-1, 2048)
allv2 = all_vectors.reshape(3, 52, 2048)

sub = np.subtract(allv2[1], allv2[2])

fig = go.Figure()

for i in range(sub.shape[0]):
    fig.add_trace(go.Scatter(
        x=list(range(sub.shape[1])),
        y=sub[i,:],
        mode='lines+markers',
        marker=dict(
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        ),
        text=df.iloc[i,:].title,  # Add title to the trace for each data point
        name=df.iloc[i,:].title   # Add title to the legend
    ))

# Set the title and axis labels
fig.update_layout(
    title="2D plot of the difference between checkpoint 2 checkpoint 3",
    xaxis_title='Embedding',
    yaxis_title='Delta activation',
    autosize=True
)

# Show the plot
fig.show()
pio.write_image(fig, "embedding_difference_2to3.png")