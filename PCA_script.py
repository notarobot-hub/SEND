import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import json
import plotly.graph_objects as go
import plotly.io as pio
import gzip
#!source .env

with open('./data/hidden_layer_activations.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Compute the current average L2 norm
avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

# Compute the scaling factor
scale = 2048 / avg_norm

# Scale each vector by the scaling factor
for col in df.columns[2:]:
    df[col] = df[col].apply(lambda x: scale * np.array(x))

values = df[['pythia_1b_107000', 'pythia_1b_125000', 'pythia_1b_143000']]
# Flatten each vector and concatenate them into a single numpy array
all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
print(all_vectors.shape)
# Reshape the array to have a shape of (52*3, 2048)
all_vectors = all_vectors.reshape(-1, 2048)
def pca_graph(all_vectors):
    # Perform PCA to 2 dim
    pca = PCA(n_components=2)
    pca.fit(all_vectors)
    pca_vectors = pca.transform(all_vectors)
    pca_vectors = pca_vectors.reshape(3, 52, 2)


    # Create a 2D scatter plot
    fig = go.Figure()

    # Define colors for each checkpoint
    colors = ['red', 'green', 'blue']

    # Add a line for each data point
    for i in range(pca_vectors.shape[1]):
        fig.add_trace(go.Scatter(
            x=pca_vectors[:, i, 0],
            y=pca_vectors[:, i, 1],
            mode='lines+markers',
            marker=dict(
                color=colors,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            ),
            text=df.iloc[i,:].title,  # Add title to the trace for each data point
            name=df.iloc[i,:].title   # Add title to the legend
        ))

    # Set the title and axis labels
    fig.update_layout(
        title="2D plot of the PCA vectors for each checkpoint activation with the pattern hallucinating -> non hallucinating -> hallucinating",
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        autosize=False,
        width=3000,  # Set the width of the plot
        height=800,  # Set the height of the plot
    )

    # Show the plot
    fig.show()


# Step 4: Function to compress data and return the size
def compress_and_get_size(data):
    # Convert the numpy array to a byte array
    byte_data = data.tobytes()
    # Use gzip to compress the byte array
    compressed_data = gzip.compress(byte_data)
    # Return the size of the compressed data
    return len(compressed_data)
allv2 = all_vectors.reshape(3, 52, 2048)

import numpy as np
import gzip

# Function to compress data and return the size
def compress_and_get_size(data):
    # Convert the numpy array to a byte array
    byte_data = data.tobytes()
    # Use gzip to compress the byte array
    compressed_data = gzip.compress(byte_data)
    # Return the size of the compressed data
    return len(compressed_data)
"""
# Iterate over the hidden embedding vectors
for i in range(allv2.shape[0]):
    for j in range(allv2.shape[1]):
        # Get the vector
        vector = allv2[i, j, :]
        # Calculate the compressed size
        size_vector = compress_and_get_size(vector)
        # Print the size
        print(f"Compressed size of vector at checkpoint {i}, data point {j}: {size_vector} bytes")
"""
# Create a 2D scatter plot
fig = go.Figure()

# Define colors for each checkpoint
colors = ['red', 'green', 'blue']

# Add a line for each data point
for i in range(allv2.shape[1]):
    fig.add_trace(go.Scatter(
        x=list(range(3)),
        y=np.mean(allv2[:, i, :], axis=1),
        mode='lines+markers',
        marker=dict(
            color=colors,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        ),
        text=df.iloc[i,:].title,  # Add title to the trace for each data point
        name=df.iloc[i,:].title   # Add title to the legend
    ))

# Set the title and axis labels
fig.update_layout(
    title="2D plot of the compression of the middle layer embedding activations with the pattern hallucinating -> non hallucinating -> hallucinating",
    xaxis_title='Checkpoints',
    yaxis_title='Compression Size',
    autosize=False,
    width=3000,  # Set the width of the plot
    height=800,  # Set the height of the plot
)

# Show the plot
fig.show()
pio.write_image(fig, "compression_analysis.png")