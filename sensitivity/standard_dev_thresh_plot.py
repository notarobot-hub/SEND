from logging import StrFormatStyle
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats
from matplotlib import pyplot as plt


comparisons = ['21_82_143', '107_110_114', '107_125_143', '124_125_126']
model_name = ['pythia_1b']
order_list = ['101']
first_diff_mean = []
first_diff_std = []

for threshold in [2, 3]:
    for order in order_list:
        for name in model_name:
            for check in comparisons:
                with open(f'./data/pen_hidden_layer_activations_model_{name}_{check}({order}).json') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                # Scaling the vectors/normalizing them
                avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

                # Compute the scaling factor
                scale = 2048 / avg_norm

                # Scale each vector by the scaling factor
                for col in df.columns[2:]:
                    df[col] = df[col].apply(lambda x: scale * np.array(x))

                checkpoints = check.split("_")
                values = df[[f'pythia_1b_{checkpoints[0]}000', f'pythia_1b_{checkpoints[1]}000', f'pythia_1b_{checkpoints[2]}000']]

                # Flatten each vector and concatenate them into a single numpy array
                all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
                size = all_vectors.size
                all_vectors = all_vectors.reshape(-1, 2048)
                allv2 = all_vectors.reshape(3, int(size/(2048*3)), 2048)
                sub1 = np.subtract(allv2[0], allv2[1])

                sig_count = np.zeros(sub1.shape[1])
                z = stats.zscore(sub1, axis=1)
                for i in range(z.shape[0]):
                    for j in range(z.shape[1]):
                        if z[i][j] > threshold or z[i][j] < -threshold:
                            sig_count[j] += 1

                sig_count = sig_count/len(z)

                plt.plot(range(len(sig_count)), sig_count, label=f'{name}_{check}_{order}')

        plt.title(f'Significance of each Embedding across training points')
        plt.xlabel("Embedding Number")
        plt.ylabel(f"Percentage of datapoints > {threshold} STD away from Mean")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/pen_embed_change_significance_of_{threshold}_first_diff_{order}")
        plt.close()

    for order in order_list:
        for name in model_name:
            for check in comparisons:
                with open(f'./data/pen_hidden_layer_activations_model_{name}_{check}({order}).json') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                # Scaling the vectors/normalizing them
                avg_norm = np.mean([np.linalg.norm(v) for col in df.columns[2:] for v in df[col]])

                # Compute the scaling factor
                scale = 2048 / avg_norm

                # Scale each vector by the scaling factor
                for col in df.columns[2:]:
                    df[col] = df[col].apply(lambda x: scale * np.array(x))

                checkpoints = check.split("_")
                values = df[[f'pythia_1b_{checkpoints[0]}000', f'pythia_1b_{checkpoints[1]}000', f'pythia_1b_{checkpoints[2]}000']]

                # Flatten each vector and concatenate them into a single numpy array
                all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
                size = all_vectors.size
                all_vectors = all_vectors.reshape(-1, 2048)
                allv2 = all_vectors.reshape(3, int(size/(2048*3)), 2048)
                sub1 = np.subtract(allv2[1], allv2[2])

                z = stats.zscore(sub1, axis=1)
                sig_count = np.zeros(sub1.shape[1])

                for i in range(z.shape[0]):
                    for j in range(z.shape[1]):
                        if z[i][j] > threshold or z[i][j] < -threshold:
                            sig_count[j] += 1
                sig_count = sig_count/len(z)

                plt.plot(range(len(sig_count)), sig_count, label=f'{name}_{check}_{order}')

        plt.title(f'Significance of each Embedding across training points')
        plt.xlabel("Embedding Number")
        plt.ylabel(f"Percentage of datapoints > {threshold} STD away from Mean")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/pen_embed_change_significance_of_{threshold}_second_diff_{order}")

        plt.close()