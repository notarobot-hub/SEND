from logging import StrFormatStyle
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats
from matplotlib import pyplot as plt


comparisons = ['21_82_124_143']
model_name = ['pythia_1b']
order_list = ['0101']
first_diff_mean = []
first_diff_std = []

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
            values = df[[f'pythia_1b_{checkpoints[0]}000', f'pythia_1b_{checkpoints[1]}000', f'pythia_1b_{checkpoints[2]}000', f'pythia_1b_{checkpoints[3]}000']]

            # Flatten each vector and concatenate them into a single numpy array
            all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
            size = all_vectors.size
            all_vectors = all_vectors.reshape(-1, 2048)
            allv2 = all_vectors.reshape(4, int(size/(2048*4)), 2048)
            sub1 = np.subtract(allv2[0], allv2[1])

            y = np.mean(sub1, axis=0)
            std = np.std(sub1, axis=0)
            y_upper = y + std
            y_lower = y - std

            plt.plot(list(range(len(y))), y, label=f'{name}_{check}_{order}')
            plt.fill_between(list(range(len(y))), y_lower, y_upper, alpha=0.4)

    plt.title(f'Penultimate Embedding Activations for Different Models by Mean and STD')
    plt.xlabel("Embedding Number")
    plt.ylabel("Average Activation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pen_gen_embed_change_plot_first_diff_{order}")
    plt.ylim(-40, 40)
    plt.savefig(f"zoomed_pen_gen_embed_change_plot_first_diff_{order}")
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
            values = df[[f'pythia_1b_{checkpoints[0]}000', f'pythia_1b_{checkpoints[1]}000', f'pythia_1b_{checkpoints[2]}000', f'pythia_1b_{checkpoints[3]}000']]

            # Flatten each vector and concatenate them into a single numpy array
            all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
            size = all_vectors.size
            all_vectors = all_vectors.reshape(-1, 2048)
            allv2 = all_vectors.reshape(4, int(size/(2048*4)), 2048)
            sub2 = np.subtract(allv2[2], allv2[3])

            y = np.mean(sub2, axis=0)
            std = np.std(sub2, axis=0)
            y_upper = y + std
            y_lower = y - std

            plt.plot(list(range(len(y))), y, label=f'{name}_{check}_{order}')
            plt.fill_between(list(range(len(y))), y_lower, y_upper, alpha=0.4)

    plt.title(f'Penultimate Embedding Activations for Different Models by Mean and STD')
    plt.xlabel("Embedding Number")
    plt.ylabel("Average Activation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pen_gen_embed_change_plot_second_diff_{order}")

    plt.ylim(-15, 15)
    plt.savefig(f"zoomed_pen_gen_embed_change_plot_second_diff_{order}")
    plt.close()

# The difference in the change between the two subtracted things
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
            values = df[[f'pythia_1b_{checkpoints[0]}000', f'pythia_1b_{checkpoints[1]}000', f'pythia_1b_{checkpoints[2]}000', f'pythia_1b_{checkpoints[3]}000']]

            # Flatten each vector and concatenate them into a single numpy array
            all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
            size = all_vectors.size
            all_vectors = all_vectors.reshape(-1, 2048)
            allv2 = all_vectors.reshape(4, int(size/(2048*4)), 2048)
            sub1 = np.subtract(allv2[0], allv2[1])
            sub2 = np.subtract(allv2[1], allv2[2])
            
            diff = np.add(sub1, sub2)

            y = np.mean(diff, axis=0)
            std = np.std(diff, axis=0)
            y_upper = y + std
            y_lower = y - std

            plt.plot(list(range(len(y))), y, label=f'{name}_{check}_{order}')
            plt.fill_between(list(range(len(y))), y_lower, y_upper, alpha=0.4)

    plt.title(f'Penultimate Sum in the Change of Embedding Activations')
    plt.xlabel("Embedding Number")
    plt.ylabel("Sum of Delta check 1-2 and check 2-3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pen_gen_embed_sum_in_change_plot_{order}")

    plt.ylim(-40, 40)

    plt.savefig(f"zoomed_pen_gen_embed_sum_in_change_plot_{order}")
    plt.close()