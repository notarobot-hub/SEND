import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats
from matplotlib import pyplot as plt

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

values = df[['pythia_1b_21000', 'pythia_1b_82000', 'pythia_1b_143000']]

# Flatten each vector and concatenate them into a single numpy array
all_vectors = np.concatenate([v.flatten() for col in values.columns for v in values[col]])
size = all_vectors.size
all_vectors = all_vectors.reshape(-1, 2048)
allv2 = all_vectors.reshape(3, int(size/(2048*3)), 2048)

sub1 = np.subtract(allv2[0], allv2[1])
sub2 = np.subtract(allv2[1], allv2[2])

def plot_diff2(sub):

    for i in range(sub.shape[0]):
        plt.plot(sub[i,:])
    
    plt.title("2D plot of the difference between checkpoint 82 checkpoint 143 - 101 ")
    plt.xlabel("Embedding")
    plt.ylabel("Delta Activation")
    plt.show()
    plt.savefig("embedding_difference_82_143_101")
    

def extract_significance(sub):
    z_normalized = stats.zscore(sub, axis=1)
    for i in range(z_normalized.shape[1]):
        counter = 0
        avg = 0
        avg_reg = 0
        for j in range(len(z_normalized)):
            avg += z_normalized[j][i]
            avg_reg += sub[j][i]
            if z_normalized[j][i] >= 3 or z_normalized[j][i] <= -3: # Making sure it is significant
                counter += 1
        
        if counter > len(z_normalized)*0.9: # checking if it's large change in more than 95% of the data
            avg = avg/len(z_normalized)
            avg_reg = avg_reg/len(z_normalized)
            print(f'Location: {i} \tAverage Embed ZScore: {round(avg, 3)}\tCounter: {counter}\tActivation: {round(avg_reg, 3)}')
            

def compare_check1_check2(sub1, sub2):
    diff = np.subtract(sub1, sub2) # Will give us the activations that are highly different

    for i in range(diff.shape[0]):
        plt.plot(diff[i,:])
    
    plt.title("Change from Checkpoints 21, 82, 143 - 101")
    plt.xlabel("Embedding")
    plt.ylabel("Delta Activation")
    plt.show()
    plt.savefig("check_diff_21_82_143_order101")

    print(f'\nZ difference between the subtracted differences between checkpoint')
    z_normalized = stats.zscore(diff, axis=1)
    print(f'Num of inputs: {len(z_normalized)}')
    for i in range(z_normalized.shape[1]):
        counter = 0
        avg = 0
        for j in range(len(z_normalized)):
            avg += z_normalized[j][i]
            if z_normalized[j][i] >= 2 or z_normalized[j][i] <= -2: # Making sure it is significant
                counter += 1

        avg = avg/len(z_normalized)
        if avg > 2 or avg < -2: # checking if it's large change in more than 95% of the data
            print(f'Location: {i} \tAverage Difference ZScore: {round(avg, 3)}\tCounter: {counter}')

    
    
print("\nChange 0-1 embedding significance")
extract_significance(sub1)

print("\nChange 1-2 embedding significance")
extract_significance(sub2)

compare_check1_check2(sub1, sub2)