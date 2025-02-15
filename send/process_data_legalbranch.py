

import datasets
import pandas as pd
import os

# Load the dataset in streaming mode
dataset = datasets.load_dataset('nguha/legalbench', "opp115_first_party_collection_use", split="test")


# Initialize lists
texts = []

# Loop through the dataset and save the texts
for sample in dataset:
    texts.append(sample['text'])

# Create a DataFrame
df = pd.DataFrame({'texts': texts})


# Save the DataFrame to a CSV file
df.to_csv('desend/data/legalbench.csv', index=False)

print("First 2000 samples saved as CSV with 'texts' column.")