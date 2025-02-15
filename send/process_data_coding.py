import datasets
import pandas as pd
import os

# Specify the number of samples you want to collect
num_samples = 2000

# Load the dataset in streaming mode
# You can choose from: 'python', 'java', 'javascript', 'go', 'ruby', 'php'
language = 'python'

dataset = datasets.load_dataset('code_search_net', language, split='train', streaming=True)

# Initialize list
texts = []

# Loop through the dataset and collect the 'code' field
for i, sample in enumerate(dataset):
    texts.append(sample['whole_func_string'])
    if i >= num_samples - 1:
        break

# Create a pandas DataFrame
df = pd.DataFrame({'texts': texts})

# Ensure the output directory exists
os.makedirs('desend/data', exist_ok=True)

# Save the DataFrame to a CSV file
df.to_csv('desend/data/codesearchnet.csv', index=False)

print(f"First {num_samples} code snippets saved as CSV with 'texts' column.")