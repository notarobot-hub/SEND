import datasets
import pandas as pd

# Load the dataset openlifescienceai/Med-HALT
dataset = datasets.load_dataset("openlifescienceai/Med-HALT", 'reasoning_FCT')

# Only select 2k samples for training
dataset = dataset['train'].select(range(2000))

# Extract the 'Title' field and rename it to 'texts'
titles = dataset['question']
df = pd.DataFrame({'texts': titles})

# Save the DataFrame to a CSV file
df.to_csv('desend/data/alpaca.csv', index=False)

print("Dataset saved as CSV with 'texts' column.")