import pandas as pd 
import json
import os


# load the actual data
input_data_dir = f'MIND/auto-labeled/output/pythia_1B_143000'
input_data_dir = os.path.abspath(input_data_dir)

# Load the data
with open(f'{input_data_dir}/data_test.json', 'r') as f:
    data = json.load(f)

# Separate data into hallucinating and non-hallucinating
new_entities = []
non_new_entities = []
for sample in data:
    if (len(sample['new_entities']) > 0) and (len(new_entities) < 100):
        new_entities.append(sample['original_text'])
    elif (len(non_new_entities) < 100):
        non_new_entities.append(sample['original_text'])

# Combine the data and create labels
all_data = new_entities + non_new_entities

# save all data as csv with only one column being the text
df = pd.DataFrame(all_data, columns=['text'])

# save the data to desend/data dir as helm.csv
output_data_dir = f'desend/data'
output_data_dir = os.path.abspath(output_data_dir)
df.to_csv(f'{output_data_dir}/helm.csv', index=False)
