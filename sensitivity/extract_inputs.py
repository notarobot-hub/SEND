import json
import pandas as pd

with open(f'.././data/hidden_layer_activations_model_30_to_43.json') as f:
    data = json.load(f)

    df = pd.DataFrame(data)

    for text in df['text']:
            print(f'{text}\n')