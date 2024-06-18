import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go


with open('./data/hidden_layer_activations.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)