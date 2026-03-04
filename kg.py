# -------------------------------
# LOAD PACKAGES
# -------------------------------
import json
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go


# -------------------------------
# LOAD JSON
# -------------------------------
with open('kg.json', 'r') as f:
    data = [json.loads(line) for line in f]
    kg_json = json.load(f)

# Convert to DataFrame
df_raw = pd.DataFrame(kg_json)
