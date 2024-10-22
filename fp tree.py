import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import plotly.graph_objects as go

df = pd.read_csv('DB_train.csv')
categorical_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Segmentation']
df_encoded = pd.get_dummies(df[categorical_cols])

frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

fig = go.Figure(data=[go.Scatter3d(
    x=rules['support'],
    y=rules['confidence'],
    z=rules['lift'],
    mode='markers',
    marker=dict(size=5, color=rules['lift'], colorscale='Viridis', opacity=0.8)
)])

fig.update_layout(scene = dict(
                    xaxis_title='Support',
                    yaxis_title='Confidence',
                    zaxis_title='Lift'),
                  title='3D Scatter Plot of FP-Growth Association Rules')

fig.show()
