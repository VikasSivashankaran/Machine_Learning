import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load your dataset
df_test = pd.read_csv('DB_test.csv')

# Select relevant numeric columns (age, work experience, family size)
data = df_test[['Age', 'Work_Experience', 'Family_Size']].fillna(0)

# Function to calculate z-score (standard score) for anomaly detection
def calculate_z_score(column):
    mean = np.mean(column)
    std_dev = np.std(column)
    z_scores = [(x - mean) / std_dev for x in column]
    return z_scores

# Apply z-score calculation to each numeric column
data['Age_zscore'] = calculate_z_score(data['Age'])
data['Work_Experience_zscore'] = calculate_z_score(data['Work_Experience'])
data['Family_Size_zscore'] = calculate_z_score(data['Family_Size'])

# Define a threshold for anomalies (e.g., points with |z-score| > 3 are anomalies)
threshold = 3
data['anomaly'] = (abs(data['Age_zscore']) > threshold) | \
                  (abs(data['Work_Experience_zscore']) > threshold) | \
                  (abs(data['Family_Size_zscore']) > threshold)

# Separate anomalies and normal points
anomalies = data[data['anomaly'] == True]
normal = data[data['anomaly'] == False]

# Plotly 3D scatter plot to visualize anomalies vs normal points
fig = go.Figure()

# Add normal points to the plot
fig.add_trace(go.Scatter3d(
    x=normal['Age'], y=normal['Work_Experience'], z=normal['Family_Size'],
    mode='markers', marker=dict(size=5, color='blue', opacity=0.7),
    name='Normal'))

# Add anomalies to the plot
fig.add_trace(go.Scatter3d(
    x=anomalies['Age'], y=anomalies['Work_Experience'], z=anomalies['Family_Size'],
    mode='markers', marker=dict(size=5, color='red', opacity=0.9),
    name='Anomalies'))

# Update layout for better readability
fig.update_layout(scene=dict(
                    xaxis_title='Age',
                    yaxis_title='Work Experience',
                    zaxis_title='Family Size'),
                  title='3D Scatter Plot for Anomaly Detection (Z-score)')

fig.show()

# Print the output with anomaly column
print(data[['Age', 'Work_Experience', 'Family_Size', 'anomaly']])
