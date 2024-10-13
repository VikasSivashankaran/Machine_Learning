import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load datasets
train_df = pd.read_csv('DB_train.csv')
test_df = pd.read_csv('DB_test.csv')

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
numerical_cols = ['Age', 'Work_Experience', 'Family_Size']

# 1. Manually encode categorical columns (using simple mapping as an example)
for col in categorical_cols:
    unique_vals = train_df[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    train_df[col] = train_df[col].map(mapping)
    test_df[col] = test_df[col].map(mapping)

# 2. Handle missing values (replace NaN with the mean for numerical columns)
for col in numerical_cols:
    train_df[col].fillna(train_df[col].mean(), inplace=True)
    test_df[col].fillna(test_df[col].mean(), inplace=True)

# 3. Normalize numerical columns (standard scaling: mean=0, std=1)
for col in numerical_cols:
    mean = train_df[col].mean()
    std = train_df[col].std()
    train_df[col] = (train_df[col] - mean) / std
    test_df[col] = (test_df[col] - mean) / std

# Drop the ID and Segmentation columns for clustering
X_train = train_df.drop(columns=['ID', 'Segmentation']).values
X_test = test_df.drop(columns=['ID']).values

# DBSCAN implementation from scratch
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def region_query(X, point_idx, eps):
    neighbors = []
    for idx, point in enumerate(X):
        if euclidean_distance(X[point_idx], point) < eps:
            neighbors.append(idx)
    return neighbors

def expand_cluster(X, labels, point_idx, cluster_id, eps, min_pts):
    neighbors = region_query(X, point_idx, eps)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1  # Mark as noise
        return False
    else:
        labels[point_idx] = cluster_id  # Assign cluster
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == 0:  # Unvisited point
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(X, neighbor_idx, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id  # Reassign from noise to cluster
            i += 1
        return True

def dbscan(X, eps, min_pts):
    labels = [0] * len(X)  # 0 means unvisited
    cluster_id = 0
    for point_idx in range(len(X)):
        if labels[point_idx] == 0:  # Unvisited
            if expand_cluster(X, labels, point_idx, cluster_id + 1, eps, min_pts):
                cluster_id += 1
    return labels

# Apply DBSCAN on train and test datasets
eps = 0.5  # Maximum distance
min_pts = 5  # Minimum points to form a dense region

train_labels = dbscan(X_train, eps, min_pts)
test_labels = dbscan(X_test, eps, min_pts)

# Add cluster labels to the datasets
train_df['Cluster'] = train_labels
test_df['Cluster'] = test_labels

# Display the cluster distribution
print("Train Dataset Cluster Distribution:\n", pd.Series(train_labels).value_counts())
print("\nTest Dataset Cluster Distribution:\n", pd.Series(test_labels).value_counts())

# Create a subplot with 1 row and 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("DBSCAN Clustering on Train Dataset", "DBSCAN Clustering on Test Dataset"),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]  # Specify scatter type for both
)

# Plotting train dataset
fig.add_trace(
    go.Scatter(
        x=train_df['Age'],
        y=train_df['Family_Size'],
        mode='markers',
        marker=dict(color=train_df['Cluster'], colorscale='Viridis', size=10),
        text=train_df[categorical_cols],  # Hover information
        name='Train Dataset'
    ),
    row=1, col=1
)

# Plotting test dataset
fig.add_trace(
    go.Scatter(
        x=test_df['Age'],
        y=test_df['Family_Size'],
        mode='markers',
        marker=dict(color=test_df['Cluster'], colorscale='Plasma', size=10),
        text=test_df[categorical_cols],  # Hover information
        name='Test Dataset'
    ),
    row=1, col=2
)

# Update layout for titles and axes labels
fig.update_layout(
    title_text="DBSCAN Clustering on Train and Test Datasets",
    xaxis_title='Age',
    yaxis_title='Family Size',
    showlegend=False  # Hide legend if you want to keep it clean
)

# Show the figure
fig.show()
# Display cluster summary messages
train_cluster_counts = pd.Series(train_labels).value_counts()
test_cluster_counts = pd.Series(test_labels).value_counts()

print(f"Train Data: {len(train_cluster_counts)} clusters formed with the following distribution:\n{train_cluster_counts}")
print(f"Test Data: {len(test_cluster_counts)} clusters formed with the following distribution:\n{test_cluster_counts}")

if len(train_cluster_counts) == 1 and train_cluster_counts.index[0] == -1:
    print("Train Data: No clusters found, all points are classified as noise.")
if len(test_cluster_counts) == 1 and test_cluster_counts.index[0] == -1:
    print("Test Data: No clusters found, all points are classified as noise.")
