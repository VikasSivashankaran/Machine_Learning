from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
wholesale_customers = fetch_ucirepo(id=292)

# data (as pandas dataframes)
X = wholesale_customers.data.features
y = wholesale_customers.data.targets

# Combine features and targets into a single DataFrame
combined_df = pd.concat([X, y], axis=1)

# Save the combined DataFrame to a CSV file
combined_df.to_csv("k_medoid_dataset.csv", index=False)

print("Data successfully saved to k_medoiddataset.csv")


import numpy as np
import pandas as pd
import random
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('k_medoid_dataset.csv')

# Select relevant columns for clustering
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = data[features].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-medoids implementation
def k_medoids(X, k, max_iter=100):
    # Step 1: Randomly select k initial medoids
    n = X.shape[0]
    medoids_idx = random.sample(range(n), k)
    medoids = X[medoids_idx]
    labels = np.zeros(n)

    print(f"Initial medoids indices: {medoids_idx}")
    print(f"Initial medoids: \n{medoids}")

    def calculate_total_cost(X, medoids, labels):
        # Calculate the sum of distances from each point to its assigned medoid
        total_cost = 0
        for i in range(len(X)):
            total_cost += np.linalg.norm(X[i] - medoids[int(labels[i])])
        return total_cost

    for iteration in range(max_iter):
        # Step 2: Assign each data point to the nearest medoid
        for i in range(n):
            distances = [np.linalg.norm(X[i] - medoid) for medoid in medoids]
            labels[i] = np.argmin(distances)

        # Calculate initial cost
        current_cost = calculate_total_cost(X, medoids, labels)
        print(f"Iteration {iteration + 1} - Current total cost: {current_cost}")

        # Step 3: Update medoids while the cost decreases
        improved = False
        for i in range(k):
            for j in range(n):
                if j not in medoids_idx:  # Consider only non-medoid points for swapping
                    # Swap medoid i with point j
                    new_medoids_idx = medoids_idx.copy()
                    new_medoids_idx[i] = j
                    new_medoids = X[new_medoids_idx]

                    # Reassign labels and calculate the new cost
                    new_labels = np.zeros(n)
                    for p in range(n):
                        distances = [np.linalg.norm(X[p] - new_medoid) for new_medoid in new_medoids]
                        new_labels[p] = np.argmin(distances)

                    new_cost = calculate_total_cost(X, new_medoids, new_labels)
                    
                    if new_cost < current_cost:
                        print(f"Found a better medoid swap: {medoids_idx[i]} swapped with {j}")
                        print(f"New total cost: {new_cost}")
                        medoids_idx = new_medoids_idx
                        medoids = new_medoids
                        labels = new_labels
                        current_cost = new_cost
                        improved = True

        if not improved:
            print("No improvement found, stopping iterations.")
            break

    print(f"Final medoids indices: {medoids_idx}")
    print(f"Final medoids: \n{medoids}")
    return labels, medoids

# Apply k-medoids clustering
k = 3  # Number of clusters
labels, medoids = k_medoids(X_scaled, k)

# Plotting using Plotly
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

# 3D Scatter Plot
scatter_plot = go.Scatter3d(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    z=X_scaled[:, 2],
    mode='markers',
    marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8),
    name='Data Points'
)

# Add medoids to scatter plot
medoid_plot = go.Scatter3d(
    x=medoids[:, 0],
    y=medoids[:, 1],
    z=medoids[:, 2],
    mode='markers',
    marker=dict(size=10, color='red', symbol='x'),
    name='Medoids'
)

fig.add_trace(scatter_plot, row=1, col=1)
fig.add_trace(medoid_plot, row=1, col=1)

# 3D Line Plot (showing connections between points and their medoids)
lines = []
for i, medoid in enumerate(medoids):
    cluster_points = X_scaled[labels == i]
    for point in cluster_points:
        line = go.Scatter3d(
            x=[medoid[0], point[0]],
            y=[medoid[1], point[1]],
            z=[medoid[2], point[2]],
            mode='lines',
            line=dict(color='black', width=2),
            opacity=0.5,
            showlegend=False
        )
        lines.append(line)

for line in lines:
    fig.add_trace(line, row=1, col=2)

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Feature 3'
    ),
    title='K-Medoids Clustering with 3D Scatter and Line Plot'
)

fig.show()

