import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('diabetics.csv')

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn(X, k=2, max_iterations=100):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    clusters = np.zeros(X.shape[0])

    for _ in range(max_iterations):
        for i, point in enumerate(X):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            clusters[i] = np.argmin(distances)

        new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return clusters, centroids

X = data.iloc[:, :-1].values

clusters, centroids = knn(X, k=2)

def get_user_input():
    input_features = []
    feature_names = data.columns[:-1]
    print("Please enter the following information for the new point:")
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                input_features.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value.")
    return np.array(input_features)

new_point = get_user_input()

distances_to_centroids = [euclidean_distance(new_point, centroid) for centroid in centroids]
assigned_cluster = np.argmin(distances_to_centroids)

print(f"The new point belongs to Cluster {assigned_cluster + 1}.")

plt.figure(figsize=(10, 8))
plt.scatter(X[clusters == 0][:, 1], X[clusters == 0][:, 5], color='red', label='Cluster 1', alpha=0.6)
plt.scatter(X[clusters == 1][:, 1], X[clusters == 1][:, 5], color='blue', label='Cluster 2', alpha=0.6)
plt.scatter(centroids[:, 1], centroids[:, 5], color='black', marker='x', s=100, label='Centroids')
plt.scatter(new_point[1], new_point[5], color='green', marker='o', s=100, label='New Point')
plt.xlabel('Glucose')  
plt.ylabel('BMI')  
plt.legend()
plt.title('KNN Clustering with New Point')
plt.show()
