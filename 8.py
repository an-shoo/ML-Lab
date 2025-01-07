import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data: 9 combinations of VAR1 and VAR2
data = {
    'VAR1': [0.123, 0.345, 0.567, 0.234, 0.456, 0.789, 0.987, 0.876, 0.654],
    'VAR2': [0.234, 0.678, 0.890, 0.345, 0.567, 0.456, 0.123, 0.234, 0.789],
    'Classification': ['Class A', 'Class B', 'Class A', 'Class C', 'Class B', 'Class C', 'Class A', 'Class C', 'Class B']
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Features (VAR1, VAR2)
X = df[['VAR1', 'VAR2']].values

# Apply K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# New data point: VAR1 = 0.906, VAR2 = 0.606
new_point = np.array([[0.906, 0.606]])

# Predict the cluster for the new data point
cluster = kmeans.predict(new_point)

# Find the centroid for the predicted cluster
centroid = kmeans.cluster_centers_[cluster]

# Output the predicted cluster and centroid
print(f"Predicted Cluster for VAR1=0.906, VAR2=0.606: Cluster {cluster[0]}")
print(f"Centroid of this Cluster: {centroid}")

# Optional: Plot the clusters and centroids
plt.scatter(df['VAR1'], df['VAR2'], c=kmeans.labels_, cmap='viridis', label='Data Points')
plt.scatter(centroid[0][0], centroid[0][1], c='red', marker='x', s=200, label='Centroid')
plt.scatter(new_point[0][0], new_point[0][1], c='blue', marker='o', s=100, label='New Point (VAR1=0.906, VAR2=0.606)')
plt.xlabel('VAR1')
plt.ylabel('VAR2')
plt.legend()
plt.title('K-means Clustering with 3 Centroids')
plt.show()
