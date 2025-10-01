from google.colab import drive
# Mount the Google Drive to access files
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA # Required for visualization step

# --- STEP 1: LOAD THE DATASET ---

df = pd.read_csv('/content/drive/MyDrive/Machine Learning/Mall_Customers.csv')
df.head()

# --- STEP 2: DATA PREPROCESSING ---

# Select the features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
# Scale the features
X_scaled = scaler.fit_transform(X)

# --- STEP 3: DETERMINE OPTIMAL K (ELBOW METHOD) ---

inertia = []
K_range = range(1, 11)

# Loop to calculate inertia for each K
for k in K_range:
    # Initialize and fit KMeans for the current K
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# --- STEP 4: PERFORM K-MEANS CLUSTERING ---

optimal_k = 3  # Set optimal K based on the plot

# Initialize the final KMeans model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
# Predict cluster labels
cluster_labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# --- STEP 5: VISUALIZE CLUSTERS ---

pca = PCA(n_components=2)
# Transform the scaled data and centroids for 2D visualization
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8, 6))
# Plot data points, colored by cluster
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
# Plot centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'K-Means Clusters with K={optimal_k}')
plt.legend()
plt.show()
