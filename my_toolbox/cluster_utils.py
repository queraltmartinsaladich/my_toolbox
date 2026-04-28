from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_elbow_method(X_vec, max_k=10, output_dir="results", analysis="test"):
    """Computes inertia for different k values and plots the elbow curve."""
    os.makedirs(output_dir, exist_ok=True)
    inertia = []
    k_values = range(1, max_k + 1)

    print("Running Elbow Method iterations...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=3, n_init='auto')
        kmeans.fit(X_vec)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow method for optimal k')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"elbow_{analysis}.png"))
    plt.show()
    print(f"Elbow plot saved to {output_dir}")    

def run_kmeans(X_vec, n_clusters, random_state=42):
    """
    Fits KMeans to the data and returns the model and cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X_vec)
    return kmeans, labels

def plot_cluster_distribution(labels, output_dir="results", analysis="test"):
    """
    Plots the count of items in each cluster.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts = pd.Series(labels).value_counts().sort_index()
    
    plt.figure(figsize=(8, 5))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Samples per Cluster')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(output_dir, f"cluster_dist_{analysis}.png")
    plt.savefig(save_path)
    plt.close()
    return counts

def plot_clusters_pca(X_vec, labels, output_dir="results", analysis="test"):
    """
    Reduces dimensions to 2D via PCA and plots the clusters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle sparse matrices if coming from TF-IDF
    if hasattr(X_vec, "toarray"):
        X_vec = X_vec.toarray()
        
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_vec)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'PCA Cluster Visualization ({analysis})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    save_path = os.path.join(output_dir, f"cluster_pca_{analysis}.png")
    plt.savefig(save_path)
    plt.close()