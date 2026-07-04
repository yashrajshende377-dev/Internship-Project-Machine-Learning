import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

class KMeansScratch:
    def __init__(self, n_clusters=4, max_iterations=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
    
    def initialize_centroids(self, X):
        """Initialize centroids using k-means++ method"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Choose first centroid randomly
        centroids = [X[np.random.randint(0, n_samples)]]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            # Choose next centroid with probability proportional to distance^2
            probabilities = distances ** 2 / np.sum(distances ** 2)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.random()
            idx = np.searchsorted(cumulative_probs, r)
            centroids.append(X[idx])
        
        return np.array(centroids)
    
    def assign_clusters(self, X, centroids):
        """Assign each sample to nearest centroid"""
        distances = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids based on mean of assigned points"""
        centroids = []
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids.append(np.mean(cluster_points, axis=0))
            else:
                # If cluster is empty, reinitialize
                centroids.append(X[np.random.randint(0, len(X))])
        return np.array(centroids)
    
    def fit(self, X):
        """Fit K-means clustering"""
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for i in range(self.max_iterations):
            # Assign clusters
            old_labels = self.labels
            self.labels = self.assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self.update_centroids(X, self.labels)
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                print(f"Converged after {i+1} iterations")
                break
            
            self.centroids = new_centroids
        
        # Calculate inertia
        self.inertia_ = sum(np.sum((X[self.labels == k] - self.centroids[k]) ** 2) 
                           for k in range(self.n_clusters))
        
        return self
    
    def predict(self, X):
        """Predict cluster for new samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels
    
    def get_cluster_stats(self, X):
        """Get statistics for each cluster"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        stats = []
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                stats.append({
                    'cluster': k,
                    'size': len(cluster_points),
                    'center': self.centroids[k],
                    'within_cluster_distance': np.mean(np.linalg.norm(cluster_points - self.centroids[k], axis=1)),
                    'max_distance': np.max(np.linalg.norm(cluster_points - self.centroids[k], axis=1)),
                    'min_distance': np.min(np.linalg.norm(cluster_points - self.centroids[k], axis=1))
                })
            else:
                stats.append({
                    'cluster': k,
                    'size': 0,
                    'center': self.centroids[k],
                    'within_cluster_distance': 0,
                    'max_distance': 0,
                    'min_distance': 0
                })
        
        return pd.DataFrame(stats)
    
    def plot_clusters(self, X, feature_names=None):
        """Visualize clusters using PCA"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centroids_pca = pca.transform(self.centroids)
        
        plt.figure(figsize=(10, 8))
        
        # Plot clusters
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
        for k in range(self.n_clusters):
            cluster_points = X_pca[self.labels == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[k]], label=f'Cluster {k}', alpha=0.6, s=30)
        
        # Plot centroids
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                   c='red', marker='X', s=200, label='Centroids', edgecolor='black', linewidth=2)
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('K-means Clustering Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage for scratch implementation
def main_scratch():
    # Generate data
    np.random.seed(42)
    n_samples = 500
    
    # Create synthetic customer data
    data = {
        'purchase_frequency': np.concatenate([
            np.random.normal(25, 5, 150),    # Frequent buyers
            np.random.normal(12, 3, 150),    # Occasional buyers
            np.random.normal(5, 2, 100),     # Rare buyers
            np.random.normal(30, 6, 100)     # Very frequent buyers
        ]),
        'avg_spend': np.concatenate([
            np.random.normal(150, 30, 150),   # Low spend
            np.random.normal(400, 80, 150),   # Medium spend
            np.random.normal(700, 100, 100),  # High spend
            np.random.normal(250, 50, 100)    # Medium-low spend
        ]),
        'recency_days': np.concatenate([
            np.random.normal(5, 3, 150),      # Recent
            np.random.normal(15, 5, 150),     # Medium recency
            np.random.normal(30, 8, 100),     # Old
            np.random.normal(3, 2, 100)       # Very recent
        ])
    }
    
    # Ensure positive values
    for key in data:
        data[key] = np.maximum(data[key], 0.1)
    
    df = pd.DataFrame(data)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Apply K-means from scratch
    print("Applying K-means from scratch...")
    kmeans_scratch = KMeansScratch(n_clusters=4, max_iterations=100)
    labels = kmeans_scratch.fit_predict(X_scaled)
    
    print(f"\nInertia: {kmeans_scratch.inertia_:.4f}")
    
    # Get cluster statistics
    stats = kmeans_scratch.get_cluster_stats(X_scaled)
    print("\nCluster Statistics:")
    print(stats)
    
    # Visualize
    kmeans_scratch.plot_clusters(X_scaled)
    
    # Compare with scikit-learn
    from sklearn.cluster import KMeans
    kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_sklearn = kmeans_sklearn.fit_predict(X_scaled)
    
    # Compare labels
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels, labels_sklearn)
    print(f"\nAdjusted Rand Index (vs scikit-learn): {ari:.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("SCIKIT-LEARN IMPLEMENTATION (Recommended)")
    print("=" * 60)
    main()
    
    print("\n" + "=" * 60)
    print("FROM SCRATCH IMPLEMENTATION")
    print("=" * 60)
    main_scratch()