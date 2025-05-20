import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import random
import time

# Define the labeling scheme based on Indian Standards
def assign_label(hardness):
    if hardness <= 75:
        return 'Soft'
    elif hardness <= 150:
        return 'Moderate'
    elif hardness <= 200:
        return 'Hard'
    else:
        return 'Very Hard'

class EF_BER_Estimator:
    def __init__(self, k_clusters=4, k_neighbors=3):
        """
        Initialize the EF-BER estimator
        
        Args:
            k_clusters: Number of clusters for K-means bisection
            k_neighbors: Number of neighbors for KNN classifier
        """
        self.k_clusters = k_clusters
        self.k_neighbors = k_neighbors
        self.clusters = []
        self.knn_model = None
        self.noise_count = 0
        self.total_count = 0
        self.ber = 0.0
    
    def kmeans_bisection(self, data, target_clusters):
        # Start with all data in one cluster
        clusters = [data]
        
        # Bisect clusters until we reach the target number
        while len(clusters) < target_clusters:
            # Find the largest cluster to bisect
            largest_cluster_idx = np.argmax([len(cluster) for cluster in clusters])
            cluster_to_split = clusters.pop(largest_cluster_idx)

            if len(cluster_to_split) < 2:
                clusters.append(cluster_to_split)
                continue
            
            # Initialize centroids for bisection
            # Choose two points randomly for initial centroids
            indices = np.random.choice(len(cluster_to_split), 2, replace=False)
            centroid_1 = cluster_to_split[indices[0]]
            centroid_2 = cluster_to_split[indices[1]]
            
            # Iterate until convergence or max iterations
            max_iter = 100
            for _ in range(max_iter):
                # Assign points to closest centroid
                cluster_1 = []
                cluster_2 = []
                
                for point in cluster_to_split:
                    dist_1 = np.abs(point - centroid_1)
                    dist_2 = np.abs(point - centroid_2)
                    
                    if dist_1 <= dist_2:
                        cluster_1.append(point)
                    else:
                        cluster_2.append(point)
                
                # Handle empty clusters
                if len(cluster_1) == 0 or len(cluster_2) == 0:
                    # Try different initial centroids
                    indices = np.random.choice(len(cluster_to_split), 2, replace=False)
                    centroid_1 = cluster_to_split[indices[0]]
                    centroid_2 = cluster_to_split[indices[1]]
                    continue
                
                # Update centroids
                new_centroid_1 = np.mean(cluster_1)
                new_centroid_2 = np.mean(cluster_2)
                
                # Check for convergence
                if np.abs(new_centroid_1 - centroid_1) < 1e-6 and np.abs(new_centroid_2 - centroid_2) < 1e-6:
                    break
                    
                centroid_1 = new_centroid_1
                centroid_2 = new_centroid_2
            
            # Add the new clusters
            if cluster_1:  # Only add non-empty clusters
                clusters.append(np.array(cluster_1))
            if cluster_2:
                clusters.append(np.array(cluster_2))
        
        self.clusters = clusters
        return clusters
    
    def is_pure_cluster(self, cluster, threshold=0.9):
        if len(cluster) == 0:
            return False
        # Get labels for all points in cluster
        labels = [assign_label(point) for point in cluster]
        
        # Count occurrences of each label
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # Find the most common label
        most_common_label = max(label_counts, key=label_counts.get)
        most_common_count = label_counts[most_common_label]
        
        # Calculate purity
        purity = most_common_count / len(cluster)
        
        return purity >= threshold
    
    def train(self, data):
        # Step 1: Load and label data
        start_time = time.time()
        labeled_data = [(x, assign_label(x)) for x in data]
        
        # Step 2: Cluster with k-means bisection
        print("Clustering data with K-means bisection...")
        clusters = self.kmeans_bisection(data, self.k_clusters)
        
        # Separate pure and mixed clusters
        pure_clusters = []
        mixed_clusters = []
        
        for cluster in clusters:
            if self.is_pure_cluster(cluster):
                pure_clusters.append(cluster)
            else:
                mixed_clusters.append(cluster)
        
        # Prepare training data from pure clusters
        X_train = []
        y_train = []
        
        for cluster in pure_clusters:
            for point in cluster:
                X_train.append([point])
                y_train.append(assign_label(point))
        
        # Handle the case where there are no pure clusters
        if not X_train:
            print("Warning: No pure clusters found. Using all data for training.")
            X_train = [[x] for x in data]
            y_train = [assign_label(x) for x in data]
        
        # Step 3: Train k-NN model on large/pure clusters
        print("Training KNN model...")
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn_model.fit(X_train, y_train)
        
        # Step 4 & 5: Predict small/mixed cluster labels and compare with actual
        self.noise_count = 0
        self.total_count = 0
        
        for cluster in mixed_clusters:
            for point in cluster:
                actual_label = assign_label(point)
                predicted_label = self.knn_model.predict([[point]])[0]
                
                if actual_label != predicted_label:
                    self.noise_count += 1
                self.total_count += 1
        
        # Also count points from pure clusters in total
        self.total_count += len(X_train)
        
        # Step 6: Compute BER
        if self.total_count > 0:
            self.ber = self.noise_count / self.total_count
        else:
            self.ber = 0.0
            
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.4f} seconds")
        print(f"BER: {self.ber:.4f} ({self.noise_count} noise points out of {self.total_count} total)")
    
    def predict(self, X):
        if self.knn_model is None:
            raise Exception("Model not trained yet!")
        
        # Reshape input if needed
        X_reshaped = X.reshape(-1, 1) if len(X.shape) == 1 else X
        
        return self.knn_model.predict(X_reshaped)
    
    def visualize_clusters(self, data):
        if not self.clusters:
            print("No clusters to visualize. Train the model first.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Data distribution with hardness thresholds
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(x=75, color='g', linestyle='--', label='Soft/Moderate (75)')
        plt.axvline(x=150, color='y', linestyle='--', label='Moderate/Hard (150)')
        plt.axvline(x=200, color='r', linestyle='--', label='Hard/Very Hard (200)')
        plt.title('Water Hardness Distribution')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 2: Clusters visualization
        plt.subplot(1, 2, 2)
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i, cluster in enumerate(self.clusters):
            # For 1D data, create a y-value for visualization
            y_values = np.random.normal(i+1, 0.1, size=len(cluster))
            plt.scatter(cluster, y_values, c=colors[i % len(colors)], label=f'Cluster {i+1}')
        
        plt.title('Clusters from K-means Bisection')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Cluster')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add thresholds
        plt.axvline(x=75, color='g', linestyle=':')
        plt.axvline(x=150, color='y', linestyle=':')
        plt.axvline(x=200, color='r', linestyle=':')
        
        plt.tight_layout()
        plt.show()

# Example usage with synthetic data (since actual data wasn't provided)
# def generate_sample_data(n_samples=200):
#     # Generate data in each category
#     soft = np.random.normal(50, 15, int(n_samples * 0.3))
#     soft = soft[(soft >= 0) & (soft <= 75)]  # Ensure values are in range
    
#     moderate = np.random.normal(110, 20, int(n_samples * 0.4))
#     moderate = moderate[(moderate > 75) & (moderate <= 150)]
    
#     hard = np.random.normal(175, 15, int(n_samples * 0.2))
#     hard = hard[(hard > 150) & (hard <= 200)]
    
#     very_hard = np.random.normal(230, 20, int(n_samples * 0.1))
#     very_hard = very_hard[very_hard > 200]
    
#     # Add some outliers
#     outliers = np.array([250, 275, 300, 10, 5])
    
#     # Combine all data
#     hardness_data = np.concatenate([soft, moderate, hard, very_hard, outliers])
    
#     # Shuffle data
#     np.random.shuffle(hardness_data)
    
#     return hardness_data

# Main execution
if __name__ == "__main__":
    # Load your data here
    # Assuming data is in a CSV file with a column named 'hardness'
    df = pd.read_csv('hardness_data.csv')
    hardness_data = df['hardness'].values
    mean_hardness = df['hardness'].mean()
    std_hardness = df['hardness'].std()
    print(f"Mean of the given data: {mean_hardness}")
    print(f"Standard deviation of the given data: {std_hardness}")
    
    # Create and train the EF-BER estimator
    estimator = EF_BER_Estimator(k_clusters=4, k_neighbors=3)
    estimator.train(hardness_data)
    
    # Visualize the results
    estimator.visualize_clusters(hardness_data)
    
    # Print distribution of labels
    labels = [assign_label(x) for x in hardness_data]
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    print("\nDistribution of water hardness categories:")
    for label, count in label_counts.items():
        print(f"{label}: {count} samples ({count/len(hardness_data)*100:.1f}%)")
    
    # Example of making predictions on new data
    new_samples = np.array([45, 95, 180, 250])
    predictions = estimator.predict(new_samples)
    
    print("\nPredictions for new samples:")
    for sample, prediction in zip(new_samples, predictions):
        print(f"Hardness: {sample} ppm -> Predicted category: {prediction}")
