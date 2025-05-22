import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import random
import time

# Define the labeling scheme based on Indian Standards (unchanged)
def assign_label(hardness):
    if hardness <= 75:
        return 'Soft'
    elif hardness <= 150:
        return 'Moderate'
    elif hardness <= 200:
        return 'Hard'
    else:
        return 'Very Hard'

# TreeNode class to represent nodes in the binary tree
class TreeNode:
    def __init__(self, data, is_leaf=True, left_child=None, right_child=None):
        # Ensure data is a numpy array, even if empty
        self.data = np.array(data) if data is not None and len(data) > 0 else np.array([])
        self.is_leaf = is_leaf
        self.left_child = left_child
        self.right_child = right_child

    def __len__(self):
        # Helper to get the number of data points in this node's cluster
        return len(self.data)

class EF_BER_Estimator:
    def __init__(self, k_clusters=4, k_neighbors=3):
        """
        Initialize the EF-BER estimator (unchanged)
        
        Args:
            k_clusters: Number of clusters for K-means bisection
            k_neighbors: Number of neighbors for KNN classifier
        """
        self.k_clusters = k_clusters
        self.k_neighbors = k_neighbors
        self.clusters = [] # Will be populated by kmeans_bisection
        self.knn_model = None
        self.noise_count = 0
        self.total_count = 0
        self.ber = 0.0
        # self.tree_root = None # Optional: to store the root of the tree if needed later
    
    def kmeans_bisection(self, data, target_clusters):
        if data is None or len(data) == 0:
            self.clusters = []
            return []

        # Start with all data in the root node of the tree
        root_node = TreeNode(data=data)
        
        # active_leaves will store the current leaf nodes of the tree
        active_leaves = [root_node]
        
        # Bisect clusters until we reach the target number of leaf nodes
        # or no more valid splits are possible
        while len(active_leaves) < target_clusters:
            # Find splittable leaves (those with at least 2 points)
            splittable_leaves = [leaf for leaf in active_leaves if len(leaf) >= 2]
            
            if not splittable_leaves:
                # No leaves are large enough to split further, or no leaves left
                break 
            
            # Find the largest leaf node among splittable ones to bisect
            largest_leaf_node = max(splittable_leaves, key=len)
            
            # Remove it from active_leaves; it will either be replaced by its children
            # or added back if the split fails.
            active_leaves.remove(largest_leaf_node)
            
            cluster_to_split_data = largest_leaf_node.data
            
            # Perform 2-means clustering (core logic is the same as original)
            indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
            centroid_1 = cluster_to_split_data[indices[0]]
            centroid_2 = cluster_to_split_data[indices[1]]
            
            max_iter = 100
            final_sub_cluster_1_data = []
            final_sub_cluster_2_data = []

            for i in range(max_iter):
                current_sub_cluster_1_data = []
                current_sub_cluster_2_data = []
                
                for point in cluster_to_split_data:
                    dist_1 = np.abs(point - centroid_1)
                    dist_2 = np.abs(point - centroid_2)
                    
                    if dist_1 <= dist_2:
                        current_sub_cluster_1_data.append(point)
                    else:
                        current_sub_cluster_2_data.append(point)
                
                # Handle empty clusters during iteration
                if not current_sub_cluster_1_data or not current_sub_cluster_2_data:
                    # Try different initial centroids if an empty cluster occurs
                    # and if we haven't exhausted retries (implicit by max_iter)
                    if i < max_iter -1 : # Avoid issues on last iteration if still empty
                        new_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
                        centroid_1 = cluster_to_split_data[new_indices[0]]
                        centroid_2 = cluster_to_split_data[new_indices[1]]
                        continue # Restart this k-means iteration with new centroids
                    else: # Last iteration and still empty, split failed
                        final_sub_cluster_1_data = [] # Mark as failed
                        final_sub_cluster_2_data = []
                        break 

                new_centroid_1 = np.mean(current_sub_cluster_1_data)
                new_centroid_2 = np.mean(current_sub_cluster_2_data)
                
                if np.abs(new_centroid_1 - centroid_1) < 1e-6 and \
                   np.abs(new_centroid_2 - centroid_2) < 1e-6:
                    final_sub_cluster_1_data = current_sub_cluster_1_data
                    final_sub_cluster_2_data = current_sub_cluster_2_data
                    break # Convergence
                    
                centroid_1 = new_centroid_1
                centroid_2 = new_centroid_2
                
                # Store current state for final if max_iter is reached
                if i == max_iter - 1:
                    final_sub_cluster_1_data = current_sub_cluster_1_data
                    final_sub_cluster_2_data = current_sub_cluster_2_data

            # Check if the split was successful in producing two non-empty clusters
            if not final_sub_cluster_1_data or not final_sub_cluster_2_data:
                # Split failed, add the original node back as a leaf
                active_leaves.append(largest_leaf_node)
                continue # Go to the next iteration of the while loop

            # Split was successful, update the tree structure
            largest_leaf_node.is_leaf = False # No longer a leaf
            
            child1_data = np.array(final_sub_cluster_1_data)
            child2_data = np.array(final_sub_cluster_2_data)

            child1 = TreeNode(data=child1_data, is_leaf=True)
            child2 = TreeNode(data=child2_data, is_leaf=True)
            
            largest_leaf_node.left_child = child1
            largest_leaf_node.right_child = child2
            
            # Add new non-empty children to active_leaves
            if len(child1) > 0:
                active_leaves.append(child1)
            if len(child2) > 0:
                active_leaves.append(child2)
        
        # self.tree_root = root_node # Optionally store the actual tree root
        
        # Extract the data arrays from the final leaf nodes
        final_cluster_data_list = [leaf.data for leaf in active_leaves if len(leaf.data) > 0]
        
        self.clusters = final_cluster_data_list
        return self.clusters
    
    def is_pure_cluster(self, cluster, threshold=0.9): # Unchanged
        if len(cluster) == 0:
            return False
        labels = [assign_label(point) for point in cluster]
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        most_common_label = max(label_counts, key=label_counts.get) if label_counts else None
        if not most_common_label: return False # Handle empty cluster after labeling
        most_common_count = label_counts[most_common_label]
        purity = most_common_count / len(cluster)
        return purity >= threshold
    
    def train(self, data): # Unchanged
        start_time = time.time()
        labeled_data = [(x, assign_label(x)) for x in data]
        
        print("Clustering data with K-means bisection...")
        # kmeans_bisection now sets self.clusters and returns the list of cluster data
        clusters = self.kmeans_bisection(data, self.k_clusters) 
        
        pure_clusters = []
        mixed_clusters = []
        
        for cluster_data in clusters: # Iterate over list of np.arrays
            if self.is_pure_cluster(cluster_data):
                pure_clusters.append(cluster_data)
            else:
                mixed_clusters.append(cluster_data)
        
        X_train = []
        y_train = []
        
        for cluster_data in pure_clusters:
            for point in cluster_data:
                X_train.append([point])
                y_train.append(assign_label(point))
        
        if not X_train:
            print("Warning: No pure clusters found. Using all data for training.")
            X_train = [[x_val] for x_val in data] # Ensure X_train is list of lists
            y_train = [assign_label(x_val) for x_val in data]

        if not X_train: # If data itself was empty
             print("Error: No data available for training KNN model.")
             self.ber = 0.0
             self.noise_count = 0
             self.total_count = 0
             return

        print("Training KNN model...")
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn_model.fit(X_train, y_train)
        
        self.noise_count = 0
        self.total_count = 0 # Reset for current training
        
        for cluster_data in mixed_clusters:
            for point in cluster_data:
                actual_label = assign_label(point)
                predicted_label = self.knn_model.predict([[point]])[0]
                
                if actual_label != predicted_label:
                    self.noise_count += 1
                self.total_count += 1
        
        self.total_count += len(X_train) # Points from pure clusters are also part of total
        
        if self.total_count > 0:
            self.ber = self.noise_count / self.total_count
        else:
            self.ber = 0.0
            
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.4f} seconds")
        print(f"BER: {self.ber:.4f} ({self.noise_count} noise points out of {self.total_count} total)")
    
    def predict(self, X): # Unchanged
        if self.knn_model is None:
            raise Exception("Model not trained yet!")
        X_reshaped = np.array(X).reshape(-1, 1) # Ensure X is numpy array for reshape
        return self.knn_model.predict(X_reshaped)
    
    def visualize_clusters(self, data): # Unchanged
        if not self.clusters: # self.clusters is now populated by kmeans_bisection
            print("No clusters to visualize. Train the model first.")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(x=75, color='g', linestyle='--', label='Soft/Moderate (75)')
        plt.axvline(x=150, color='y', linestyle='--', label='Moderate/Hard (150)')
        plt.axvline(x=200, color='r', linestyle='--', label='Hard/Very Hard (200)')
        plt.title('Water Hardness Distribution')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i, cluster_data in enumerate(self.clusters): # self.clusters is the list of np.arrays
            if len(cluster_data) > 0: # Ensure cluster is not empty before plotting
                y_values = np.random.normal(i + 1, 0.1, size=len(cluster_data))
                plt.scatter(cluster_data, y_values, c=colors[i % len(colors)], label=f'Cluster {i+1}')
        
        plt.title('Clusters from K-means Bisection')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Cluster (arbitrary y-axis for visualization)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.axvline(x=75, color='g', linestyle=':')
        plt.axvline(x=150, color='y', linestyle=':')
        plt.axvline(x=200, color='r', linestyle=':')
        
        plt.tight_layout()
        plt.show()

# Main execution block (unchanged from original structure)
if __name__ == "__main__":
    # Load your data here
    # Assuming data is in a CSV file with a column named 'hardness'
    try:
        df = pd.read_csv('water_potability.csv')
        mean_hardness = df['Hardness'].mean()
        hardness_data = df['Hardness'].values
        std_hardness = df['Hardness'].std()
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

    except FileNotFoundError:
        print("Error: file not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")