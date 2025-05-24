import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report # Not used, so commented out for now
from sklearn.model_selection import train_test_split, KFold
import random
import time
import traceback # For more detailed error reporting in main

# --- Global Helper Functions ---

def assign_label(hardness):
    """
    Assigns a water hardness category based on Indian Standards.
    (Kept as per original specification)
    """
    if hardness <= 100:
        return 'Soft'
    elif hardness <= 175:
        return 'Moderate'
    elif hardness <= 250:
        return 'Hard'
    else:
        return 'Very Hard'

# --- TreeNode Class for K-means Bisection ---

class TreeNode:
    """
    Represents a node in the binary tree created during K-means bisection.
    Each node can hold a cluster of data points.
    """
    def __init__(self, data, is_leaf=True, left_child=None, right_child=None):
        # Ensure data is a numpy array, even if it's empty, for consistency
        self.data = np.array(data) if data is not None and len(data) > 0 else np.array([])
        self.is_leaf = is_leaf
        self.left_child = left_child
        self.right_child = right_child

    def __len__(self):
        """Helper to get the number of data points in this node's cluster."""
        return len(self.data)

# --- Main Estimator Class ---

class EF_BER_Estimator:
    """
    Estimator for Bit Error Rate (BER) using a combination of K-means bisection
    for clustering and a KNN classifier trained on 'pure' clusters.
    Includes 2-fold cross-validation for BER estimation.
    """
    def __init__(self, k_clusters=4, k_neighbors=3):
        """
        Initializes the EF-BER estimator.

        Args:
            k_clusters (int): Target number of clusters for K-means bisection.
            k_neighbors (int): Number of neighbors for the KNN classifier.
        """
        self.k_clusters = k_clusters
        self.k_neighbors = k_neighbors
        self.clusters = []  # Stores clusters from the final model (full dataset)
        self.knn_model = None # KNN model trained on the full dataset
        
        # For BER calculation based on the original method (full dataset)
        self.noise_count = 0
        self.total_count = 0
        self.ber = 0.0 
        
        # For BER estimated via cross-validation
        self.cv_ber_estimate = 0.0
        
        # self.tree_root = None # Optional: Could be used to store the root of the bisection tree

    def kmeans_bisection(self, data_points, target_clusters):
        """
        Performs K-means bisection to partition data_points into target_clusters.
        This method updates self.clusters with the generated clusters and also returns them.
        """
        if data_points is None or len(data_points) == 0:
            self.clusters = []
            return []

        # Initialize with all data in the root node
        root_node = TreeNode(data=data_points)
        active_leaves = [root_node] # List of current leaf nodes to consider for splitting
        
        # Iteratively bisect the largest splittable cluster until target_clusters is reached
        # or no more valid splits can be made.
        while len(active_leaves) < target_clusters:
            # Find leaves that are large enough to be split (must have at least 2 points for 2-means)
            splittable_leaves = [leaf for leaf in active_leaves if len(leaf) >= 2]
            
            if not splittable_leaves:
                # Stop if no leaves are large enough to split further
                break 
            
            # Choose the largest leaf node (by number of data points) to bisect
            largest_leaf_node = max(splittable_leaves, key=len)
            active_leaves.remove(largest_leaf_node) # It will be replaced by its children if split is successful
            
            cluster_to_split_data = largest_leaf_node.data
            
            # --- Perform 2-means clustering on the selected cluster_to_split_data ---
            # Randomly pick two initial centroids from the data
            initial_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
            centroid_1 = cluster_to_split_data[initial_indices[0]]
            centroid_2 = cluster_to_split_data[initial_indices[1]]
            
            max_iterations_2_means = 100  # Max iterations for 2-means convergence
            convergence_threshold = 1e-6   # Threshold for centroid movement
            
            final_sub_cluster_1_data = []
            final_sub_cluster_2_data = []

            for i in range(max_iterations_2_means):
                current_sub_cluster_1_data = []
                current_sub_cluster_2_data = []
                
                # Assign points to the nearest centroid
                for point in cluster_to_split_data:
                    dist_1 = np.abs(point - centroid_1) # Using absolute difference for 1D data
                    dist_2 = np.abs(point - centroid_2)
                    
                    if dist_1 <= dist_2:
                        current_sub_cluster_1_data.append(point)
                    else:
                        current_sub_cluster_2_data.append(point)
                
                # Handle cases where a sub-cluster might become empty during iteration
                if not current_sub_cluster_1_data or not current_sub_cluster_2_data:
                    if i < max_iterations_2_means - 1: 
                        # Try to recover by picking new random centroids if not the last iteration
                        new_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
                        centroid_1 = cluster_to_split_data[new_indices[0]]
                        centroid_2 = cluster_to_split_data[new_indices[1]]
                        continue # Restart this 2-means iteration with new centroids
                    else: 
                        # If it's the last iteration and a cluster is still empty, the split failed
                        final_sub_cluster_1_data = [] 
                        final_sub_cluster_2_data = []
                        break # Exit 2-means loop

                # Recalculate centroids
                new_centroid_1 = np.mean(current_sub_cluster_1_data)
                new_centroid_2 = np.mean(current_sub_cluster_2_data)
                
                # Check for convergence
                if np.abs(new_centroid_1 - centroid_1) < convergence_threshold and \
                   np.abs(new_centroid_2 - centroid_2) < convergence_threshold:
                    final_sub_cluster_1_data = current_sub_cluster_1_data
                    final_sub_cluster_2_data = current_sub_cluster_2_data
                    break # Converged
                    
                centroid_1 = new_centroid_1
                centroid_2 = new_centroid_2
                
                # If max_iterations is reached, use the current clustering
                if i == max_iterations_2_means - 1:
                    final_sub_cluster_1_data = current_sub_cluster_1_data
                    final_sub_cluster_2_data = current_sub_cluster_2_data
            # --- End of 2-means clustering ---

            # If the split failed to produce two non-empty sub-clusters,
            # add the original node back as a leaf and try splitting another node.
            if not final_sub_cluster_1_data or not final_sub_cluster_2_data:
                active_leaves.append(largest_leaf_node) # Add back, as it wasn't successfully split
                continue # Move to the next iteration of the while loop

            # Split was successful, update the tree structure
            largest_leaf_node.is_leaf = False # This node is no longer a leaf
            
            child1_data = np.array(final_sub_cluster_1_data)
            child2_data = np.array(final_sub_cluster_2_data)

            child1 = TreeNode(data=child1_data, is_leaf=True)
            child2 = TreeNode(data=child2_data, is_leaf=True)
            
            largest_leaf_node.left_child = child1
            largest_leaf_node.right_child = child2
            
            # Add new non-empty children to the list of active leaves
            if len(child1) > 0:
                active_leaves.append(child1)
            if len(child2) > 0:
                active_leaves.append(child2)
        
        # self.tree_root = root_node # Optionally store the tree if needed for other purposes
        
        # Extract the data arrays from the final leaf nodes to form the clusters
        final_cluster_data_list = [leaf.data for leaf in active_leaves if len(leaf.data) > 0]
        
        self.clusters = final_cluster_data_list # Update instance's clusters
        return self.clusters # Also return the list of cluster data
    
    def is_pure_cluster(self, cluster_data, purity_threshold=0.9):
        """
        Checks if a given cluster is 'pure' based on the majority label.
        A cluster is pure if the proportion of the most common label exceeds purity_threshold.
        """
        if len(cluster_data) == 0:
            return False # An empty cluster cannot be pure
            
        labels_in_cluster = [assign_label(point) for point in cluster_data]
        
        if not labels_in_cluster: return False # Should not happen if len(cluster_data)>0

        label_counts = {}
        for label in labels_in_cluster:
            label_counts[label] = label_counts.get(label, 0) + 1
            
        most_common_label = max(label_counts, key=label_counts.get) if label_counts else None
        
        if not most_common_label: return False # Should not happen if labels_in_cluster is not empty
            
        most_common_label_count = label_counts[most_common_label]
        purity = most_common_label_count / len(cluster_data)
        
        return purity >= purity_threshold
    
    def train(self, data):
        """
        Trains the EF-BER estimator. This involves:
        1. Performing 2-fold cross-validation to get a robust BER estimate.
        2. Training a final model on the entire dataset for predictions and visualization.
        """
        overall_start_time = time.time()
        
        if not isinstance(data, np.ndarray):
            data = np.array(data) # Ensure data is a NumPy array for easier handling

        # --- Part 1: 2-Fold Cross-Validation for BER Estimation ---
        print("\n--- Starting 2-Fold Cross-Validation for BER Estimation ---")
        kf = KFold(n_splits=2, shuffle=True, random_state=42) # random_state for reproducibility
        fold_bers = [] # To store BER from each fold's test set
        
        for fold_num, (train_indices, test_indices) in enumerate(kf.split(data)):
            current_fold_label = f"Fold {fold_num + 1}/2"
            print(f"\nProcessing {current_fold_label}...")
            
            data_train_fold, data_test_fold = data[train_indices], data[test_indices]

            if len(data_train_fold) == 0 or len(data_test_fold) == 0:
                print(f"Warning ({current_fold_label}): Training or testing set is empty. Skipping this fold.")
                fold_bers.append(np.nan) # Mark as NaN if fold can't be processed
                continue

            # 1a. Cluster the training data of the current fold using K-means bisection
            print(f"({current_fold_label}): Running K-means bisection on training data ({len(data_train_fold)} points)...")
            # Note: self.kmeans_bisection updates self.clusters, but for CV, we primarily use its return value.
            # The final self.clusters will be set later when training on the full dataset.
            clusters_for_fold_training = self.kmeans_bisection(data_train_fold, self.k_clusters)

            # 1b. Identify pure clusters from the fold's training data
            pure_clusters_fold_train = []
            for cluster_item in clusters_for_fold_training:
                if self.is_pure_cluster(cluster_item): # is_pure_cluster takes cluster data directly
                    pure_clusters_fold_train.append(cluster_item)
            
            # 1c. Prepare training data for the fold's KNN model from these pure clusters
            X_knn_train_fold = []
            y_knn_train_fold = []
            for pure_cluster_item in pure_clusters_fold_train:
                for point in pure_cluster_item:
                    X_knn_train_fold.append([point]) # KNN expects 2D array-like for X
                    y_knn_train_fold.append(assign_label(point))
            
            # If no pure clusters were found, use all data from this fold's training set for KNN
            if not X_knn_train_fold:
                print(f"Warning ({current_fold_label}): No pure clusters found. Using all {len(data_train_fold)} training points for KNN.")
                if len(data_train_fold) > 0:
                    X_knn_train_fold = [[x_val] for x_val in data_train_fold]
                    y_knn_train_fold = [assign_label(x_val) for x_val in data_train_fold]
                else: # This case should be caught by the earlier check, but being defensive
                    print(f"Error ({current_fold_label}): Training data is empty when attempting to use all points. Skipping KNN.")
                    fold_bers.append(np.nan)
                    continue
            
            if not X_knn_train_fold: # If still no data for KNN (e.g., data_train_fold was empty initially)
                 print(f"Warning ({current_fold_label}): No data available to train KNN model. Assigning max error (BER=1.0) for this fold.")
                 fold_bers.append(1.0) # Assume maximum error if KNN cannot be trained
                 continue

            # 1d. Train a KNN model for this fold
            print(f"({current_fold_label}): Training KNN model on {len(X_knn_train_fold)} points derived from pure clusters (or all train data)...")
            knn_model_fold = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            knn_model_fold.fit(X_knn_train_fold, y_knn_train_fold)
            
            # 1e. Evaluate this fold's KNN model on the fold's test data
            print(f"({current_fold_label}): Evaluating KNN model on test data ({len(data_test_fold)} points)...")
            predictions_test_fold = knn_model_fold.predict(np.array(data_test_fold).reshape(-1, 1))
            actual_labels_test_fold = [assign_label(point) for point in data_test_fold]
            
            errors_in_fold_test = 0
            for actual, predicted in zip(actual_labels_test_fold, predictions_test_fold):
                if actual != predicted:
                    errors_in_fold_test += 1
            
            current_fold_ber_on_test = errors_in_fold_test / len(data_test_fold) if len(data_test_fold) > 0 else 0.0
            fold_bers.append(current_fold_ber_on_test)
            print(f"({current_fold_label}): BER on test set = {current_fold_ber_on_test:.4f} ({errors_in_fold_test} errors / {len(data_test_fold)} points)")
        
        # Calculate the average BER from valid folds
        valid_fold_bers = [b for b in fold_bers if not np.isnan(b)] # Filter out NaNs
        if valid_fold_bers:
            self.cv_ber_estimate = np.mean(valid_fold_bers)
            print(f"\n--- Cross-Validation Complete ---")
            print(f"Average BER estimated from {len(valid_fold_bers)} valid fold(s) of CV: {self.cv_ber_estimate:.4f}")
        else:
            self.cv_ber_estimate = np.nan # Indicate CV failed or produced no useful results
            print("\n--- Cross-Validation Warning ---")
            print("Cross-validation did not produce any valid BERs (e.g., due to insufficient data in folds).")
        
        # --- Part 2: Train Final Model on the Full Dataset ---
        print("\n--- Starting Final Model Training on Full Dataset ---")
        
        # 2a. Perform K-means bisection on the entire dataset.
        # This will set self.clusters to be used by the final model and visualization.
        print(f"Clustering full dataset ({len(data)} points) with K-means bisection...")
        self.clusters = self.kmeans_bisection(data, self.k_clusters) # self.clusters is now for the full dataset
        
        pure_clusters_full_data = []
        mixed_clusters_full_data = []
        
        for cluster_item in self.clusters: # Use self.clusters (from full data)
            if self.is_pure_cluster(cluster_item):
                pure_clusters_full_data.append(cluster_item)
            else:
                mixed_clusters_full_data.append(cluster_item)
        
        # 2b. Prepare training data for the final KNN model from pure clusters of the full dataset
        X_train_final_knn = []
        y_train_final_knn = []
        for pure_cluster_item in pure_clusters_full_data:
            for point in pure_cluster_item:
                X_train_final_knn.append([point])
                y_train_final_knn.append(assign_label(point))
        
        # If no pure clusters found in the full dataset, use all data for the final KNN model
        if not X_train_final_knn:
            print(f"Warning (Full Dataset): No pure clusters found. Using all {len(data)} data points for final KNN training.")
            if len(data) > 0:
                X_train_final_knn = [[x_val] for x_val in data]
                y_train_final_knn = [assign_label(x_val) for x_val in data]
            else: # This means original data was empty
                print("Error (Full Dataset): Original data is empty. Cannot train final KNN model.")
                self.ber = np.nan # Original BER
                self.noise_count = 0
                self.total_count = 0
                self.knn_model = None # Ensure knn_model is None
                overall_end_time = time.time()
                print(f"Training process failed. Total time: {overall_end_time - overall_start_time:.4f} seconds")
                return # Exit training

        if not X_train_final_knn: # Should be caught if data was empty, but as a final check
             print("Error (Full Dataset): No data available for training the final KNN model (even after fallback).")
             self.ber = np.nan
             self.noise_count = 0
             self.total_count = 0
             self.knn_model = None
             overall_end_time = time.time()
             print(f"Training process failed. Total time: {overall_end_time - overall_start_time:.4f} seconds")
             return

        # 2c. Train the final KNN model
        print(f"Training final KNN model on {len(X_train_final_knn)} points from full dataset...")
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn_model.fit(X_train_final_knn, y_train_final_knn)
        
        # 2d. Calculate BER using the original method (based on mixed clusters of the full dataset)   # This BER is more of a 'training BER' based on the specific mixed/pure cluster logic.
        self.noise_count = 0 # Reset for final model's BER calculation
        
        # Count errors only in mixed clusters (as per original BER definition)
        num_points_in_mixed_clusters = 0
        for mixed_cluster_item in mixed_clusters_full_data:
            num_points_in_mixed_clusters += len(mixed_cluster_item)
            for point in mixed_cluster_item:
                actual_label = assign_label(point)
                # Use the final trained KNN model for prediction here
                predicted_label = self.knn_model.predict([[point]])[0] 
                if actual_label != predicted_label:
                    self.noise_count += 1
        
        # Total count for original BER: points in pure clusters + points in mixed clusters
        self.total_count = len(X_train_final_knn) + num_points_in_mixed_clusters
        
        if self.total_count > 0:
            self.ber = self.noise_count / self.total_count 
        else:
            # If total_count is 0 but data existed, it's an issue. If data was empty, BER is 0.
            self.ber = np.nan if len(data) > 0 else 0.0 
            
        overall_end_time = time.time()
        print(f"\n--- Final Model Training Complete ---")
        print(f"Total training process (including CV) completed in {overall_end_time - overall_start_time:.4f} seconds.")
        
        print(f"\n--- BER Summary ---")
        print(f"BER (Original Method, Full Dataset): {self.ber:.4f}")
        print(f"  (Calculated as {self.noise_count} errors from mixed clusters / {self.total_count} relevant points)")
        print(f"Estimated BER (2-Fold Cross-Validation): {self.cv_ber_estimate:.4f}")

    def predict(self, X_new_samples):
        """
        Predicts hardness categories for new samples using the trained KNN model.
        """
        if self.knn_model is None:
            raise Exception("Model has not been trained yet, or training failed. Cannot predict.")
        
        # Ensure X_new_samples is a 2D array-like structure for KNN's predict method
        X_reshaped = np.array(X_new_samples).reshape(-1, 1)
        return self.knn_model.predict(X_reshaped)
    
    def visualize_clusters(self, original_data):
        """
        Visualizes the overall data distribution and the clusters found by
        K-means bisection on the full dataset.
        """
        if not self.clusters: # self.clusters should be from the final model on full data
            print("No clusters to visualize. Please train the model first, or training might have failed.")
            return
        
        plt.figure(figsize=(14, 7)) # Slightly wider for better text fit
        
        # Plot 1: Histogram of original data distribution with category boundaries
        plt.subplot(1, 2, 1)
        plt.hist(original_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        # Category boundaries based on assign_label function: Soft (<=100), Moderate (<=175), Hard (<=250), Very Hard (>250)
        plt.axvline(x=100, color='darkgreen', linestyle='--', label='Soft/Moderate Boundary (100 ppm)')
        plt.axvline(x=175, color='goldenrod', linestyle='--', label='Moderate/Hard Boundary (175 ppm)')
        plt.axvline(x=250, color='darkred', linestyle='--', label='Hard/Very Hard Boundary (250 ppm)')
        plt.title('Overall Water Hardness Distribution')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        
        # Plot 2: Scatter plot of the clusters
        plt.subplot(1, 2, 2)
        # Define a good set of distinct colors for clusters
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, cluster_data_points in enumerate(self.clusters):
            if len(cluster_data_points) > 0:
                # Use a small amount of jitter on the y-axis for visualization purposes
                y_values_for_plot = np.random.normal(loc=i + 1, scale=0.1, size=len(cluster_data_points))
                plt.scatter(cluster_data_points, y_values_for_plot, 
                            color=cluster_colors[i % len(cluster_colors)], 
                            label=f'Cluster {i+1} ({len(cluster_data_points)} points)', 
                            alpha=0.7, s=50) # s is marker size
        
        plt.title('Clusters from K-means Bisection (Full Dataset)')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Cluster Index (Y-axis for separation)') # More descriptive Y-label
        # Optional: plt.yticks([]) to remove y-axis ticks if they are not meaningful
        plt.grid(True, linestyle=':', alpha=0.5)
        if self.clusters: # Add legend only if there are clusters
             plt.legend(title="Clusters", loc="best")

        # Add category boundary lines to the cluster plot for reference
        plt.axvline(x=100, color='darkgreen', linestyle=':', alpha=0.6)
        plt.axvline(x=175, color='goldenrod', linestyle=':', alpha=0.6)
        plt.axvline(x=250, color='darkred', linestyle=':', alpha=0.6)
        
        plt.tight_layout() # Adjusts plot parameters for a tight layout
        plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- EF-BER Water Hardness Estimation Script ---")
    
    hardness_data = None # Initialize
    data_source_name = ""

    # Try to load data from the primary CSV file
    primary_csv_path = 'water_potability.csv'
    try:
        print(f"\nAttempting to load data from '{primary_csv_path}'...")
        df = pd.read_csv(primary_csv_path)
        data_source_name = primary_csv_path
        
        if 'Hardness' not in df.columns:
            print(f"Error: 'Hardness' column not found in '{primary_csv_path}'.")
            df = None # Signal to try fallback or generate random data
        else:
            # Handle potential NaN values in the 'Hardness' column
            original_count = len(df)
            df.dropna(subset=['Hardness'], inplace=True)
            cleaned_count = len(df)
            if original_count > cleaned_count:
                print(f"Removed {original_count - cleaned_count} rows with NaN Hardness values.")
            
            hardness_data = df['Hardness'].values
            if len(hardness_data) == 0:
                print("Error: No valid (non-NaN) hardness data found in the file.")
                hardness_data = None # Reset to trigger fallback

    except FileNotFoundError:
        print(f"Warning: Primary data file '{primary_csv_path}' not found.")
        df = None # Ensure df is None so fallback is attempted
    except Exception as e:
        print(f"An error occurred while loading '{primary_csv_path}': {e}")
        df = None

    # If loading failed or resulted in no data, try a fallback or generate random data
    if hardness_data is None or len(hardness_data) < 10: # Need at least a few points for CV
        if hardness_data is None:
             print("Proceeding to generate random data for demonstration purposes.")
        elif len(hardness_data) < 10:
             print(f"Loaded data from {data_source_name} but it has very few samples ({len(hardness_data)}). Generating random data instead for a better demo.")

        print("\nGenerating random hardness data for demonstration...")
        np.random.seed(42) # For reproducible random data
        hardness_data = np.concatenate([
            np.random.normal(loc=80, scale=20, size=150),  # Soft range
            np.random.normal(loc=150, scale=25, size=150), # Moderate range
            np.random.normal(loc=220, scale=30, size=100), # Hard range
            np.random.normal(loc=300, scale=35, size=50)   # Very Hard range
        ])
        hardness_data = hardness_data[hardness_data > 0] # Ensure hardness values are positive
        hardness_data = np.clip(hardness_data, 1, 400) # Clip to a reasonable range
        data_source_name = "Randomly Generated Data"
        
        if len(hardness_data) < 10: # Final check on random data
             print("Critical Error: Failed to generate sufficient random data. Exiting.")
             exit()
        print(f"Generated {len(hardness_data)} random data points.")

    # --- Data Summary ---
    print(f"\n--- Data Summary ({data_source_name}) ---")
    print(f"Number of data points being used: {len(hardness_data)}")
    print(f"Mean hardness: {hardness_data.mean():.2f} ppm")
    print(f"Standard deviation of hardness: {hardness_data.std():.2f} ppm")
    print(f"Min hardness: {hardness_data.min():.2f} ppm, Max hardness: {hardness_data.max():.2f} ppm")

    # --- Model Training and Evaluation ---
    print("\n--- Initializing and Training EF-BER Estimator ---")
    # Using k_clusters=4 as there are 4 hardness categories, k_neighbors=3 is a common default
    estimator = EF_BER_Estimator(k_clusters=4, k_neighbors=3)
    
    try:
        estimator.train(hardness_data) # This method now includes CV and final training
        
        # --- Visualization ---
        print("\n--- Visualizing Results ---")
        estimator.visualize_clusters(hardness_data) # Visualizes clusters from the final model
        
        # --- Label Distribution in Full Dataset ---
        print("\n--- Distribution of Hardness Categories (Full Dataset) ---")
        labels_full_data = [assign_label(x) for x in hardness_data]
        label_counts_full_data = {}
        for label in labels_full_data:
            label_counts_full_data[label] = label_counts_full_data.get(label, 0) + 1
        
        for label, count in sorted(label_counts_full_data.items()): # Sorted for consistent order
            print(f"{label}: {count} samples ({count/len(hardness_data)*100:.1f}%)")
        
        # --- Example Predictions ---
        if estimator.knn_model: # Check if the final KNN model was successfully trained
            print("\n--- Example Predictions on New Samples ---")
            # A few sample hardness values spanning different categories
            new_hardness_samples = np.array([45, 90, 110, 170, 200, 260, 310])
            predictions = estimator.predict(new_hardness_samples)
            
            for sample_val, predicted_cat in zip(new_hardness_samples, predictions):
                print(f"Hardness: {sample_val} ppm -> Predicted Category: {predicted_cat} (Actual: {assign_label(sample_val)})")
        else:
            print("\nWarning: Final KNN model not available. Skipping example predictions.")

    except Exception as e:
        print(f"\n--- An Error Occurred During Processing ---")
        print(f"Error details: {e}")
        print("Traceback:")
        traceback.print_exc() # Prints the full traceback for debugging

    print("\n--- Script Finished ---")