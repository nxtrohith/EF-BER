import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report # Not explicitly used in the final logic but good to keep if needed
from sklearn.model_selection import train_test_split, KFold # Added KFold
import random
import time

# Define the labeling scheme based on Indian Standards (unchanged)
def assign_label(hardness):
    if hardness <= 100:
        return 'Soft'
    elif hardness <= 175:
        return 'Moderate'
    elif hardness <= 250:
        return 'Hard'
    else:
        return 'Very Hard'

# TreeNode class to represent nodes in the binary tree (unchanged)
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
        Initialize the EF-BER estimator
        
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
        self.cv_ber_estimate = 0.0 # Added for cross-validated BER
        # self.tree_root = None # Optional: to store the root of the tree if needed later
    
    def kmeans_bisection(self, data, target_clusters): # Mostly unchanged, still sets self.clusters but also returns them
        if data is None or len(data) == 0:
            self.clusters = []
            return []

        # Start with all data in the root node of the tree
        root_node = TreeNode(data=data)
        
        active_leaves = [root_node]
        
        while len(active_leaves) < target_clusters:
            splittable_leaves = [leaf for leaf in active_leaves if len(leaf) >= 2]
            
            if not splittable_leaves:
                break 
            
            largest_leaf_node = max(splittable_leaves, key=len)
            active_leaves.remove(largest_leaf_node)
            cluster_to_split_data = largest_leaf_node.data
            
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
                
                if not current_sub_cluster_1_data or not current_sub_cluster_2_data:
                    if i < max_iter -1 :
                        new_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
                        centroid_1 = cluster_to_split_data[new_indices[0]]
                        centroid_2 = cluster_to_split_data[new_indices[1]]
                        continue 
                    else: 
                        final_sub_cluster_1_data = [] 
                        final_sub_cluster_2_data = []
                        break 

                new_centroid_1 = np.mean(current_sub_cluster_1_data)
                new_centroid_2 = np.mean(current_sub_cluster_2_data)
                
                if np.abs(new_centroid_1 - centroid_1) < 1e-6 and \
                   np.abs(new_centroid_2 - centroid_2) < 1e-6:
                    final_sub_cluster_1_data = current_sub_cluster_1_data
                    final_sub_cluster_2_data = current_sub_cluster_2_data
                    break 
                    
                centroid_1 = new_centroid_1
                centroid_2 = new_centroid_2
                
                if i == max_iter - 1:
                    final_sub_cluster_1_data = current_sub_cluster_1_data
                    final_sub_cluster_2_data = current_sub_cluster_2_data

            if not final_sub_cluster_1_data or not final_sub_cluster_2_data:
                active_leaves.append(largest_leaf_node)
                continue

            largest_leaf_node.is_leaf = False
            child1_data = np.array(final_sub_cluster_1_data)
            child2_data = np.array(final_sub_cluster_2_data)
            child1 = TreeNode(data=child1_data, is_leaf=True)
            child2 = TreeNode(data=child2_data, is_leaf=True)
            largest_leaf_node.left_child = child1
            largest_leaf_node.right_child = child2
            
            if len(child1) > 0:
                active_leaves.append(child1)
            if len(child2) > 0:
                active_leaves.append(child2)
        
        final_cluster_data_list = [leaf.data for leaf in active_leaves if len(leaf.data) > 0]
        self.clusters = final_cluster_data_list # It sets self.clusters
        return self.clusters # And returns them
    
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
        if not most_common_label: return False
        most_common_count = label_counts[most_common_label]
        purity = most_common_count / len(cluster)
        return purity >= threshold
    
    def train(self, data):
        overall_start_time = time.time()
        
        # --- Part 1: Cross-validation for BER estimation ---
        print("\nPerforming 2-fold cross-validation for BER estimation...")
        kf = KFold(n_splits=2, shuffle=True, random_state=42) # Using a fixed random_state for reproducibility
        fold_bers = []
        
        # Ensure data is a numpy array for KFold indexing
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        for fold_num, (train_index, test_index) in enumerate(kf.split(data)):
            print(f"--- Fold {fold_num + 1}/2 ---")
            data_train_fold, data_test_fold = data[train_index], data[test_index]

            if len(data_train_fold) == 0 or len(data_test_fold) == 0:
                print(f"Warning: Fold {fold_num + 1} has empty train or test set. Skipping this fold.")
                # Optionally append a specific value like np.nan or 0, or just skip.
                # If skipped, np.mean will adapt if list is empty later.
                continue

            # 1. K-means bisection on data_train_fold
            # kmeans_bisection will temporarily set self.clusters to the fold's clusters,
            # but we primarily use the returned clusters_fold_train.
            # self.clusters will be correctly set for the final model later.
            print("Clustering training data for the fold...")
            clusters_fold_train = self.kmeans_bisection(data_train_fold, self.k_clusters)

            pure_clusters_fold = []
            for cluster_data_item in clusters_fold_train: # Renamed to avoid conflict
                if self.is_pure_cluster(cluster_data_item):
                    pure_clusters_fold.append(cluster_data_item)
            
            X_knn_train_fold = []
            y_knn_train_fold = []
            for cluster_data_item in pure_clusters_fold:
                for point in cluster_data_item:
                    X_knn_train_fold.append([point])
                    y_knn_train_fold.append(assign_label(point))
            
            if not X_knn_train_fold:
                print("Warning: No pure clusters found in fold. Using all fold training data for KNN.")
                if len(data_train_fold) > 0: # Check if data_train_fold is not empty
                    X_knn_train_fold = [[x_val] for x_val in data_train_fold]
                    y_knn_train_fold = [assign_label(x_val) for x_val in data_train_fold]
                else: # Should not happen if initial check passed, but defensive
                    print(f"Error: data_train_fold is empty in Fold {fold_num + 1} when trying to use all data. Skipping KNN training.")
                    fold_bers.append(np.nan) # Or 0.0 or some indicator of failure
                    continue


            if not X_knn_train_fold: 
                 print(f"Warning: No data available for training KNN model in fold {fold_num + 1}. Assigning BER of 1.0 for this fold.")
                 fold_bers.append(1.0) # Or np.nan, assuming max error if model cannot be trained
                 continue

            print("Training KNN model for the fold...")
            knn_model_fold = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            knn_model_fold.fit(X_knn_train_fold, y_knn_train_fold)
            
            print("Evaluating KNN model on test data for the fold...")
            predictions_test_fold = knn_model_fold.predict(np.array(data_test_fold).reshape(-1, 1))
            actual_labels_test_fold = [assign_label(point) for point in data_test_fold]
            
            noise_count_fold_test = 0
            for actual, predicted in zip(actual_labels_test_fold, predictions_test_fold):
                if actual != predicted:
                    noise_count_fold_test += 1
            
            current_fold_ber = noise_count_fold_test / len(data_test_fold) if len(data_test_fold) > 0 else 0.0
            fold_bers.append(current_fold_ber)
            print(f"Fold {fold_num + 1} BER (on test set): {current_fold_ber:.4f}")
        
        valid_fold_bers = [b for b in fold_bers if not np.isnan(b)]
        if valid_fold_bers:
            self.cv_ber_estimate = np.mean(valid_fold_bers)
            print(f"\nAverage BER from {len(valid_fold_bers)} valid fold(s) of Cross-Validation: {self.cv_ber_estimate:.4f}")
        else:
            self.cv_ber_estimate = np.nan # Indicate CV failed or produced no valid BERs
            print("\nCross-validation did not produce valid BERs (e.g., no data in folds).")
        
        # --- Part 2: Train on the full dataset to finalize the model ---
        print("\nTraining final model on the full dataset...")
        # This part re-uses the original training logic
        
        # kmeans_bisection will now operate on the full 'data' and set self.clusters correctly.
        print("Clustering full data with K-means bisection...")
        self.clusters = self.kmeans_bisection(data, self.k_clusters) 
        
        pure_clusters_full = []
        mixed_clusters_full = []
        
        for cluster_data_item in self.clusters: # Use self.clusters from full data processing
            if self.is_pure_cluster(cluster_data_item):
                pure_clusters_full.append(cluster_data_item)
            else:
                mixed_clusters_full.append(cluster_data_item)
        
        X_train_full = []
        y_train_full = []
        
        for cluster_data_item in pure_clusters_full:
            for point in cluster_data_item:
                X_train_full.append([point])
                y_train_full.append(assign_label(point))
        
        if not X_train_full:
            print("Warning: No pure clusters found in full dataset. Using all data for final KNN training.")
            if len(data) > 0: # Check if original data is not empty
                X_train_full = [[x_val] for x_val in data]
                y_train_full = [assign_label(x_val) for x_val in data]
            else:
                print("Error: Original data is empty. Cannot train final KNN model.")
                self.ber = np.nan
                self.noise_count = 0
                self.total_count = 0
                self.knn_model = None
                overall_end_time = time.time()
                print(f"Training (full model) failed. Total time (incl. CV): {overall_end_time - overall_start_time:.4f} seconds")
                return


        if not X_train_full: # Should be caught by above if data was empty.
             print("Error: No data available for training final KNN model.")
             self.ber = np.nan 
             self.noise_count = 0
             self.total_count = 0
             self.knn_model = None
             overall_end_time = time.time()
             print(f"Training (full model) failed. Total time (incl. CV): {overall_end_time - overall_start_time:.4f} seconds")
             return

        print("Training final KNN model on full dataset...")
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn_model.fit(X_train_full, y_train_full)
        
        # Calculate original BER based on mixed clusters of the full dataset
        self.noise_count = 0 
        self.total_count = 0 
        
        for cluster_data_item in mixed_clusters_full: # From full dataset clustering
            for point in cluster_data_item:
                actual_label = assign_label(point)
                predicted_label = self.knn_model.predict([[point]])[0] 
                
                if actual_label != predicted_label:
                    self.noise_count += 1
                self.total_count += 1 # Counts points in mixed clusters
        
        # Add points from pure clusters to total_count for original BER definition
        self.total_count += len(X_train_full) 
        
        if self.total_count > 0:
            self.ber = self.noise_count / self.total_count # This is the original BER for the full model
        else:
            self.ber = np.nan if len(data)>0 else 0.0 # if total_count is 0 but data existed, it's problematic (NaN)
                                                 # if data itself was empty, BER is 0.
            
        overall_end_time = time.time()
        print(f"Full model training completed. Total time (incl. CV): {overall_end_time - overall_start_time:.4f} seconds")
        print(f"BER (full dataset, original method): {self.ber:.4f} ({self.noise_count} noise points out of {self.total_count} total points considered for this BER)")
        print(f"Estimated BER (from 2-fold CV): {self.cv_ber_estimate:.4f}")

    def predict(self, X): # Unchanged
        if self.knn_model is None:
            raise Exception("Model not trained yet! Or training failed.")
        X_reshaped = np.array(X).reshape(-1, 1)
        return self.knn_model.predict(X_reshaped)
    
    def visualize_clusters(self, data): # Unchanged, uses self.clusters from final full model
        if not self.clusters: 
            print("No clusters to visualize. Train the model first or training might have failed.")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, color='skyblue', edgecolor='black')
        # Original boundary lines were 75, 150, 200. Labels are <=100 (Soft), <=175 (Moderate), <=250 (Hard), >250 (Very Hard)
        # The lines should be at the transition points.
        plt.axvline(x=100, color='g', linestyle='--', label='Soft/Moderate (100)')
        plt.axvline(x=175, color='y', linestyle='--', label='Moderate/Hard (175)')
        plt.axvline(x=250, color='r', linestyle='--', label='Hard/Very Hard (250)')
        plt.title('Water Hardness Distribution')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i, cluster_data in enumerate(self.clusters): 
            if len(cluster_data) > 0: 
                y_values = np.random.normal(i + 1, 0.1, size=len(cluster_data)) # Use cluster_data here
                plt.scatter(cluster_data, y_values, c=colors[i % len(colors)], label=f'Cluster {i+1}')
        
        plt.title('Clusters from K-means Bisection (Full Dataset)')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Cluster (arbitrary y-axis for visualization)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.axvline(x=100, color='g', linestyle=':')
        plt.axvline(x=175, color='y', linestyle=':')
        plt.axvline(x=250, color='r', linestyle=':')
        
        plt.tight_layout()
        plt.show()

# Main execution block (unchanged structure, but will use the modified train method)
if __name__ == "__main__":
    try:
        # df = pd.read_csv('water_potability.csv') # Original file
        # For testing, let's use a more generic CSV load or ensure 'Hardness' column exists.
        # If you have 'hardness_data.csv' from previous context, use that.
        # Assuming 'water_potability.csv' is the target.
        try:
            df = pd.read_csv('water_potability.csv')
            if 'Hardness' not in df.columns:
                print("Error: 'Hardness' column not found in water_potability.csv")
                exit()
            # Handle potential NaN values in Hardness column if any
            df.dropna(subset=['Hardness'], inplace=True)
            hardness_data = df['Hardness'].values
            if len(hardness_data) == 0:
                print("Error: No valid hardness data after removing NaNs.")
                exit()

        except FileNotFoundError:
            print("Error: 'water_potability.csv' not found. Please ensure the file exists.")
            # Fallback to generating some random data for demonstration if file not found
            print("Generating random data for demonstration as file was not found.")
            np.random.seed(42)
            hardness_data = np.concatenate([
                np.random.normal(loc=80, scale=20, size=100),
                np.random.normal(loc=150, scale=30, size=100),
                np.random.normal(loc=220, scale=40, size=100),
                np.random.normal(loc=300, scale=50, size=50)
            ])
            hardness_data = hardness_data[hardness_data > 0] # Ensure positive
            if len(hardness_data) < 10: # Ensure enough data for 2-fold CV
                 print("Error: Not enough random data generated.")
                 exit()


        mean_hardness = hardness_data.mean()
        std_hardness = hardness_data.std()
        print(f"Number of data points: {len(hardness_data)}")
        print(f"Mean of the given data: {mean_hardness:.2f}")
        print(f"Standard deviation of the given data: {std_hardness:.2f}")
        
        estimator = EF_BER_Estimator(k_clusters=4, k_neighbors=3)
        estimator.train(hardness_data) # This now includes CV
        
        estimator.visualize_clusters(hardness_data)
        
        labels = [assign_label(x) for x in hardness_data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\nDistribution of water hardness categories (full dataset):")
        for label, count in label_counts.items():
            print(f"{label}: {count} samples ({count/len(hardness_data)*100:.1f}%)")
        
        if estimator.knn_model: # Check if model was trained successfully
            new_samples = np.array([45, 95, 180, 250, 300])
            predictions = estimator.predict(new_samples)
            print("\nPredictions for new samples:")
            for sample, prediction in zip(new_samples, predictions):
                print(f"Hardness: {sample} ppm -> Predicted category: {prediction}")
        else:
            print("\nKNN model not available for predictions (training might have failed).")

    except FileNotFoundError:
        # This outer except might be redundant if inner one handles it, but keep for general file issues.
        print("Error: A required file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()