import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import random
import time
import traceback

# --- Helper Functions ---
def assign_label(hardness):
    """Assigns a water hardness category based on Indian Standards."""
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
    """Represents a node in the binary tree for K-means bisection."""
    def __init__(self, data, is_leaf=True, left_child=None, right_child=None):
        self.data = np.array(data) if data is not None and len(data) > 0 else np.array([])
        self.is_leaf = is_leaf
        self.left_child = left_child
        self.right_child = right_child

    def __len__(self):
        return len(self.data)

# --- Main Estimator Class ---
class EF_BER_Estimator:
    """
    Estimator for Bit Error Rate (BER) using K-means bisection and KNN,
    with 10-fold cross-validation for BER estimation.
    """
    def __init__(self, k_clusters=4, k_neighbors=3):
        self.k_clusters = k_clusters
        self.k_neighbors = k_neighbors
        self.clusters = []
        self.knn_model = None
        self.noise_count = 0
        self.total_count = 0
        self.ber = 0.0
        self.cv_ber_estimate = 0.0

    def kmeans_bisection(self, data_points, target_clusters):
        """Performs K-means bisection to partition data into target_clusters."""
        if data_points is None or len(data_points) == 0:
            self.clusters = []
            return []

        root_node = TreeNode(data=data_points)
        active_leaves = [root_node]
        
        while len(active_leaves) < target_clusters:
            splittable_leaves = [leaf for leaf in active_leaves if len(leaf) >= 2]
            if not splittable_leaves:
                break 
            
            largest_leaf_node = max(splittable_leaves, key=len)
            active_leaves.remove(largest_leaf_node)
            cluster_to_split_data = largest_leaf_node.data
            
            initial_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
            centroid_1, centroid_2 = cluster_to_split_data[initial_indices[0]], cluster_to_split_data[initial_indices[1]]
            
            max_iterations_2_means = 100
            convergence_threshold = 1e-6
            final_sub_cluster_1_data, final_sub_cluster_2_data = [], []

            for i in range(max_iterations_2_means):
                current_sub_cluster_1_data, current_sub_cluster_2_data = [], []
                for point in cluster_to_split_data:
                    dist_1, dist_2 = np.abs(point - centroid_1), np.abs(point - centroid_2)
                    if dist_1 <= dist_2: current_sub_cluster_1_data.append(point)
                    else: current_sub_cluster_2_data.append(point)
                
                if not current_sub_cluster_1_data or not current_sub_cluster_2_data:
                    if i < max_iterations_2_means - 1: 
                        new_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
                        centroid_1, centroid_2 = cluster_to_split_data[new_indices[0]], cluster_to_split_data[new_indices[1]]
                        continue 
                    else: 
                        final_sub_cluster_1_data, final_sub_cluster_2_data = [], []
                        break 

                new_centroid_1, new_centroid_2 = np.mean(current_sub_cluster_1_data), np.mean(current_sub_cluster_2_data)
                
                if np.abs(new_centroid_1 - centroid_1) < convergence_threshold and \
                   np.abs(new_centroid_2 - centroid_2) < convergence_threshold:
                    final_sub_cluster_1_data, final_sub_cluster_2_data = current_sub_cluster_1_data, current_sub_cluster_2_data
                    break 
                centroid_1, centroid_2 = new_centroid_1, new_centroid_2
                if i == max_iterations_2_means - 1:
                    final_sub_cluster_1_data, final_sub_cluster_2_data = current_sub_cluster_1_data, current_sub_cluster_2_data

            if not final_sub_cluster_1_data or not final_sub_cluster_2_data:
                active_leaves.append(largest_leaf_node); continue

            largest_leaf_node.is_leaf = False
            child1 = TreeNode(data=np.array(final_sub_cluster_1_data), is_leaf=True)
            child2 = TreeNode(data=np.array(final_sub_cluster_2_data), is_leaf=True)
            largest_leaf_node.left_child, largest_leaf_node.right_child = child1, child2
            
            if len(child1) > 0: active_leaves.append(child1)
            if len(child2) > 0: active_leaves.append(child2)
        
        final_cluster_data_list = [leaf.data for leaf in active_leaves if len(leaf.data) > 0]
        self.clusters = final_cluster_data_list
        return self.clusters
    
    def is_pure_cluster(self, cluster_data, purity_threshold=0.9):
        if len(cluster_data) == 0: return False
        labels_in_cluster = [assign_label(point) for point in cluster_data]
        if not labels_in_cluster: return False
        label_counts = {label: labels_in_cluster.count(label) for label in set(labels_in_cluster)}
        if not label_counts: return False
        return (max(label_counts.values()) / len(cluster_data)) >= purity_threshold
    
    def train(self, data):
        overall_start_time = time.time()
        if not isinstance(data, np.ndarray): data = np.array(data)

        # --- Part 1: 10-Fold Cross-Validation for BER Estimation ---
        print("\n--- Starting 10-Fold Cross-Validation for BER Estimation ---")
        kf = KFold(n_splits=10, shuffle=True, random_state=42) # Changed to 10 splits
        fold_bers = []
        
        for fold_num, (train_indices, test_indices) in enumerate(kf.split(data)):
            current_fold_label = f"Fold {fold_num + 1}/10" # Updated label
            print(f"\nProcessing {current_fold_label}...")
            data_train_fold, data_test_fold = data[train_indices], data[test_indices]

            if len(data_train_fold) == 0 or len(data_test_fold) == 0:
                print(f"Warning ({current_fold_label}): Empty train/test set. Skipping.")
                fold_bers.append(np.nan); continue

            clusters_for_fold_training = self.kmeans_bisection(data_train_fold, self.k_clusters)
            pure_clusters_fold_train = [item for item in clusters_for_fold_training if self.is_pure_cluster(item)]
            
            X_knn_train_fold, y_knn_train_fold = [], []
            if pure_clusters_fold_train:
                for pure_cluster_item in pure_clusters_fold_train:
                    for point in pure_cluster_item:
                        X_knn_train_fold.append([point])
                        y_knn_train_fold.append(assign_label(point))
            else: 
                print(f"Warning ({current_fold_label}): No pure clusters. Using all {len(data_train_fold)} training points for KNN.")
                if len(data_train_fold) > 0:
                    X_knn_train_fold = [[x_val] for x_val in data_train_fold]
                    y_knn_train_fold = [assign_label(x_val) for x_val in data_train_fold]
            
            if not X_knn_train_fold:
                 print(f"Warning ({current_fold_label}): No data for KNN. Assigning BER=1.0.")
                 fold_bers.append(1.0); continue

            knn_model_fold = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            knn_model_fold.fit(X_knn_train_fold, y_knn_train_fold)
            
            predictions_test_fold = knn_model_fold.predict(np.array(data_test_fold).reshape(-1, 1))
            actual_labels_test_fold = [assign_label(point) for point in data_test_fold]
            
            errors_in_fold_test = sum(1 for actual, predicted in zip(actual_labels_test_fold, predictions_test_fold) if actual != predicted)
            current_fold_ber_on_test = errors_in_fold_test / len(data_test_fold) if len(data_test_fold) > 0 else 0.0
            fold_bers.append(current_fold_ber_on_test)
            print(f"({current_fold_label}): BER on test set = {current_fold_ber_on_test:.4f}")
        
        valid_fold_bers = [b for b in fold_bers if not np.isnan(b)]
        if valid_fold_bers:
            self.cv_ber_estimate = np.mean(valid_fold_bers)
            print(f"\n--- Cross-Validation Complete ---")
            print(f"Average BER from {len(valid_fold_bers)} valid fold(s) of CV: {self.cv_ber_estimate:.4f}")
        else:
            self.cv_ber_estimate = np.nan
            print("\n--- Cross-Validation Warning: No valid BERs produced. ---")
        
        # --- Part 2: Train Final Model on the Full Dataset ---
        print("\n--- Starting Final Model Training on Full Dataset ---")
        self.clusters = self.kmeans_bisection(data, self.k_clusters)
        
        pure_clusters_full_data = [item for item in self.clusters if self.is_pure_cluster(item)]
        mixed_clusters_full_data = [item for item in self.clusters if not self.is_pure_cluster(item)]
        
        X_train_final_knn, y_train_final_knn = [], []
        if pure_clusters_full_data:
            for pure_cluster_item in pure_clusters_full_data:
                for point in pure_cluster_item:
                    X_train_final_knn.append([point])
                    y_train_final_knn.append(assign_label(point))
        else: 
            print(f"Warning (Full Dataset): No pure clusters. Using all {len(data)} data points for final KNN.")
            if len(data) > 0:
                X_train_final_knn = [[x_val] for x_val in data]
                y_train_final_knn = [assign_label(x_val) for x_val in data]
        
        if not X_train_final_knn:
             print("Error (Full Dataset): No data for final KNN model. Training failed.")
             self.ber, self.knn_model = np.nan, None; return

        self.knn_model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn_model.fit(X_train_final_knn, y_train_final_knn)
        
        self.noise_count = 0
        num_points_in_mixed_clusters = 0
        for mixed_cluster_item in mixed_clusters_full_data:
            num_points_in_mixed_clusters += len(mixed_cluster_item)
            for point in mixed_cluster_item:
                if assign_label(point) != self.knn_model.predict([[point]])[0]:
                    self.noise_count += 1
        
        self.total_count = len(X_train_final_knn) + num_points_in_mixed_clusters
        self.ber = self.noise_count / self.total_count if self.total_count > 0 else (np.nan if len(data) > 0 else 0.0)
            
        overall_end_time = time.time()
        print(f"\n--- Final Model Training Complete (Total time: {overall_end_time - overall_start_time:.2f}s) ---")
        print(f"BER (Original Method, Full Dataset): {self.ber:.4f}")
        print(f"Estimated BER (10-Fold Cross-Validation): {self.cv_ber_estimate:.4f}")

    def predict(self, X_new_samples):
        if self.knn_model is None: raise Exception("Model not trained or training failed.")
        return self.knn_model.predict(np.array(X_new_samples).reshape(-1, 1))
    
    def visualize_clusters(self, original_data):
        if not self.clusters:
            print("No clusters to visualize. Train model first or training failed."); return
        
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.hist(original_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        boundaries = {'Soft/Mod (100)': 100, 'Mod/Hard (175)': 175, 'Hard/V.Hard (250)': 250}
        colors = ['darkgreen', 'goldenrod', 'darkred']
        for i, (label, val) in enumerate(boundaries.items()):
            plt.axvline(x=val, color=colors[i % len(colors)], linestyle='--', label=f'{label} ppm')
        plt.title('Overall Water Hardness Distribution')
        plt.xlabel('Hardness (ppm)'); plt.ylabel('Frequency')
        plt.legend(); plt.grid(True, linestyle=':', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, cluster_data in enumerate(self.clusters):
            if len(cluster_data) > 0:
                y_values = np.random.normal(loc=i + 1, scale=0.1, size=len(cluster_data))
                plt.scatter(cluster_data, y_values, color=cluster_colors[i % len(cluster_colors)], 
                            label=f'Cluster {i+1} ({len(cluster_data)} pts)', alpha=0.7, s=50)
        plt.title('Clusters from K-means Bisection (Full Dataset)')
        plt.xlabel('Hardness (ppm)'); plt.ylabel('Cluster Index (Visual Separation)')
        plt.grid(True, linestyle=':', alpha=0.5)
        if self.clusters: plt.legend(title="Clusters", loc="best")
        for val_color, val in zip(colors, boundaries.values()):
             plt.axvline(x=val, color=val_color, linestyle=':', alpha=0.6)
        
        plt.tight_layout(); plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- EF-BER Water Hardness Estimation (10-Fold CV) ---") # Updated title
    hardness_data, data_source_name = None, ""
    primary_csv_path = 'synthetic_water_dataset_2.csv'

    try:
        df = pd.read_csv(primary_csv_path)
        data_source_name = primary_csv_path
        if 'Hardness' not in df.columns:
            print(f"Error: 'Hardness' column missing in '{primary_csv_path}'."); df = None
        else:
            df.dropna(subset=['Hardness'], inplace=True)
            hardness_data = df['Hardness'].values
            if len(hardness_data) == 0: hardness_data = None
    except FileNotFoundError: print(f"Warning: '{primary_csv_path}' not found.")
    except Exception as e: print(f"Error loading '{primary_csv_path}': {e}")

    min_data_for_cv = 30 # Adjusted for 10-fold (e.g., at least 3 per test fold)
    if hardness_data is None or len(hardness_data) < min_data_for_cv: 
        reason = "data loading failed" if hardness_data is None else f"insufficient data ({len(hardness_data)} points)"
        print(f"Warning: {reason}. Generating random data for demonstration.")
        np.random.seed(42)
        hardness_data = np.concatenate([
            np.random.normal(80, 20, 150), np.random.normal(150, 25, 150),
            np.random.normal(220, 30, 100), np.random.normal(300, 35, 50)
        ])
        hardness_data = np.clip(hardness_data[hardness_data > 0], 1, 400)
        data_source_name = "Randomly Generated Data"
        if len(hardness_data) < min_data_for_cv:
             print("Critical Error: Failed to generate sufficient random data. Exiting."); exit()

    print(f"\n--- Data Summary ({data_source_name}) ---")
    print(f"Points: {len(hardness_data)}, Mean: {hardness_data.mean():.2f}, StdDev: {hardness_data.std():.2f}")

    estimator = EF_BER_Estimator(k_clusters=4, k_neighbors=3)
    try:
        estimator.train(hardness_data)
        estimator.visualize_clusters(hardness_data)
        
        labels_full = [assign_label(x) for x in hardness_data]
        print("\n--- Hardness Categories Distribution (Full Dataset) ---")
        for label, count in sorted({lbl: labels_full.count(lbl) for lbl in set(labels_full)}.items()):
            print(f"{label}: {count} samples ({count/len(hardness_data)*100:.1f}%)")
        
        if estimator.knn_model:
            print("\n--- Example Predictions ---")
            samples = np.array([45, 90, 110, 170, 200, 260, 310])
            predictions = estimator.predict(samples)
            for s, p in zip(samples, predictions):
                print(f"Hardness: {s} ppm -> Predicted: {p} (Actual: {assign_label(s)})")
        else:
            print("\nWarning: Final KNN model not available for predictions.")
    except Exception as e:
        print(f"\n--- Error During Processing: {e} ---"); traceback.print_exc()
    print("\n--- Script Finished ---")
