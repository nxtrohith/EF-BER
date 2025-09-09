import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import random
import time
import traceback

# --- Scikit-learn Classifiers ---
from sklearn.neighbors import KNeighborsClassifier # For EF-BER
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# --- Global Helper Functions ---

def assign_label(hardness):
    """
    Assigns a water hardness category based on Indian Standards.
    """
    if hardness <= 100:
        return 'Soft'
    elif hardness <= 175:
        return 'Moderate'
    elif hardness <= 250:
        return 'Hard'
    else:
        return 'Very Hard'

# --- TreeNode Class for K-means Bisection (Used by EF-BER) ---
class TreeNode:
    """
    Represents a node in the binary tree created during K-means bisection.
    Each node can hold a cluster of data points.
    """
    def __init__(self, data, is_leaf=True, left_child=None, right_child=None):
        self.data = np.array(data) if data is not None and len(data) > 0 else np.array([])
        self.is_leaf = is_leaf
        self.left_child = left_child
        self.right_child = right_child

    def __len__(self):
        return len(self.data)

# --- Main Classifier Class ---
class HardnessClassifier:
    """
    A generalized classifier for water hardness estimation, supporting multiple models
    including EF-BER and standard scikit-learn classifiers.
    """
    def __init__(self, model_name='ef-ber', k_clusters=4, k_neighbors=3, random_state=42, **model_params):
        """
        Initializes the Hardness Classifier.

        Args:
            model_name (str): Name of the model to use. Options:
                              'ef-ber', 'svm', 'dt', 'rf', 'adaboost', 'gnb'.
            k_clusters (int): For 'ef-ber', target number of clusters for K-means bisection.
            k_neighbors (int): For 'ef-ber' (KNN part) and if 'knn' is chosen directly.
            random_state (int): Random state for reproducibility.
            **model_params: Additional parameters for scikit-learn models.
        """
        self.model_name = model_name.lower()
        self.k_clusters = k_clusters
        self.k_neighbors = k_neighbors # Used by EF-BER's internal KNN
        self.random_state = random_state
        self.model_params = model_params

        self.model = None # This will hold the trained scikit-learn model or EF-BER's KNN
        self.ef_ber_clusters = [] # Specific to EF-BER for visualization

        # For BER calculation
        self.cv_ber_estimate = np.nan # BER estimated via cross-validation (1 - accuracy)
        self.final_model_accuracy = np.nan # Accuracy of the final model on the full training set
        
        # EF-BER specific BER attributes
        self.ef_ber_noise_count = 0
        self.ef_ber_total_points_for_ber = 0
        self.ef_ber_original_ber = np.nan

        # Ensure model_params has random_state if applicable for the model
        if self.model_name in ['svm', 'dt', 'rf', 'adaboost'] and 'random_state' not in self.model_params:
            # For models that accept random_state, pass it if not already in params
            # SVC doesn't always use it directly for determinism in the same way as ensemble/tree methods
            if self.model_name in ['dt', 'rf', 'adaboost']:
                 self.model_params['random_state'] = self.random_state


    def _initialize_sklearn_model(self):
        """Helper to create the scikit-learn model instance."""
        if self.model_name == 'svm':
            # Common SVM params: C, kernel, gamma. Add probability=True if needed for predict_proba
            return SVC(random_state=self.random_state, **self.model_params)
        elif self.model_name == 'dt':
            return DecisionTreeClassifier(**self.model_params) # random_state is in model_params
        elif self.model_name == 'rf':
            return RandomForestClassifier(**self.model_params) # random_state is in model_params
        elif self.model_name == 'adaboost':
            # AdaBoost typically uses DecisionTreeClassifier as a base estimator by default
            # If a different base_estimator is desired, it should be passed in model_params
            # For AdaBoost, n_estimators and learning_rate are common params.
            return AdaBoostClassifier(**self.model_params) # random_state is in model_params
        elif self.model_name == 'gnb':
            return GaussianNB(**self.model_params) # No random_state for GNB
        elif self.model_name == 'knn': # Direct KNN, not part of EF-BER
            return KNeighborsClassifier(n_neighbors=self.k_neighbors, **self.model_params)
        else:
            raise ValueError(f"Unsupported sklearn model_name: {self.model_name}")

    # --- EF-BER Specific Methods ---
    def _ef_ber_kmeans_bisection(self, data_points, target_clusters):
        """Performs K-means bisection (specific to EF-BER)."""
        if data_points is None or len(data_points) == 0:
            return []

        root_node = TreeNode(data=data_points)
        active_leaves = [root_node]
        
        while len(active_leaves) < target_clusters:
            splittable_leaves = [leaf for leaf in active_leaves if len(leaf) >= 2]
            if not splittable_leaves: break
            
            largest_leaf_node = max(splittable_leaves, key=len)
            active_leaves.remove(largest_leaf_node)
            cluster_to_split_data = largest_leaf_node.data
            
            if len(cluster_to_split_data) < 2: # Should be caught by splittable_leaves
                active_leaves.append(largest_leaf_node)
                continue

            initial_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
            centroid_1 = cluster_to_split_data[initial_indices[0]]
            centroid_2 = cluster_to_split_data[initial_indices[1]]
            
            max_iterations_2_means = 100
            convergence_threshold = 1e-6
            final_sub_cluster_1_data, final_sub_cluster_2_data = [], []

            for _ in range(max_iterations_2_means):
                current_sub_1, current_sub_2 = [], []
                for point in cluster_to_split_data:
                    if np.abs(point - centroid_1) <= np.abs(point - centroid_2):
                        current_sub_1.append(point)
                    else:
                        current_sub_2.append(point)
                
                if not current_sub_1 or not current_sub_2:
                    if _ < max_iterations_2_means - 1:
                        new_indices = np.random.choice(len(cluster_to_split_data), 2, replace=False)
                        centroid_1, centroid_2 = cluster_to_split_data[new_indices[0]], cluster_to_split_data[new_indices[1]]
                        continue
                    else:
                        final_sub_cluster_1_data, final_sub_cluster_2_data = [], []
                        break

                new_c1, new_c2 = np.mean(current_sub_1), np.mean(current_sub_2)
                if np.abs(new_c1 - centroid_1) < convergence_threshold and \
                   np.abs(new_c2 - centroid_2) < convergence_threshold:
                    final_sub_cluster_1_data, final_sub_cluster_2_data = current_sub_1, current_sub_2
                    break
                centroid_1, centroid_2 = new_c1, new_c2
                if _ == max_iterations_2_means - 1:
                    final_sub_cluster_1_data, final_sub_cluster_2_data = current_sub_1, current_sub_2
            
            if not final_sub_cluster_1_data or not final_sub_cluster_2_data:
                active_leaves.append(largest_leaf_node)
                continue

            largest_leaf_node.is_leaf = False
            child1 = TreeNode(data=np.array(final_sub_cluster_1_data), is_leaf=True)
            child2 = TreeNode(data=np.array(final_sub_cluster_2_data), is_leaf=True)
            largest_leaf_node.left_child, largest_leaf_node.right_child = child1, child2
            if len(child1) > 0: active_leaves.append(child1)
            if len(child2) > 0: active_leaves.append(child2)
        
        return [leaf.data for leaf in active_leaves if len(leaf.data) > 0]

    def _ef_ber_is_pure_cluster(self, cluster_data, purity_threshold=0.9):
        """Checks cluster purity (specific to EF-BER)."""
        if len(cluster_data) == 0: return False
        labels_in_cluster = [assign_label(point) for point in cluster_data]
        if not labels_in_cluster: return False
        label_counts = pd.Series(labels_in_cluster).value_counts()
        if label_counts.empty: return False
        purity = label_counts.iloc[0] / len(cluster_data)
        return purity >= purity_threshold

    # --- Training and Prediction ---
    def train(self, hardness_values, true_labels):
        """
        Trains the classifier. Includes 2-fold CV for BER estimation and
        trains a final model on the full dataset.
        """
        overall_start_time = time.time()
        
        # Ensure hardness_values are a 2D array for scikit-learn
        X_data = np.array(hardness_values).reshape(-1, 1)
        y_data = np.array(true_labels) # Labels are already 1D

        # --- Part 1: 2-Fold Cross-Validation for BER Estimation ---
        print(f"\n--- Starting 2-Fold Cross-Validation for '{self.model_name.upper()}' ---")
        kf = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
        fold_accuracies = []
        
        for fold_num, (train_indices, test_indices) in enumerate(kf.split(X_data)):
            current_fold_label = f"Fold {fold_num + 1}/2"
            print(f"\nProcessing {current_fold_label}...")
            
            X_train_fold, X_test_fold = X_data[train_indices], X_data[test_indices]
            y_train_fold, y_test_fold = y_data[train_indices], y_data[test_indices]

            if len(X_train_fold) == 0 or len(X_test_fold) == 0:
                print(f"Warning ({current_fold_label}): Training or testing set is empty. Skipping fold.")
                fold_accuracies.append(np.nan)
                continue

            fold_model = None
            if self.model_name == 'ef-ber':
                print(f"({current_fold_label}): EF-BER - Running K-means bisection on training data...")
                # For EF-BER, X_train_fold is 1D array of hardness values for bisection
                clusters_fold_train = self._ef_ber_kmeans_bisection(X_train_fold.flatten(), self.k_clusters)
                
                pure_X_fold, pure_y_fold = [], []
                for cluster_item in clusters_fold_train:
                    if self._ef_ber_is_pure_cluster(cluster_item):
                        for point in cluster_item:
                            pure_X_fold.append([point]) # KNN expects 2D
                            pure_y_fold.append(assign_label(point))
                
                if not pure_X_fold:
                    print(f"Warning ({current_fold_label}): EF-BER - No pure clusters. Using all fold training data for KNN.")
                    pure_X_fold = X_train_fold # Already 2D
                    pure_y_fold = y_train_fold # Corresponding labels
                
                if not pure_X_fold: # If pure_X_fold is still empty (e.g. X_train_fold was empty)
                    print(f"Error ({current_fold_label}): EF-BER - No data for KNN training. Skipping fold.")
                    fold_accuracies.append(np.nan)
                    continue
                
                fold_model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
                fold_model.fit(pure_X_fold, pure_y_fold)
            else: # Standard sklearn model
                print(f"({current_fold_label}): Training '{self.model_name.upper()}' model...")
                fold_model = self._initialize_sklearn_model()
                fold_model.fit(X_train_fold, y_train_fold)

            # Evaluate on the fold's test data
            if fold_model:
                predictions_test_fold = fold_model.predict(X_test_fold)
                accuracy_fold_test = accuracy_score(y_test_fold, predictions_test_fold)
                fold_accuracies.append(accuracy_fold_test)
                print(f"({current_fold_label}): Accuracy on test set = {accuracy_fold_test:.4f}")
            else: # Should not happen if checks above are fine
                fold_accuracies.append(np.nan)
        
        valid_fold_accuracies = [acc for acc in fold_accuracies if not np.isnan(acc)]
        if valid_fold_accuracies:
            mean_cv_accuracy = np.mean(valid_fold_accuracies)
            self.cv_ber_estimate = 1 - mean_cv_accuracy
            print(f"\n--- CV Summary for '{self.model_name.upper()}' ---")
            print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
            print(f"Estimated BER from CV (1 - Mean Accuracy): {self.cv_ber_estimate:.4f}")
        else:
            print(f"\n--- CV Warning for '{self.model_name.upper()}' ---")
            print("Cross-validation did not produce valid accuracy scores.")
            self.cv_ber_estimate = np.nan

        # --- Part 2: Train Final Model on the Full Dataset ---
        print(f"\n--- Training Final '{self.model_name.upper()}' Model on Full Dataset ---")
        
        if self.model_name == 'ef-ber':
            print("EF-BER: Running K-means bisection on full dataset...")
            # X_data.flatten() to pass 1D array of hardness values for bisection
            self.ef_ber_clusters = self._ef_ber_kmeans_bisection(X_data.flatten(), self.k_clusters)
            
            pure_X_full, pure_y_full = [], []
            mixed_clusters_full_data = [] # For EF-BER's original BER calculation

            for cluster_item in self.ef_ber_clusters:
                if self._ef_ber_is_pure_cluster(cluster_item):
                    for point in cluster_item:
                        pure_X_full.append([point])
                        pure_y_full.append(assign_label(point))
                else:
                    mixed_clusters_full_data.append(cluster_item) # Store mixed cluster data
            
            if not pure_X_full:
                print("Warning (EF-BER Full Dataset): No pure clusters. Using all data for final KNN.")
                pure_X_full = X_data # Use full X_data (already 2D)
                pure_y_full = y_data # Use full y_data
            
            if not pure_X_full.size: # Check if pure_X_full is empty (as numpy array)
                 print("Error (EF-BER Full Dataset): No data for final KNN training. EF-BER model not trained.")
                 self.model = None
            else:
                self.model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
                self.model.fit(pure_X_full, pure_y_full)
                print("EF-BER: Final KNN model trained.")
                # Calculate EF-BER specific BER
                self.ef_ber_noise_count = 0
                num_points_in_mixed_ef_ber = 0
                for mixed_cluster_item in mixed_clusters_full_data:
                    num_points_in_mixed_ef_ber += len(mixed_cluster_item)
                    for point_val in mixed_cluster_item:
                        actual_lbl = assign_label(point_val)
                        predicted_lbl = self.model.predict(np.array([[point_val]]))[0]
                        if actual_lbl != predicted_lbl:
                            self.ef_ber_noise_count += 1
                
                self.ef_ber_total_points_for_ber = len(pure_X_full) + num_points_in_mixed_ef_ber
                if self.ef_ber_total_points_for_ber > 0:
                    self.ef_ber_original_ber = self.ef_ber_noise_count / self.ef_ber_total_points_for_ber
                else:
                    self.ef_ber_original_ber = np.nan
        else: # Standard sklearn model
            self.model = self._initialize_sklearn_model()
            self.model.fit(X_data, y_data)
            print(f"Final '{self.model_name.upper()}' model trained on full dataset.")

        # Evaluate final model on the full training set (as a measure of fit)
        if self.model:
            predictions_full_train = self.model.predict(X_data)
            self.final_model_accuracy = accuracy_score(y_data, predictions_full_train)
            print(f"Accuracy of final '{self.model_name.upper()}' model on full training set: {self.final_model_accuracy:.4f}")
        else:
            print(f"Warning: Final model for '{self.model_name.upper()}' was not trained successfully.")
            self.final_model_accuracy = np.nan
            
        overall_end_time = time.time()
        print(f"\nTotal training process for '{self.model_name.upper()}' completed in {overall_end_time - overall_start_time:.4f} seconds.")

    def predict(self, X_new_samples):
        """Predicts categories for new samples."""
        if self.model is None:
            raise Exception(f"Model '{self.model_name.upper()}' not trained yet or training failed.")
        
        X_reshaped = np.array(X_new_samples).reshape(-1, 1)
        return self.model.predict(X_reshaped)

    def visualize_clusters(self, original_hardness_data):
        """Visualizes data distribution and EF-BER clusters if applicable."""
        plt.figure(figsize=(14, 7))

        # Plot 1: Histogram of original data
        plt.subplot(1, 2, 1)
        plt.hist(original_hardness_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        boundaries = {'Soft/Moderate': 100, 'Moderate/Hard': 175, 'Hard/Very Hard': 250}
        colors = {'Soft/Moderate': 'darkgreen', 'Moderate/Hard': 'goldenrod', 'Hard/Very Hard': 'darkred'}
        for label, val in boundaries.items():
            plt.axvline(x=val, color=colors[label], linestyle='--', label=f'{label} ({val} ppm)')
        plt.title('Overall Water Hardness Distribution')
        plt.xlabel('Hardness (ppm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)

        # Plot 2: EF-BER Clusters (if applicable)
        plt.subplot(1, 2, 2)
        if self.model_name == 'ef-ber' and self.ef_ber_clusters:
            cluster_plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for i, cluster_data in enumerate(self.ef_ber_clusters):
                if len(cluster_data) > 0:
                    y_jitter = np.random.normal(loc=i + 1, scale=0.1, size=len(cluster_data))
                    plt.scatter(cluster_data, y_jitter,
                                color=cluster_plot_colors[i % len(cluster_plot_colors)],
                                label=f'Cluster {i+1} ({len(cluster_data)} pts)', alpha=0.7, s=50)
            plt.title('EF-BER: Clusters from K-means Bisection (Full Dataset)')
            plt.xlabel('Hardness (ppm)')
            plt.ylabel('Cluster Index (Y-axis for separation)')
            if self.ef_ber_clusters: plt.legend(title="EF-BER Clusters", loc="best")
            for val in boundaries.values(): plt.axvline(x=val, color='gray', linestyle=':', alpha=0.6)
        else:
            # For non-EF-BER models, or if EF-BER clusters aren't available
            # We can plot the data points colored by their true labels or predicted labels by the final model
            if self.model: # If a final model is trained
                predictions_viz = self.model.predict(np.array(original_hardness_data).reshape(-1,1))
                unique_labels_viz = sorted(list(set(predictions_viz)))
                label_to_color_map = {lbl: cluster_plot_colors[i % len(cluster_plot_colors)] for i, lbl in enumerate(unique_labels_viz)}

                for lbl in unique_labels_viz:
                    points_for_label = original_hardness_data[predictions_viz == lbl]
                    if len(points_for_label) > 0:
                         # Create some y-jitter based on label for visualization
                        y_jitter_val = unique_labels_viz.index(lbl) + 1
                        y_jitter = np.random.normal(loc=y_jitter_val, scale=0.1, size=len(points_for_label))
                        plt.scatter(points_for_label, y_jitter, color=label_to_color_map[lbl], label=f'Predicted: {lbl}', alpha=0.6, s=30)
                plt.title(f'Data Points Colored by Predicted Labels ({self.model_name.upper()})')
                plt.legend(title="Predicted Labels", loc="best")

            else: # Fallback if no model trained
                plt.text(0.5, 0.5, f'Visualization for "{self.model_name.upper()}" model.\n(Cluster plot is specific to EF-BER)',
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.xlabel('Hardness (ppm)')
            plt.grid(True, linestyle=':', alpha=0.5)


        plt.tight_layout()
        plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Comprehensive Water Hardness Classification Script ---")
    
    hardness_data_values = None
    data_source_name = ""
    primary_csv_path = 'water_potability.csv'

    try:
        print(f"\nAttempting to load data from '{primary_csv_path}'...")
        df = pd.read_csv(primary_csv_path)
        data_source_name = primary_csv_path
        if 'Hardness' not in df.columns:
            print(f"Error: 'Hardness' column not found in '{primary_csv_path}'.")
            df = None
        else:
            original_count = len(df)
            df.dropna(subset=['Hardness'], inplace=True)
            if original_count > len(df): print(f"Removed {original_count - len(df)} rows with NaN Hardness.")
            hardness_data_values = df['Hardness'].values
            if len(hardness_data_values) == 0: hardness_data_values = None
    except FileNotFoundError:
        print(f"Warning: Primary data file '{primary_csv_path}' not found.")
    except Exception as e:
        print(f"Error loading '{primary_csv_path}': {e}")

    if hardness_data_values is None or len(hardness_data_values) < 20: # Need enough for CV splits
        if hardness_data_values is not None and len(hardness_data_values) < 20 :
             print(f"Loaded data from {data_source_name} has only {len(hardness_data_values)} samples. Generating random data for a better demo.")
        else:
             print("Proceeding to generate random data for demonstration.")
        np.random.seed(42)
        hardness_data_values = np.concatenate([
            np.random.normal(loc=80, scale=20, size=150), np.random.normal(loc=150, scale=25, size=150),
            np.random.normal(loc=220, scale=30, size=100), np.random.normal(loc=300, scale=35, size=50)
        ])
        hardness_data_values = hardness_data_values[hardness_data_values > 0]
        hardness_data_values = np.clip(hardness_data_values, 1, 400)
        data_source_name = "Randomly Generated Data"
        if len(hardness_data_values) < 20:
            print("Critical Error: Failed to generate sufficient random data. Exiting.")
            exit()
        print(f"Generated {len(hardness_data_values)} random data points.")

    # Prepare labels
    true_labels_list = [assign_label(h) for h in hardness_data_values]

    print(f"\n--- Data Summary ({data_source_name}) ---")
    print(f"Number of data points: {len(hardness_data_values)}")
    print(f"Mean hardness: {hardness_data_values.mean():.2f} ppm, Std dev: {hardness_data_values.std():.2f} ppm")
    print(f"Min hardness: {hardness_data_values.min():.2f} ppm, Max hardness: {hardness_data_values.max():.2f} ppm")
    label_dist = pd.Series(true_labels_list).value_counts(normalize=True) * 100
    print("Label Distribution:\n", label_dist.to_string(float_format="%.1f%%"))


    # --- Model Selection ---
    print("\n--- Model Selection ---")
    model_choices = {
        '1': 'ef-ber', '2': 'svm', '3': 'dt', 
        '4': 'rf', '5': 'adaboost', '6': 'gnb', '7': 'knn'
    }
    print("Available models:")
    for key, name in model_choices.items():
        print(f"  {key}. {name.upper()}")
    
    chosen_model_key = input(f"Enter the number of the model to use (1-{len(model_choices)}): ")
    selected_model_name = model_choices.get(chosen_model_key)

    if not selected_model_name:
        print("Invalid choice. Defaulting to 'ef-ber'.")
        selected_model_name = 'ef-ber'
    
    print(f"You selected: {selected_model_name.upper()}")

    # --- Initialize and Train Classifier ---
    # Example of passing model-specific parameters:
    model_specific_params = {}
    if selected_model_name == 'svm':
        model_specific_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'} # Example SVM params
    elif selected_model_name == 'rf':
        model_specific_params = {'n_estimators': 100, 'max_depth': None} # Example RF params
    # Add more params for other models as needed

    classifier = HardnessClassifier(model_name=selected_model_name, 
                                    k_clusters=4, k_neighbors=3, # Relevant for EF-BER or KNN
                                    random_state=42, 
                                    **model_specific_params)
    
    try:
        classifier.train(hardness_data_values, true_labels_list)
        
        print(f"\n--- Results for {selected_model_name.upper()} ---")
        print(f"Estimated BER from 2-Fold CV: {classifier.cv_ber_estimate:.4f}")
        print(f"Accuracy of Final Model on Full Training Set: {classifier.final_model_accuracy:.4f}")
        if selected_model_name == 'ef-ber':
            print(f"EF-BER Original Method BER: {classifier.ef_ber_original_ber:.4f}")
            print(f"  (Based on {classifier.ef_ber_noise_count} noise points out of {classifier.ef_ber_total_points_for_ber} considered points)")

        classifier.visualize_clusters(hardness_data_values)
        
        if classifier.model:
            print("\n--- Example Predictions on New Samples ---")
            new_samples = np.array([45, 90, 110, 170, 200, 260, 310])
            predictions = classifier.predict(new_samples)
            for sample, pred_label in zip(new_samples, predictions):
                actual_label = assign_label(sample)
                print(f"Hardness: {sample} ppm -> Predicted: {pred_label} (Actual: {actual_label})")
        else:
            print("\nWarning: Final model not available for predictions.")

    except Exception as e:
        print(f"\n--- An Error Occurred During {selected_model_name.upper()} Processing ---")
        print(f"Error details: {e}")
        print("Traceback:")
        traceback.print_exc()

    print("\n--- Script Finished ---")
