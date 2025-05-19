import numpy as np
from sklearn.cluster import KMeans
import warnings
from sklearn.neighbors import NearestNeighbors

# Constants from the lab manual
EQUIVALENT_WEIGHT_MGSO4 = 123  # Equivalent weight of MgSO4·7H2O
EQUIVALENT_WEIGHT_CACO3 = 50   # Equivalent weight of CaCO3
SIGMA = 0.2                    # Standard deviation for titration readings (mL)
N_READINGS = 10                # Number of titration readings for EF-BER

def apply_ef_ber(data, sigma=SIGMA):
    """
    Apply EF-BER method to estimate weighted mean and uncertainty.
    
    Based on the EF-BER method from the research paper which transforms BER estimation
    into a noise identification problem using clustering and nearest neighbor analysis.
    """
    data = np.array(data).reshape(-1, 1)
    
    # Step 1: Construct homogeneous clusters using k-means
    # For this application, we use 2 clusters to identify potential outliers
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.flatten()
    
    # Step 2: Identify potential noise samples using nearest neighbors
    # Use approximate k-nearest neighbor algorithm (k=3 for this small dataset)
    k = min(3, len(data) - 1)  # Ensure k is not larger than dataset size - 1
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 because it includes the point itself
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    
    # Remove self-distance (first column)
    neighbor_distances = distances[:, 1:]
    
    # Calculate mean distance to neighbors for each point
    mean_distances = np.mean(neighbor_distances, axis=1)
    
    # Calculate noise probability based on distance to cluster centroid and neighbors
    centroid_distances = np.abs(data.flatten() - centroids[labels])
    
    # Normalize distances for better scaling
    normalized_centroid_dist = centroid_distances / np.max(centroid_distances) if np.max(centroid_distances) > 0 else centroid_distances
    normalized_neighbor_dist = mean_distances / np.max(mean_distances) if np.max(mean_distances) > 0 else mean_distances
    
    # Calculate noise probability (combining cluster distance and neighbor information)
    P_noise = 1 - np.exp(-(normalized_centroid_dist**2 + normalized_neighbor_dist**2) / (2 * sigma**2))
    P_reliable = 1 - P_noise
    
    # Ensure P_reliable is not zero to avoid division by zero
    if np.sum(P_reliable) == 0:
        warnings.warn("All data points have zero reliability, using simple mean instead.")
        return np.mean(data), np.std(data)
    
    # Compute weighted mean and variance using reliability probabilities
    weighted_mean = np.sum(data.flatten() * P_reliable) / np.sum(P_reliable)
    weighted_variance = np.sum(P_reliable * (data.flatten() - weighted_mean)**2) / np.sum(P_reliable)
    weighted_std = np.sqrt(weighted_variance)
    
    return weighted_mean, weighted_std

def main():
    # Step 1: Preparation of standard MgSO4·7H2O solution
    print("Step 1: Preparation of standard MgSO4·7H2O solution")
    try:
        w1 = float(input("Enter weight of MgSO4·7H2O salt + weighing bottle (W1, g): "))
        w2 = float(input("Enter weight of empty weighing bottle (W2, g): "))
        weight_mgso4 = w1 - w2
        print(f"Weight of MgSO4·7H2O salt transferred (W1 - W2): {weight_mgso4:.4f} g")
        
        # Calculate normality of MgSO4·7H2O solution
        normality_mgso4 = (weight_mgso4 / EQUIVALENT_WEIGHT_MGSO4) * (1000 / 100)
        print(f"Normality of MgSO4·7H2O solution: {normality_mgso4:.4f} N")
        
        # Step 2: Standardization of EDTA solution
        print("\nStep 2: Standardization of EDTA solution")
        volume_mgso4 = float(input("Enter volume of MgSO4·7H2O solution pipetted (V1, mL): "))
        
        # Sample titration readings for EDTA volume
        edta_volumes = [21.2, 21.1, 21.3, 21.2, 21.0, 21.3, 21.1, 22.0, 21.2, 21.1]
        print(f"Titration readings for EDTA volume: {edta_volumes}")
        
        # Apply EF-BER to EDTA titration data
        mean_edta_volume, std_edta_volume = apply_ef_ber(edta_volumes)
        print(f"EF-BER estimated EDTA volume: {mean_edta_volume:.2f} mL ± {std_edta_volume:.2f} mL")
        
        # Calculate normality of EDTA solution
        normality_edta = (normality_mgso4 * volume_mgso4) / mean_edta_volume
        print(f"Normality of EDTA solution (N2): {normality_edta:.4f} N")
        
        # Step 3: Estimation of total hardness of water sample
        print("\nStep 3: Estimation of total hardness of water sample")
        volume_water = float(input("Enter volume of water sample pipetted (mL): "))
        
        # Sample titration readings for water sample
        water_readings = [16.5, 16.4, 16.6, 16.5, 16.3, 16.6, 16.4, 17.2, 16.5, 16.6]
        print(f"Titration readings for water sample: {water_readings}")
        
        # Apply EF-BER to water sample EDTA volumes
        mean_edta_water, std_edta_water = apply_ef_ber(water_readings)
        print(f"EF-BER estimated volume for water sample: {mean_edta_water:.2f} mL ± {std_edta_water:.2f} mL")
        
        # Calculate total hardness
        hardness = (normality_edta * mean_edta_water * EQUIVALENT_WEIGHT_CACO3 * 1000) / volume_water
        # Calculate uncertainty in hardness
        hardness_uncertainty = (std_edta_water / mean_edta_water) * hardness
        print(f"Total hardness of water sample (as CaCO3 equivalents): {hardness:.2f} ± {hardness_uncertainty:.2f} ppm")
        
        # Step 4: Percentage error calculation
        print("\nStep 4: Percentage error calculation")
        try:
            given_hardness = float(input("Enter given hardness value (ppm): "))
            reported_hardness = hardness
            percent_error = abs(given_hardness - reported_hardness) / given_hardness * 100
            print(f"Reported hardness: {reported_hardness:.2f} ppm")
            print(f"Given hardness: {given_hardness:.2f} ppm")
            print(f"Percentage error: {percent_error:.2f}%")
            
            # Summary of results
            print("\nResult Summary:")
            print(f"Amount of total hardness in the water sample: {hardness:.2f} ppm")
            print(f"Uncertainty in hardness (based on EF-BER): ±{hardness_uncertainty:.2f} ppm")
            print(f"Percentage error: {percent_error:.2f}%")
        except ValueError:
            print("No given hardness value provided. Skipping percentage error calculation.")
            
            # Summary of results without percentage error
            print("\nResult Summary:")
            print(f"Amount of total hardness in the water sample: {hardness:.2f} ppm")
            print(f"Uncertainty in hardness (based on EF-BER): ±{hardness_uncertainty:.2f} ppm")
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure all inputs are valid numbers.")

if __name__ == "__main__":
    main()