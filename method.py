import numpy as np
from sklearn.cluster import KMeans

# Constants from the lab manual
EQUIVALENT_WEIGHT_MGSO4 = 123  # Equivalent weight of MgSO4·7H2O
EQUIVALENT_WEIGHT_CACO3 = 50   # Equivalent weight of CaCO3
SIGMA = 0.2                    # Standard deviation for titration readings (mL)
N_READINGS = 10                # Number of titration readings for EF-BER

def apply_ef_ber(data, sigma=SIGMA):
    """Apply EF-BER method to estimate weighted mean and uncertainty."""
    data = np.array(data).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.flatten()
    distances = np.abs(data.flatten() - centroids[labels])
    
    # Calculate noise and reliability probabilities
    P_noise = 1 - np.exp(-distances**2 / (2 * sigma**2))
    P_reliable = 1 - P_noise
    
    # Compute weighted mean and variance
    weighted_mean = np.sum(data.flatten() * P_reliable) / np.sum(P_reliable)
    weighted_variance = np.sum(P_reliable * (data.flatten() - weighted_mean)**2) / np.sum(P_reliable)
    weighted_std = np.sqrt(weighted_variance)
    
    return weighted_mean, weighted_std

# Step 1: Preparation of standard MgSO4·7H2O solution
print("Step 1: Preparation of standard MgSO4·7H2O solution")
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
print(f"Enter {N_READINGS} titration readings for EDTA volume:")
edta_volumes = [21.2,21.1,21.3,21.2,21.0,21.3,21.1,22,21.2,21.1]
# for i in range(N_READINGS):
#     initial = float(input(f"Titration {i+1} initial burette reading (mL): "))
#     final = float(input(f"Titration {i+1} final burette reading (mL): "))
#     volume = final - initial
#     edta_volumes.append(volume)
#     print(f"Volume of EDTA for titration {i+1}: {volume:.2f} mL")

# Calculate normality of EDTA solution
normality_edta = (normality_mgso4 * volume_mgso4) / edta_volumes
print(f"Normality of EDTA solution (N2): {normality_edta:.4f} N")

# Step 3: Estimation of total hardness of water sample
print("\nStep 3: Estimation of total hardness of water sample")
volume_water = float(input("Enter volume of water sample pipetted (mL): "))
print(f"Enter {N_READINGS} titration readings for water sample EDTA volume:")
edta_volumes_water = []
# for i in range(N_READINGS):
#     initial = float(input(f"Water titration {i+1} initial burette reading (mL): "))
#     final = float(input(f"Water titration {i+1} final burette reading (mL): "))
#     volume = final - initial
#     edta_volumes_water.append(volume)
#     print(f"Volume of EDTA for water titration {i+1}: {volume:.2f} mL")

# Apply EF-BER to water sample EDTA volumes
# Create a list of your titration readings
readings = [16.5,16.4,16.6,16.5,16.3,16.6,16.4,17.2,16.5,16.6]  # Replace with your actual readings

# Call the apply_ef_ber function
weighted_mean, weighted_std = apply_ef_ber(readings)

# Print the results
print(f"EF-BER estimated volume: {weighted_mean:.2f} mL ± {weighted_std:.2f} mL")

# Calculate total hardness
hardness = (normality_edta * volume_edta_water * EQUIVALENT_WEIGHT_CACO3 * 1000) / volume_water
print(f"Total hardness of water sample (as CaCO3 equivalents): {hardness:.2f} ppm")

# Step 4: Percentage error calculation
print("\nStep 4: Percentage error calculation")
given_hardness = float(input("Enter given hardness value (ppm): "))
reported_hardness = hardness
percent_error = abs(given_hardness - reported_hardness) / given_hardness * 100
print(f"Reported hardness: {reported_hardness:.2f} ppm")
print(f"Given hardness: {given_hardness:.2f} ppm")
print(f"Percentage error: {percent_error:.2f}%")

# Summary of results
print("\nResult Summary:")
print(f"Amount of total hardness in the water sample: {hardness:.2f} ppm")
print(f"Uncertainty in hardness (based on V3 uncertainty): ±{(uncertainty_edta_water / volume_edta_water * hardness):.2f} ppm")
print(f"Percentage error: {percent_error:.2f}%")