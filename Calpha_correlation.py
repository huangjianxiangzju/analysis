import mdtraj as md
import numpy as np
from scipy.stats import pearsonr

# Load trajectory and topology (assuming .dcd trajectory and .pdb topology files)
trajectory = md.load('trajectory.dcd', top='topology.pdb')

# Select Cα atoms
ca_indices = trajectory.topology.select('name CA')
num_atoms = len(ca_indices)
num_frames = trajectory.n_frames

# Step 1: Extract Cα coordinates over all frames
ca_coordinates = trajectory.atom_slice(ca_indices).xyz  # Shape: (num_frames, num_atoms, 3)

# Step 2: Calculate pairwise distances for each frame
distance_matrix = np.zeros((num_frames, num_atoms, num_atoms))
for frame_idx in range(num_frames):
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):  # Only upper triangle
            dist = np.linalg.norm(ca_coordinates[frame_idx, i] - ca_coordinates[frame_idx, j])
            distance_matrix[frame_idx, i, j] = dist
            distance_matrix[frame_idx, j, i] = dist  # Symmetric

# Step 3: Compute Pearson correlation between distance pairs across frames
correlation_matrix = np.zeros((num_atoms, num_atoms))
for i in range(num_atoms):
    for j in range(i+1, num_atoms):
        # Extract time series of distances between Cα atoms i and j
        distances_i_j = distance_matrix[:, i, j]
        
        # Calculate Pearson correlation
        correlation, _ = pearsonr(distances_i_j, distances_i_j)
        correlation_matrix[i, j] = correlation
        correlation_matrix[j, i] = correlation  # Symmetric

# The result is a correlation matrix showing correlation coefficients for each pair of Cα atoms
print("Correlation matrix of Cα atoms:")
print(correlation_matrix)
