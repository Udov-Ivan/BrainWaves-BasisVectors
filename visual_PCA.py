import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define our mathematical Covariance matrix
cov = np.array([[100, 95, 0],
                [95, 100, 0],
                [0, 0, 2]])
mean = [0, 0, 0]
eigenvalues, eigenvectors = np.linalg.eigh(cov)


# Generate 1000 data points (time samples) mimicking this exact geometry
np.random.seed(42)
data = np.random.multivariate_normal(mean, cov, 1000)
data_new_base = data @ eigenvectors
data_new_base[:, 2] = 0
data_new_base = data_new_base @ eigenvectors.T

pd.DataFrame(data).to_json("raw_data.json")
pd.DataFrame(data_new_base).to_json("new_data.json")
# Extract axes
Rp1 = data[:, 0]
Rp2 = data[:, 1]
Q1 = data[:, 2]

Fp1 = data_new_base[:, 0]
Fp2 = data_new_base[:, 1]
O1 = data_new_base[:, 2]

# Plotting the Point Cloud
fig = plt.figure(figsize=(10, 8))
fig1 = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
bx = fig1.add_subplot(111, projection='3d')
ax.scatter(Fp1, Fp2, O1, alpha=0.3, color='blue', label='EEG Time Samples (Point Cloud)')
bx.scatter(Rp1, Rp2, Q1, alpha=0.3, color='purple', label='EEG Time Samples (Point Cloud)')
# Plotting the Eigenvectors (scaled by the square root of eigenvalues for visual length)


colors = ['green', 'orange', 'red']
labels = ['PC3 (Brainwaves: λ=2)', 'PC2 (Local Noise: λ=5)', 'PC1 (Eye Blink: λ=195)']

# eigh returns eigenvalues in ascending order, so index 2 is the largest
for i in range(3):
    vec = eigenvectors[:, i]
    val = eigenvalues[i]
    length = np.sqrt(val) * 2  # Scaling for visibility
    ax.quiver(0, 0, 0, vec[0]*length, vec[1]*length, vec[2]*length,
              color=colors[i], linewidth=4, label=labels[i])
    bx.quiver(0, 0, 0, vec[0] * length, vec[1] * length, vec[2] * length,
              color=colors[i], linewidth=4, label=labels[i])

ax.set_xlabel('Fp1 Voltage')
ax.set_ylabel('Fp2 Voltage')
ax.set_zlabel('O1 Voltage')
ax.set_title('3D Geometric View of EEG Covariance and Eigenvectors')
ax.legend()
bx.set_xlabel('Fp1 Voltage')
bx.set_ylabel('Fp2 Voltage')
bx.set_zlabel('O1 Voltage')
bx.set_title('3D Geometric View of EEG Covariance and Eigenvectors')
bx.legend()
plt.show()