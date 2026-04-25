import numpy as np
from sage.all import *

# =================================================================
# PHASE 2: SPATIAL FILTERING & BASIS VECTORS
# =================================================================
# 1. Define the Theoretical Covariance Matrix
cov_matrix = matrix(RDF, [[100, 95, 0], [95, 100, 0], [0, 0, 2]])

# 2. Extract Eigenvectors (Basis Vectors)
eigen_sys = cov_matrix.eigenvectors_right()
eigen_sys.sort(key=lambda x: x[0], reverse=True)

# 3. Create the Projection Matrix (P = I - v*v^T)
# This mathematical "surgery" removes the subspace defined by the noise vector
lambda1, vecs, _ = eigen_sys[0]
v_noise = matrix(RDF, vecs[0]).transpose()
v_norm = v_noise / v_noise.norm()
P = identity_matrix(RDF, 3) - v_norm * v_norm.transpose()

# --- CREATE VISUAL BASIS VECTORS ---
vectors_plot = Graphics()
colors = ['red', 'orange', 'purple'] 
for i, (val, vecs_list, _) in enumerate(eigen_sys):
    vec = vecs_list[0] # Get the first vector for this eigenvalue
    v_end = vector(vec) * sqrt(abs(val)) 
    vectors_plot += arrow3d((0,0,0), v_end, color=colors[i], radius=0.1)
    vectors_plot += text3d(f"v{i+1}", v_end*1.1, color=colors[i])

# 4. Generate Synthetic EEG Data
np.random.seed(42)
fs = 100.0
t = np.arange(1000) / fs

# Ensure data types are compatible with NumPy (convert Sage types to float/int)
mean_vec = [float(0), float(0), float(0)]
cov_np = cov_matrix.numpy().astype(float)
num_samples = int(1000)

raw_data_np = np.random.multivariate_normal(mean_vec, cov_np, num_samples)

# 5. INJECTION: Add hidden 10 Hz Alpha Rhythm to O1 channel (Index 2)
alpha_signal = np.cos(2 * np.pi * 10 * t) * 1.5
raw_data_np[:, 2] += alpha_signal

# Convert to Sage Matrix and apply the Projection (Filtering)
points_raw = matrix(RDF, raw_data_np)
points_cleaned = points_raw * P

# --- 3D VISUALIZATION ---
# Plot raw data with large dots and eigenvectors
p1 = point3d(points_raw.rows(), size=12, color='blue', opacity=0.6)
p1 += vectors_plot

# Plot cleaned data and add a text label in the 3D space
p2 = point3d(points_cleaned.rows(), size=12, color='green', opacity=0.6)
p2 += vectors_plot
p2 += text3d("CLEANED SUBSPACE", (0, 0, 15), color="darkgreen", fontsize=20)

# =================================================================
# PHASE 4: SPECTRAL ANALYSIS
# =================================================================
def calculate_psd(signal, fs):
    """Calculates Power Spectral Density using FFT."""
    sig = np.array(signal, dtype=float).flatten()
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), d=1/float(fs))
    mask = freqs > 0
    return freqs[mask], np.abs(fft_vals[mask])**2

f_axis, psd_o1 = calculate_psd(points_cleaned.column(2), fs)
_, psd_fp2 = calculate_psd(points_cleaned.column(1), fs)

# Create Spectral Plots
spec_o1 = line(zip(f_axis, psd_o1), color='darkred', title="O1: Global Spectrum")
zoom_o1 = line(zip(f_axis, psd_o1), color='darkred', thickness=2, title="O1: Alpha Band")
zoom_o1 += line([(10, 0), (10, max(psd_o1))], color='blue', linestyle="--")

spec_fp2 = line(zip(f_axis, psd_fp2), color='indigo', title="Fp2: Global Spectrum")
zoom_fp2 = line(zip(f_axis, psd_fp2), color='indigo', thickness=2, title="Fp2: Alpha Band")

# Set consistent ranges for comparison
for p in [spec_o1, spec_fp2]: p.set_axes_range(0, 40, 0, max(psd_o1)*1.1)
for p in [zoom_o1, zoom_fp2]: p.set_axes_range(7, 13, 0, max(psd_o1)*1.1)

# =================================================================
# DISPLAY RESULTS
# =================================================================
print("DISPLAYING PHASE 2: 3D Projection with Basis Vectors")
show(p1, title="3D Projection: Raw Data (Blink Axis in Red)", frame=True)
show(p2, title="3D Projection: Cleaned Data (Blink Removed)", frame=True)

print("DISPLAYING PHASE 4: Comparative Spectral Analysis")
spectral_matrix = graphics_array([[spec_o1, zoom_o1], [spec_fp2, zoom_fp2]])
show(spectral_matrix, axes_labels=['Hz', 'Power'], figsize=(12, 8))