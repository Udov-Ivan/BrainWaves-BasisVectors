import sys
!{sys.executable} -m pip install mne

import numpy as np
import mne
from mne.datasets import eegbci
from sage.all import *


# =================================================================
# PHASE 1: LOAD REAL EEG DATA
# =================================================================
# Load subject 1, run 1 (Baseline)
edf_files = eegbci.load_data(subjects=int(1), runs=int(1))

# FIX: Pass the first file in the list [0] to the reader
raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)

# Select 3 channels: Fc1/Fc2 (Frontal) and O1 (Occipital)
# Note: In this specific dataset, channel names often end with a dot
raw.pick_channels(['Fc1.', 'Fc2.', 'O1..']) 

real_data = raw.get_data().T  # Shape: (Time, 3)
fs = float(raw.info['sfreq'])

# CONVERSION: Convert from Volts to Microvolts (uV) 
real_data_uv = real_data * 1e6 

# Select 3 channels: Fc1/Fc2 (Frontal) and O1 (Occipital)
raw.pick_channels(['Fc1.', 'Fc2.', 'O1..']) 
real_data = raw.get_data().T  # Shape: (Time, 3)
fs = float(raw.info['sfreq'])

# CONVERSION: Convert from Volts (1e-6) to Microvolts (uV) 
# This makes the values large enough to see in the 3D space
real_data_uv = real_data * 1e6 

# =================================================================
# PHASE 2: SPATIAL FILTERING & EMPIRICAL BASIS
# =================================================================
# 1. Prepare data for Sage
raw_slice = np.ascontiguousarray(real_data_uv[:1000, :])
points_raw_mat = matrix(RDF, raw_slice)

# 2. CENTER THE DATA: Calculate mean manually for each column (axis=0)
# We sum each column and divide by the number of rows (1000)
num_rows = points_raw_mat.nrows()
mean_vec = [sum(points_raw_mat.column(j)) / num_rows for j in range(points_raw_mat.ncols())]

# Subtract the mean vector from every row
centered_data = points_raw_mat - matrix(RDF, [mean_vec] * num_rows)

# 3. Calculate Empirical Covariance Matrix
cov_matrix = (centered_data.transpose() * centered_data) / num_rows

# 4. Extract Eigenvectors
eigen_sys = cov_matrix.eigenvectors_right()
eigen_sys.sort(key=lambda x: x[0], reverse=True)

# 5. Create Projection Matrix P = I - v*v^T
val1, vecs1, _ = eigen_sys[0]
v_noise = matrix(RDF, vecs1[0]).transpose()
v_norm = v_noise / v_noise.norm()
P = identity_matrix(RDF, 3) - v_norm * v_norm.transpose()

# --- CREATE VISUAL BASIS VECTORS ---
vectors_plot = Graphics()
colors = ['red', 'orange', 'purple'] 
for i, (val, vecs_list, _) in enumerate(eigen_sys):
    vec = vecs_list[0]
    v_end = vector(vec) * sqrt(abs(val)) * 1.5 
    vectors_plot += arrow3d((0,0,0), v_end, color=colors[i], radius=0.4)
    vectors_plot += text3d(f"v{i+1}", v_end*1.2, color=colors[i])

# 6. Apply Filtering
points_cleaned = centered_data * P

# --- DISPLAY ---
p1 = point3d(centered_data.rows(), size=10, color='blue', opacity=0.4) + vectors_plot
p2 = point3d(points_cleaned.rows(), size=10, color='green', opacity=0.4) + vectors_plot




# =================================================================
# PHASE 4: SPECTRAL ANALYSIS
# =================================================================
def calculate_psd(signal, fs):
    sig = np.array(signal, dtype=float).flatten()
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), d=1/fs)
    mask = freqs > 0
    return freqs[mask], np.abs(fft_vals[mask])**2

# Analyze the real O1 and Frontal channels
f_axis, psd_o1 = calculate_psd(points_cleaned.column(2), fs)
_, psd_fp = calculate_psd(points_cleaned.column(1), fs)

# Create Plots
spec_o1 = line(zip(f_axis, psd_o1), color='darkred', title="O1 (Real Data) Spectrum")
zoom_o1 = line(zip(f_axis, psd_o1), color='darkred', thickness=2, title="O1: Alpha Band")
zoom_o1.set_axes_range(7, 13, 0, max(psd_o1)*1.1)

# =================================================================
# DISPLAY
# =================================================================
print("DISPLAYING PHASE 2: 3D Projection of Real EEG Data")
show(p1, title="Raw MNE Data Cluster", frame=True)
show(p2, title="Cleaned MNE Data Cluster", frame=True)

print("DISPLAYING PHASE 4: Spectral Analysis of Real Data")
show(graphics_array([spec_o1, zoom_o1]), figsize=(12, 4))