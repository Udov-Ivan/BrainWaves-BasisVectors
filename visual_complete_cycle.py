import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Define our mathematical Covariance matrix
cov = np.array([[100, 95, 0],
                [95, 100, 0],
                [0, 0, 2]])
mean = [0, 0, 0]
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# 2. Setup Time and Frequencies for EEG simulation
fs = 100  # Sampling rate in Hz (100 samples per second)
t = np.arange(1000) / fs  # 10 seconds of data

# 3. Generate 1000 data points (time samples) mimicking this exact geometry
np.random.seed(42)
data = np.random.multivariate_normal(mean, cov, 1000)

# INJECTION: Add a hidden 10 Hz Alpha wave to the O1 channel (index 2)
# This gives the Fourier Transform something rhythmic to find!
alpha_wave = np.cos(2 * np.pi * 10 * t) * 1.5
data[:, 2] += alpha_wave

# 4. Perform the "Math Surgery" (Spatial Filtering)
data_new_base = data @ eigenvectors
data_new_base[:, 2] = 0  # Zero out the Eye Blink (eigenvalue 195 is at index 2)
data_new_base = data_new_base @ eigenvectors.T

# Save data
pd.DataFrame(data).to_json("raw_data.json")
pd.DataFrame(data_new_base).to_json("new_data.json")

# Extract axes for plotting
Rp1, Rp2, Q1 = data[:, 0], data[:, 1], data[:, 2]
Fp1, Fp2, O1 = data_new_base[:, 0], data_new_base[:, 1], data_new_base[:, 2]

# 5. FOURIER TRANSFORM (FFT)
fft_result_o1 = np.fft.fft(O1)
frequencies_o1 = np.fft.fftfreq(len(O1), d=1 / fs)
fft_result_fp2 = np.fft.fft(Fp2)
frequencies_fp2 = np.fft.fftfreq(len(Fp2), d=1 / fs)

# filter of positive freq
pos_mask_o1 = frequencies_o1 > 0
freqs_o1_pos = frequencies_o1[pos_mask_o1]
pos_mask_fp2 = frequencies_fp2 > 0
freqs_fp2_pos = frequencies_fp2[pos_mask_fp2]
# Calculate Power Spectral Density (Magnitude squared)
psd_o1 = np.abs(fft_result_o1[pos_mask_o1]) ** 2
psd_fp2 = np.abs(fft_result_fp2[pos_mask_fp2]) ** 2

# visualizations
fig = plt.figure(figsize=(15, 6))

# Plot A: Original Noisy Data
bx = fig.add_subplot(141, projection='3d')
bx.scatter(Rp1, Rp2, Q1, alpha=0.3, color='purple', label='Raw EEG')
bx.set_title('Original Noisy EEG')
bx.set_xlabel('Fp1'); bx.set_ylabel('Fp2'); bx.set_zlabel('O1')

# Plot B: Cleaned Data
ax = fig.add_subplot(142, projection='3d')
ax.scatter(Fp1, Fp2, O1, alpha=0.3, color='blue', label='Cleaned EEG')
ax.set_title('Cleaned EEG (Blink Removed)')
ax.set_xlabel('Fp1'); ax.set_ylabel('Fp2'); ax.set_zlabel('O1')

# Add Eigenvectors to both 3D plots
colors = ['green', 'orange', 'red']
labels = ['PC3 (Brainwaves: λ=2)', 'PC2 (Local Noise: λ=5)', 'PC1 (Eye Blink: λ=195)']
for i in range(3):
    vec = eigenvectors[:, i]
    length = np.sqrt(eigenvalues[i]) * 2
    for axis in [ax, bx]:
        axis.quiver(0, 0, 0, vec[0]*length, vec[1]*length, vec[2]*length,
                    color=colors[i], linewidth=4, label=labels[i] if axis == ax else "")

if ax.get_legend_handles_labels()[1]: ax.legend()

# Plot C: Power Spectral Density (The Fourier Result)
cx = fig.add_subplot(143)
cx.plot(freqs_o1_pos, psd_o1, color='darkred')
cx.set_title('Frequency Spectrum (FFT of Clean O1)')
cx.set_xlabel('Frequency (Hz)')
cx.set_ylabel('Power')
cx.set_xlim(0, 30) # Zoom in on 0-30 Hz (where Alpha/Beta live)
cx.axvline(x=10, color='lightgray', linestyle='--', alpha=0.5, label='10 Hz Alpha Signal')
cx.legend()

dx = fig.add_subplot(144)
dx.plot(freqs_fp2_pos, psd_fp2, color='darkred')
dx.set_title('Frequency Spectrum (FFT of Cleaned Fp2)')
dx.set_xlabel('Frequency (Hz)')
dx.set_ylabel('Power')
dx.set_xlim(0, 30)
dx.axvline(x=10, color='lightgray', linestyle='--', alpha=0.5, label='10 Hz Alpha Signal')
dx.legend()


plt.tight_layout()
plt.show()