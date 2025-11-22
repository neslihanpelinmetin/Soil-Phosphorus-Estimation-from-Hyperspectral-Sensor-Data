import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# === Seed ===
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

# === Directory ===
plot_dir = "spectral_plots"
os.makedirs(plot_dir, exist_ok=True)

# === Data ===
data_df = pd.read_excel("data.xlsx", header=None)  # (bands, samples)
X = data_df.values.T  # (samples, bands)

# === Number of Samples ===
# for plot 1 and 3
num_samples = 5
X_subset = X[:num_samples]

# for plot 2
num_samples_2 = 1
X_subset_2 = X[:num_samples_2]

# === Color Map ===
cmap = plt.get_cmap("viridis", num_samples)
colors = cmap(np.linspace(0, 1, num_samples))

# === 1. Graph before Normalization ===
plt.figure(figsize=(16, 12))
for i in range(num_samples):
    plt.plot(range(X.shape[1]), X_subset[i], label=f"Sample #{i}", color=colors[i], linewidth=4)
plt.title("Spectra (Before Standardization)", fontsize=30)
plt.xlabel("Wavelength (nm)", fontsize=28)
plt.ylabel("Reflectance", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/spectra_raw.png", dpi=300)

# === Read Selected Band Indices ===
selected_band_indices = np.loadtxt("selected_band_indices.txt", dtype=int)

# === 2. Graph before Normalization and Selected Bands ===
print(selected_band_indices[:10])  # Show first 10 bands
print(selected_band_indices[10:])  # Show last 10 bands
print("Min index:", selected_band_indices.min())
print("Max index:", selected_band_indices.max())
print("Selected Band count:", selected_band_indices.shape[0])
print("Band count:", X.shape[1])


plt.figure(figsize=(16, 12))
for i in range(num_samples_2):
    plt.plot(range(X.shape[1]), X_subset_2[i], label=f"Sample #{i}", color=colors[i], linewidth=4)
    # Add points on the selected bands
    plt.scatter(selected_band_indices, X_subset_2[i, selected_band_indices],
                color=colors[4], s=100, edgecolor='black', marker='o', zorder=5)

plt.title("The 300 most informative spectral bands\nselected via MIR", fontsize=30)
plt.xlabel("Wavelength (nm)", fontsize=28)
plt.ylabel("Reflectance", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/spectra_raw_with_selected_bands_sample0.png", dpi=300)

# === Normalization Function ===
def row_normalize(X):
    min_vals = X.min(axis=1, keepdims=True)
    max_vals = X.max(axis=1, keepdims=True)
    return (X - min_vals) / (max_vals - min_vals + 1e-8)

X_norm = row_normalize(X_subset)

# === 3. Graph after Normalization ===
plt.figure(figsize=(16, 12))
for i in range(num_samples):
    plt.plot(range(X.shape[1]), X_norm[i], label=f"Sample #{i}", color=colors[i], linewidth=4)
plt.title("Spectra (After Standardization)", fontsize=30)
plt.xlabel("Wavelength (nm)", fontsize=28)
plt.ylabel("Standardized Reflectance", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/spectra_normalized.png", dpi=300)


# === 4. Graphs for Different Phosphorus Values ===
labels_df = pd.read_excel("labels.xlsx", header=None)  # (samples, 1)

X = data_df.values.T  # (samples, bands)
y = labels_df.values.squeeze()  # (samples,)

selected_indices = [13, 0, 97, 54, 25, 38, 17, 151, 136, 114]
X_subset = X[selected_indices]
y_subset = y[selected_indices]

# === Color Map ===
num_samples = len(selected_indices)
cmap = plt.get_cmap("viridis", num_samples)
colors = cmap(np.linspace(0, 1, num_samples))

# === Spectral Graph ===
plt.figure(figsize=(16, 12))
for i, idx in enumerate(selected_indices):
    plt.plot(
        range(X.shape[1]),
        X[idx],
        label=f"Sample #{idx} (P={y[idx]:.2f})",
        color=colors[i],
        linewidth=4
    )

plt.xlabel("Wavelength (nm)", fontsize=28)
plt.ylabel("Reflectance", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15, loc="best", frameon=True)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/reflectance_vs_phosphorus.png", dpi=300)
plt.close()
