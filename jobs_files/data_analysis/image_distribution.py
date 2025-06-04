"""
Analyze Ground Truth Image Intensity Distribution for LoDoPaB-CT Dataset

This script analyzes the intensity distribution of all ground truth CT images in the LoDoPaB-CT dataset.
It processes `.hdf5` files across the training, validation, and test splits to:

    - Extract all CT image slices.
    - Flatten and concatenate all pixel intensities.
    - Compute the global intensity histogram.
    - Calculate the 95th percentile of all pixel values.
    - Analyze the distribution of per-image maximum values.

-----------------
Motivation:
-----------------
As described in the LoDoPaB-CT paper:

    "LoDoPaB-CT: A Benchmark Dataset for Low-Dose Computed Tomography Reconstruction"
    - Leuschner et al., 2021

CT images are normalized by a global constant `μ_max = 81.35858`, resulting in most pixel intensities 
falling in the range [0, 1]. This normalization is essential for:

    - consistent training of deep learning models
    - numerically meaningful loss values
    - standardized and interpretable image visualization

-----------------
Observed Results:
-----------------
Global pixel intensity statistics:
    - Percentile 95, all values: **0.297807**

Per-image maximum intensity statistics:
    - Global maximum: 1.0000
    - Global minimum: 0.0000
    - Mean: 0.6462
    - Median: 0.5937
    - 90th percentile: 0.9828
    - Number of images with max > 0.9: 5588
    - Number of images with max < 0.3: 113

-----------------
Conclusion:
-----------------
The results confirm that:
    - The majority of image pixel intensities lie within [0, 1], as expected.
    - Most individual images have maximum values close to 1.
    - Only a very small fraction of images have very low maximums (< 0.3).

Therefore, **no further rescaling is needed**. We will use a global scaling factor `alpha = 1`, preserving:
    - the relative and physical integrity of the data
    - training stability and loss interpretability
    - consistent visualization with `vmin=0`, `vmax=1`

-----------------
Outputs:
-----------------
- Prints summary statistics and percentile values.
- Saves:
    - `images_distribucion.png`: Histogram of all pixel values with 95th percentile marked.
    - `maxima_images.png`: Histogram of per-image maximum values with median and 90th percentile markers.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# path to the folders with the ground truth images
folders = [
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_validation",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test"
]

# store all values from images and the maxima
values = []
maxima = []

# read and process all files
for folder in folders:
    files = glob(os.path.join(folder, '*.hdf5'))

    for file in files:
        try:
            with h5py.File(file, 'r') as f:
                images = f["data"][:]
                values.append(images.flatten())
                for img in images:
                    max_val = np.max(img)
                    maxima.append(max_val)
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Concatenar todos los valores en un único array
maxima = np.array(maxima)
all_values = np.concatenate(values)

# calculate percentiles for all values
percentil_95_all = np.percentile(all_values, 95)

print(f'\n Percentile 95, all values: {percentil_95_all:.6f}')

# (Opcional) Visualizar histograma de intensidades
plt.hist(all_values, bins=500, log=True)
plt.axvline(percentil_95_all, color='r', linestyle='--', label='percentile 95')
plt.title('Image values distribution')
plt.xlabel('Intensities')
plt.ylabel('Frecuency (log)')
plt.legend()
plt.savefig(f"images_distribucion.png")
plt.close()


# Analyze per-image maxima
print("\n-- Statistics of per-image maxima --")
print(f"Global maximum: {np.max(maxima):.4f}")
print(f"Global minimum: {np.min(maxima):.4f}")
print(f"Mean: {np.mean(maxima):.4f}")
print(f"Median: {np.median(maxima):.4f}")
print(f"90th percentile: {np.percentile(maxima, 90):.4f}")
print(f"Images with max > 0.9: {(maxima > 0.9).sum()}")
print(f"Images with max < 0.3: {(maxima < 0.3).sum()}")

# Plot histogram of per-image maxima
plt.hist(maxima, bins=100)
plt.axvline(np.median(maxima), color='g', linestyle='--', label='Median')
plt.axvline(np.percentile(maxima, 90), color='r', linestyle='--', label='90th percentile')
plt.title("Distribution of per-image maximum values")
plt.xlabel("Maximum value (per image)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("maxima_images.png")
plt.close()