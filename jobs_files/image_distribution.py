"""
Analyze Ground Truth Image Intensity Distribution for LoDoPaB-CT Dataset

This script scans all `.hdf5` files across the train, validation, and test splits of the LoDoPaB-CT dataset,
extracts all CT image slices, flattens them, and analyzes the global intensity distribution.

Its primary goal is to validate that the images have been correctly normalized using a scaling factor `μ_max`,
as described in the original LoDoPaB-CT paper:

    "LoDoPaB-CT: A Benchmark Dataset for Low-Dose Computed Tomography Reconstruction"
    - Leuschner et al., 2021

According to the paper, images were originally normalized by `μ_max = 81.35858` so that pixel values should 
lie approximately in the range [0, 1]. This normalization is essential for:
    - consistent training behavior across neural networks
    - physically meaningful loss values
    - stable visualization and interpretability

-----------------
Expected outcome:
-----------------
If normalization was applied correctly, the intensity histogram should show that most image pixels lie within
[0, 1]. The 95th percentile should fall slightly below 1.

-----------------
Observed result:
-----------------
In this analysis, we found:
    Percentile 95: 0.297807

This confirms that most pixel values are within the expected range, but skewed toward the lower end of the 
interval [0, 1].

-----------------
Implication:
-----------------
To ensure stable training and avoid vanishing gradients due to very small intensity ranges (e.g., many values 
concentrated in [0, 0.3]), we will use the 95th percentile value as a global scaling factor (`alpha = 0.297807`)
in our dataset loader.

This preserves the **relative physical meaning** of intensities while ensuring:
    - image and sinogram values are rescaled to roughly [0, 1]
    - losses remain numerically significant
    - plotting remains standardized with `vmin=0`, `vmax=1`

-----------------
Output:
-----------------
- Logs the 95th percentile value.
- Saves a histogram plot (`image_distribucion.png`) showing the global intensity distribution.

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Rutas a las tres carpetas que contienen archivos .hdf5
carpetas = [
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_validation",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test"
]

# Almacena todos los valores de todas las imágenes
valores = []

# Recorre las carpetas y procesa los archivos hdf5
for carpeta in carpetas:
    archivos = glob(os.path.join(carpeta, '*.hdf5'))

    for archivo in archivos:
        try:
            with h5py.File(archivo, 'r') as f:
                imagenes = f["data"][:]
                valores.append(imagenes.flatten())
        except Exception as e:
            print(f"Error reading {archivo}: {e}")

# Concatenar todos los valores en un único array
todos_los_valores = np.concatenate(valores)

# Calcular percentiles para análisis
percentil_95 = np.percentile(todos_los_valores, 95)

print(f'\n Percentile 95: {percentil_95:.6f}')

# (Opcional) Visualizar histograma de intensidades
plt.hist(todos_los_valores, bins=500, log=True)
plt.axvline(percentil_95, color='r', linestyle='--', label='percentile 95')
plt.title('Image values distribution')
plt.xlabel('Intensities')
plt.ylabel('Frecuency (log)')
plt.legend()
plt.savefig(f"image_distribucion.png")
plt.close()