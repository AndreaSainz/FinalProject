import os
import h5py
import numpy as np

# Ruta del archivo de imagen y sinograma
image_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train/ground_truth_train_279.hdf5"
sinogram_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/observation_train/observation_train_279.hdf5"
output_file = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_train/paired_279.hdf5"

# Crear carpeta de salida si no existe
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Leer imagen
with h5py.File(image_file, "r") as f_img:
    imgs = f_img["data"][:]

# Leer sinograma
with h5py.File(sinogram_file, "r") as f_sino:
    sinograms = f_sino["data"][:]  # o usa "sinograms" si es el nombre real

# Guardar archivo combinado
with h5py.File(output_file, "w") as f_out:
    f_out.create_dataset("data", data=imgs, compression="gzip")
    f_out.create_dataset("sinograms", data=sinograms, compression="gzip")

print(f"[OK] Archivo combinado guardado en: {output_file}")