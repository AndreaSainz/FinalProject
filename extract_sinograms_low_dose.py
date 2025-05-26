import os
import h5py
import numpy as np
from tqdm import tqdm

# Directorios de entrada y salida
image_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test"       # Ej: contiene ground_truth_test_000.hdf5, etc.
sinogram_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/observation_test"     # Ej: contiene observation_test_000.hdf5, etc.
output_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_test"
os.makedirs(output_dir, exist_ok=True)

# Listar archivos relevantes
img_files = sorted([f for f in os.listdir(image_dir) if f.startswith("ground_truth_test_") and f.endswith(".hdf5")])
sino_files = sorted([f for f in os.listdir(sinogram_dir) if f.startswith("observation_test_") and f.endswith(".hdf5")])

# Función para extraer el número final
def get_suffix(fname):
    return fname.split("_")[-1].replace(".hdf5", "")

# Crear diccionarios por sufijo
img_map = {get_suffix(f): f for f in img_files}
sino_map = {get_suffix(f): f for f in sino_files}

# Emparejar por sufijo
common_suffixes = sorted(set(img_map) & set(sino_map))

for suffix in tqdm(common_suffixes, desc="Combinando pares"):
    img_path = os.path.join(image_dir, img_map[suffix])
    sino_path = os.path.join(sinogram_dir, sino_map[suffix])
    output_path = os.path.join(output_dir, f"paired_{suffix}.hdf5")

    # Saltar si ya existe
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} ya existe.")
        continue

    # Leer datos
    with h5py.File(img_path, "r") as f_img:
        imgs = f_img["data"][:]

    with h5py.File(sino_path, "r") as f_sino:
        sinograms = f_sino["data"][:]  # asume que el dataset se llama "data", cambia si es "sinograms"

    # Validación
    if imgs.shape[0] != sinograms.shape[0]:
        print(f"[ERROR] Tamaños diferentes en {suffix}: {imgs.shape[0]} imágenes vs {sinograms.shape[0]} sinogramas")
        continue

    # Guardar archivo combinado
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("data", data=imgs, compression="gzip")
        f_out.create_dataset("sinograms", data=sinograms, compression="gzip")

    print(f"[OK] Guardado: {output_path}")