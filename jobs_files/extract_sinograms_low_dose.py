import os
import h5py
import numpy as np
from tqdm import tqdm


# input and output directories
image_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train"       
sinogram_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/observation_train"     
output_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_train"
os.makedirs(output_dir, exist_ok=True)

# list of files
img_files = sorted([f for f in os.listdir(image_dir) if f.startswith("ground_truth_") and f.endswith(".hdf5")])
sino_files = sorted([f for f in os.listdir(sinogram_dir) if f.startswith("observation_") and f.endswith(".hdf5")])

# Function to extract the final number
def get_suffix(fname):
    return fname.split("_")[-1].replace(".hdf5", "")

# creater dir by the numbers 
img_map = {get_suffix(f): f for f in img_files}
sino_map = {get_suffix(f): f for f in sino_files}

# match by the number file 
common_suffixes = sorted(set(img_map) & set(sino_map))

for suffix in tqdm(common_suffixes, desc="Combining pares"):
    img_path = os.path.join(image_dir, img_map[suffix])
    sino_path = os.path.join(sinogram_dir, sino_map[suffix])
    output_path = os.path.join(output_dir, f"paired_{suffix}.hdf5")

    # continue if it already exits 
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exits.")
        continue

    #read data
    with h5py.File(img_path, "r") as f_img:
        imgs = f_img["data"][:]

    with h5py.File(sino_path, "r") as f_sino:
        sinograms = f_sino["data"][:]  

    #validation
    if imgs.shape[0] != sinograms.shape[0]:
        print(f"[ERROR] Different shapes in {suffix}: {imgs.shape[0]} imágenes vs {sinograms.shape[0]} sinograms")
        continue

    # save combine file
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("data", data=imgs, compression="gzip")
        f_out.create_dataset("sinograms", data=sinograms, compression="gzip")

    print(f"[OK] Save: {output_path}")


# input and output directories
image_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test"       
sinogram_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/observation_test"     
output_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_test"
os.makedirs(output_dir, exist_ok=True)

# list of files
img_files = sorted([f for f in os.listdir(image_dir) if f.startswith("ground_truth_") and f.endswith(".hdf5")])
sino_files = sorted([f for f in os.listdir(sinogram_dir) if f.startswith("observation_") and f.endswith(".hdf5")])

# Function to extract the final number
def get_suffix(fname):
    return fname.split("_")[-1].replace(".hdf5", "")

# creater dir by the numbers 
img_map = {get_suffix(f): f for f in img_files}
sino_map = {get_suffix(f): f for f in sino_files}

# match by the number file 
common_suffixes = sorted(set(img_map) & set(sino_map))

for suffix in tqdm(common_suffixes, desc="Combining pares"):
    img_path = os.path.join(image_dir, img_map[suffix])
    sino_path = os.path.join(sinogram_dir, sino_map[suffix])
    output_path = os.path.join(output_dir, f"paired_{suffix}.hdf5")

    # continue if it already exits 
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exits.")
        continue

    #read data
    with h5py.File(img_path, "r") as f_img:
        imgs = f_img["data"][:]

    with h5py.File(sino_path, "r") as f_sino:
        sinograms = f_sino["data"][:]  

    #validation
    if imgs.shape[0] != sinograms.shape[0]:
        print(f"[ERROR] Different shapes in {suffix}: {imgs.shape[0]} imágenes vs {sinograms.shape[0]} sinograms")
        continue

    # save combine file
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("data", data=imgs, compression="gzip")
        f_out.create_dataset("sinograms", data=sinograms, compression="gzip")

    print(f"[OK] Save: {output_path}")


# input and output directories
image_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_validation"       
sinogram_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/observation_validation"     
output_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_validation"
os.makedirs(output_dir, exist_ok=True)

# list of files
img_files = sorted([f for f in os.listdir(image_dir) if f.startswith("ground_truth_") and f.endswith(".hdf5")])
sino_files = sorted([f for f in os.listdir(sinogram_dir) if f.startswith("observation_") and f.endswith(".hdf5")])

# Function to extract the final number
def get_suffix(fname):
    return fname.split("_")[-1].replace(".hdf5", "")

# creater dir by the numbers 
img_map = {get_suffix(f): f for f in img_files}
sino_map = {get_suffix(f): f for f in sino_files}

# match by the number file 
common_suffixes = sorted(set(img_map) & set(sino_map))

for suffix in tqdm(common_suffixes, desc="Combining pares"):
    img_path = os.path.join(image_dir, img_map[suffix])
    sino_path = os.path.join(sinogram_dir, sino_map[suffix])
    output_path = os.path.join(output_dir, f"paired_{suffix}.hdf5")

    # continue if it already exits 
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exits.")
        continue

    #read data
    with h5py.File(img_path, "r") as f_img:
        imgs = f_img["data"][:]

    with h5py.File(sino_path, "r") as f_sino:
        sinograms = f_sino["data"][:]  

    #validation
    if imgs.shape[0] != sinograms.shape[0]:
        print(f"[ERROR] Different shapes in {suffix}: {imgs.shape[0]} imágenes vs {sinograms.shape[0]} sinograms")
        continue

    # save combine file
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("data", data=imgs, compression="gzip")
        f_out.create_dataset("sinograms", data=sinograms, compression="gzip")

    print(f"[OK] Save: {output_path}")