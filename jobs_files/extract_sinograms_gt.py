import os
import h5py
import torch
import numpy as np
import tomosipo as ts
from tqdm import tqdm

#Scan parameters from the paper and data
pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
num_angles = 1000
num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
src_orig_dist = 575
src_det_dist = 1050

# Create tomosipo volume and projection geometry
vg = ts.volume(shape=(1,pixels,pixels))                                                  # Volumen
angles = np.linspace(0, np.pi, num_angles, endpoint=True)                                # Angles
pg = ts.cone(angles = angles, src_orig_dist=src_orig_dist, shape=(1, num_detectors))     # Fan beam structure
A = ts.operator(vg,pg)                                                                  # Operator

# paths
input_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train"
output_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train"
os.makedirs(output_dir, exist_ok=True)

# extracting files .hdf5
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.hdf5') and not f.startswith('._')])


for fname in tqdm(files, desc="file processing"):
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname) 

    # Skip if already exists in output folder with synograms
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f_out:
            if "sinograms" in f_out:
                print(f"[SKIP] {fname} has sinograms in output_dir.")
                continue

    with h5py.File(input_path, "r") as f_in:
        imgs = f_in["data"][:]  # (128, 362, 362)

    sinograms = np.zeros((imgs.shape[0], num_angles, num_detectors), dtype=np.float32)

    for i in range(imgs.shape[0]):
        img_tensor = torch.tensor(imgs[i], dtype=torch.float32).unsqueeze(0).to("cuda")
        sino = A(img_tensor).squeeze(0).cpu().numpy()
        sinograms[i] = sino

    # Save new file with both datasets
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("data", data=imgs, compression="gzip")
        f_out.create_dataset("sinograms", data=sinograms, compression="gzip")


# paths
input_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test"
output_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test"
os.makedirs(output_dir, exist_ok=True)

# extracting files .hdf5
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.hdf5') and not f.startswith('._')])


for fname in tqdm(files, desc="file processing"):
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname) 

    # Skip if already exists in output folder with synograms
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f_out:
            if "sinograms" in f_out:
                print(f"[SKIP] {fname} has sinograms in output_dir.")
                continue

    with h5py.File(input_path, "r") as f_in:
        imgs = f_in["data"][:]  # (128, 362, 362)

    sinograms = np.zeros((imgs.shape[0], num_angles, num_detectors), dtype=np.float32)

    for i in range(imgs.shape[0]):
        img_tensor = torch.tensor(imgs[i], dtype=torch.float32).unsqueeze(0).to("cuda")
        sino = A(img_tensor).squeeze(0).cpu().numpy()
        sinograms[i] = sino

    # Save new file with both datasets
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("data", data=imgs, compression="gzip")
        f_out.create_dataset("sinograms", data=sinograms, compression="gzip")

# paths
input_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_validation"
output_dir = "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation"
os.makedirs(output_dir, exist_ok=True)

# extracting files .hdf5
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.hdf5') and not f.startswith('._')])


for fname in tqdm(files, desc="file processing"):
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname) 

    # Skip if already exists in output folder with synograms
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f_out:
            if "sinograms" in f_out:
                print(f"[SKIP] {fname} has sinograms in output_dir.")
                continue

    with h5py.File(input_path, "r") as f_in:
        imgs = f_in["data"][:]  # (128, 362, 362)

    sinograms = np.zeros((imgs.shape[0], num_angles, num_detectors), dtype=np.float32)

    for i in range(imgs.shape[0]):
        img_tensor = torch.tensor(imgs[i], dtype=torch.float32).unsqueeze(0).to("cuda")
        sino = A(img_tensor).squeeze(0).cpu().numpy()
        sinograms[i] = sino

    # Save new file with both datasets
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("data", data=imgs, compression="gzip")
        f_out.create_dataset("sinograms", data=sinograms, compression="gzip")