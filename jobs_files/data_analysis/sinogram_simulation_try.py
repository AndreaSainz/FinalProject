"""
This script simulates full-dose and low-dose sinograms from high-resolution projections of CT images. The simulation process mirrors the methodology outlined in the original LoDoPaB-CT dataset paper,
ensuring consistency with its assumptions and avoiding common problems like the inverse crime.

LoDoPaB-CT: A benchmark dataset for low-dose computed tomography reconstruction
Scientific Data | (2021) 8:109 | https://doi.org/10.1038/s41597-021-00893-z

Notes:
- The sinograms are generated using tomosipo’s fan-beam geometry via ts.cone, which corresponds
  to a 2D fan-beam CT system.
- $\mu_{\text{max}}$ and the minimum photon count threshold (0.1) are critical for stable log-transforms and are applied as described in the original publication.
"""

import os
import h5py
import numpy as np
import tomosipo as ts
from tqdm import tqdm
import torch
from torch.nn.functional import interpolate
from glob import glob

#Scan parameters from the paper and data
pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
pixels_upsampling = 1000
num_angles = 1000
num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
src_orig_dist = 575
src_det_dist = 1050
N0 = 4096
u_max = 81.35858

# Create tomosipo volume and projection geometry
vg = ts.volume(shape=(1,pixels_upsampling,pixels_upsampling))                                                  # Volumen
angles = np.linspace(0, np.pi, num_angles, endpoint=True)                                # Angles
pg = ts.cone(angles = angles, src_orig_dist=src_orig_dist, shape=(1, num_detectors))     # Fan beam structure
A = ts.operator(vg,pg)                                                                  # Operator

#  folder's paths
input_folders = [
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_train",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_validation",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data/ground_truth_test"
]

output_folders_gt = [
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_train",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_validation",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/ground_truth_test"
]

output_folders_ld = [
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_train",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_validation",
    "/home/as3628/rds/hpc-work/final_project_dis/as3628/data_sino/observation_test"
]

# Loop through folders and process each file
for folder, out_folder_gt, out_folder_ld  in zip(input_folders, output_folders_gt, output_folders_ld):
     # make sure the output directories exists
    os.makedirs(out_folder_gt, exist_ok=True)  
    os.makedirs(out_folder_ld, exist_ok=True)

    files = glob(os.path.join(folder, '*.hdf5'))

    for file in tqdm(files, desc=f"Processing {folder}"):
        try:
            with h5py.File(file, 'r') as f:
                images = f["data"][:]
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        # Extract mode (train/validation/test) and file number from filename
        base_name = os.path.basename(file)
        parts = base_name.split('_')
        if len(parts) < 3:
            print(f"Unexpected filename format: {base_name}")
            continue
        mode = parts[2]  # e.g., 'test' from 'ground_truth_test_000.hdf5'
        file_num = parts[-1].split('.')[0]  # e.g., '000'

        # Upsample from 362x362 to 1000x1000 to avoid inverse crime
        images_tensor = torch.tensor(images)
        if images_tensor.ndim == 2:
            images_tensor = images_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(1)  # [N, 1, H, W]

        images_upscaled = interpolate(images_tensor, size=(1000, 1000), mode='bilinear', align_corners=False)

        # Forward projection to get sinograms (one by one)
        sinograms = torch.stack([A(img) for img in images_upscaled])

        # Save ground truth with sinograms
        output_path_gt = os.path.join(out_folder_gt, f"ground_truth_{mode}_{file_num}.hdf5")
        with h5py.File(output_path_gt, 'w') as out_f:
            out_f.create_dataset("data", data=images, compression="gzip")
            out_f.create_dataset("sinograms", data=sinograms, compression="gzip")

        # now we need to simulate the low-dose sinograms
        sinograms = sinograms.float()
        simulated_photons = torch.poisson( N0 * torch.exp(-sinograms))

        # maximum between pixel and 0.1, taken pixel-wise
        simulated_photons = torch.clamp(simulated_photons, min=0.1)

        # Convert back to log domain (using Beer–Lambert Law) and divide by u_max (for images in range [0-1])
        low_dose_sinogram = -torch.log(simulated_photons / N0)/ u_max

         # Save images and corresponding simulated low-dose sinograms
        output_path_ld = os.path.join(out_folder_ld, f"low_dose_{mode}_{file_num}.hdf5")
        with h5py.File(output_path_ld, 'w') as out_f:
            out_f.create_dataset("data", data=images, compression="gzip")
            out_f.create_dataset("sinograms", data=low_dose_sinogram, compression="gzip")