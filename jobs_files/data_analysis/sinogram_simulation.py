"""
This script generates full-dose sinograms from CT images using forward projection
and saves them in HDF5 format for later use in reconstruction tasks.

Important Notes:
----------------
Although the original LoDoPaB-CT paper suggests simulating projections from higher-resolution
images (e.g., 1000×1000) to avoid the *inverse crime*, this implementation skips that step
due to empirical issues observed during reconstruction (poor sinogram quality or instabilities).

Instead, sinograms are generated directly from 362×362 CT images using tomosipo's fan-beam geometry.
The low-dose effect will be simulated later during training by applying strong Poisson noise.

Summary of Process:
-------------------

1. **Input Loading**:
   - Ground truth CT slices of size 362×362 are loaded from preprocessed HDF5 files.

2. **Forward Projection**:
   - Each image is projected using tomosipo’s `ts.cone` operator (fan-beam geometry)
     to simulate clean (noise-free) sinograms.
   - No upsampling or resolution change is performed — images are projected at their native resolution.

3. **Saving Output**:
   - Each output HDF5 file contains:
       - `"data"`: the original CT slice.
       - `"sinograms"`: the corresponding full-dose, clean sinogram (`Ax`).

4. **Low-dose Simulation (deferred)**:
   - No Poisson or Gaussian noise is added in this script.
   - Low-dose conditions will be simulated **on-the-fly during training** by applying Poisson noise
     with low photon counts (e.g., `i_0 = 1e3`) directly to these sinograms.
   - No normalization by `μ_max` is performed, as it is no longer relevant in this noise model.

Directory Structure:
--------------------
This script generates files into the following structure:

  project_root/
  └── data_sino/
      ├── ground_truth_train/
      ├── ground_truth_validation/
      └── ground_truth_test/

Each output HDF5 file contains two datasets:
  - `data`: Ground truth CT image (shape: [362, 362])
  - `sinograms`: Clean sinogram (shape: [num_angles, num_detectors])

Technical Notes:
----------------
- Uses tomosipo’s `ts.cone` with 1000 projection angles and 513 detectors.
- Fan-beam parameters are based on the LoDoPaB-CT setup (src_orig_dist=575, src_det_dist=1050).
- GPU backends can be used for acceleration if tomosipo is properly configured.
"""

import os
import h5py
import numpy as np
import tomosipo as ts
from tqdm import tqdm
import torch
from glob import glob

#Scan parameters from the paper and data
pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
pixel_size = 26.0
pixels_upsampling = 1000
num_angles = 1000
num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
src_orig_dist = 575
src_det_dist = 1050
N0 = 4096
u_max = 81.35858


# Create tomosipo volume and projection geometry
vg = ts.volume(shape=(1, pixels, pixels))                                               # Volumen
angles = np.linspace(0, np.pi, num_angles, endpoint=True)                                # Angles
pg = ts.cone(
    angles=angles,
    src_orig_dist=src_orig_dist,
    shape=(1, num_detectors)
)     # Fan beam structure
A = ts.operator(vg,pg)



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

# Loop through folders and process each file
for folder, out_folder_gt in zip(input_folders, output_folders_gt):
     # make sure the output directories exists
    os.makedirs(out_folder_gt, exist_ok=True)  

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

        images_tensor = torch.tensor(images)
        if images_tensor.ndim == 2:
          images_tensor = images_tensor.unsqueeze(0)  # [1, H, W]

        # Forward projection to get sinograms (one by one)
        sinograms = torch.stack([A(img.unsqueeze(0) if img.ndim == 2 else img) for img in images_tensor])

        # Save ground truth with sinograms
        output_path_gt = os.path.join(out_folder_gt, f"ground_truth_{mode}_{file_num}.hdf5")
        with h5py.File(output_path_gt, 'w') as out_f:
            out_f.create_dataset("data", data=images, compression="gzip")
            out_f.create_dataset("sinograms", data=sinograms, compression="gzip")