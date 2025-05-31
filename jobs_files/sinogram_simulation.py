"""
This script simulates full-dose and low-dose sinograms from high-resolution projections of CT images,
following the data generation methodology described in the paper:

LoDoPaB-CT: A benchmark dataset for low-dose computed tomography reconstruction
Scientific Data | (2021) 8:109 | https://doi.org/10.1038/s41597-021-00893-z

Overview:
---------
The simulation process mirrors the methodology outlined in the original LoDoPaB-CT dataset paper,
ensuring consistency with its assumptions and avoiding common problems like the inverse crime.

Steps performed:
----------------

1. **Image Upsampling**:
   - Ground truth CT images are originally sized at 362×362 pixels.
   - To avoid the *inverse crime* (i.e., using the same discretization for simulation and reconstruction),
     the images are upsampled to 1000×1000 using bilinear interpolation.
   - This ensures that the reconstruction task is based on a different resolution than the simulation.

2. **Forward Projection (Ray Transform)**:
   - The upsampled images are projected using tomosipo’s fan-beam operator (`ts.cone`) to simulate the
     continuous X-ray transform (analogous to the Radon transform in 2D).
   - This generates clean, full-dose sinograms.

3. **Full-dose Output**:
   - The original ground truth image (362×362) and the corresponding clean sinogram are saved
     in the `ground_truth_*` folders.

4. **Low-dose Simulation (Poisson noise model)**:
   - Simulates reduced photon counts using a Poisson process:
       photons ~ Poisson(N₀ * exp(-Ax))
   - N₀ is set to 4096, matching the setup in the original paper for low-dose acquisition.
   - Photon counts of zero are clipped to a minimum of 0.1 to avoid issues in the log-domain.
   - The logarithmic measurement is computed using the Beer–Lambert law:
       y = -log(photons / N₀)
   - Finally, the log-transformed sinogram is normalized by dividing by μ_max = 81.35858,
     which ensures consistency with the normalization of ground truth images into the [0, 1] range.

5. **Low-dose Output**:
   - The same ground truth image (362×362) and the simulated low-dose sinogram are saved
     in the `observation_*` folders.

Output structure:
-----------------
Each HDF5 file written contains two datasets:
  - `data`: the ground truth CT image (shape: [362, 362])
  - `sinograms`: either the full-dose or low-dose sinogram (shape: [num_angles, num_detectors])

Folder structure (after running the script for all dataset splits):
  project_root/
  ├── data_sino/
  │   ├── ground_truth_train/
  │   ├── ground_truth_validation/
  │   ├── ground_truth_test/
  │   ├── observation_train/
  │   ├── observation_validation/
  │   └── observation_test/

Notes:
------
- The sinograms are generated using tomosipo’s fan-beam geometry via `ts.cone`, which corresponds
  to a 2D fan-beam CT system.
- This pipeline uses the CPU backend (`astra_cpu`) in the referenced paper due to numerical inaccuracies
  reported with `astra_cuda` at specific angles and detector positions. However, GPU backends can be used
  here for performance if validated.
- μ_max and the minimum photon count threshold (0.1) are critical for stable log-transforms and are applied
  as described in the original publication.
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
pixel_size = 26.0
pixels_upsampling = 1000
num_angles = 1000
num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
src_orig_dist = 575
src_det_dist = 1050
N0 = 4096
u_max = 81.35858


# Create tomosipo volume and projection geometry
vg = ts.volume(shape=(1, pixels, pixels),size=(1.0, pixel_size,pixel_size))                                               # Volumen
angles = np.linspace(0, np.pi, num_angles, endpoint=True)                                # Angles
pg = ts.cone(
    angles=angles,
    src_orig_dist=src_orig_dist,
    src_det_dist=src_det_dist,
    shape=(1, num_detectors),
    size=(1.0, pixel_size) 
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

        images_tensor = torch.tensor(images)
        if images_tensor.ndim == 2:
          images_tensor = images_tensor.unsqueeze(0)  # [1, H, W]

        # Forward projection to get sinograms (one by one)
        sinograms = torch.stack([A(img) for img in images_tensor])

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