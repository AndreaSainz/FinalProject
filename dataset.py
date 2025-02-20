import os
from torch.utils.data import Dataset
import tomosipo as ts
import h5py
import torch
import numpy as np

class LoDoPaBDataset(Dataset):

    def __init__(self, ground_truth_dir):
        """
        Args:
            ground_truth_dir (str): Directory where the ground truth HDF5 files are stored.

        """
        self.ground_truth_dir = ground_truth_dir
        # List all files in the directory
        self.files = [os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir) if f.endswith('.h5')]

        self.pixel_size = 26/362 #Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.detector_bin_width = 513/26 #513 equidistant detector bins s spanning the image diameter.
        self.scan_distance = 79 # (cm) This is not specified, this is an assumption
        self.cone_angle = np.arctan(26/(2*self.scan_distance)) 


    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to be shown.
        
        Returns:
            A sample containing the ground truth data images and the sinogram.
        """

        file_number = idx//len(self.files)

        # Get the file corresponding to the index
        file_path = self.files[file_number]
        
        # Read the HDF5 file
        with h5py.File(file_path, 'r') as f:
            # For example, let's assume the ground truth data is stored in 'data' key in the HDF5 file
            ground_truth = f['data'][:]
        
        # Convert to tensor
        sample = torch.tensor(ground_truth, dtype=torch.float32)
        
        # Compute the local index within the file
        local_idx = idx % 128
        
        # Extract the specific slice
        sample_slice = sample[local_idx]

        # Create Sinogram using tomosipo
        # Volumen
        vg = ts.volume(shape=(1,362,362))

        #angels
        angles = np.linspace(0, np.pi, 1000, endpoint=True)

        # Fan beam structure
        pg = ts.cone(angles = angles, cone_angle = self.cone_angle )

        # Operator 
        A = ts.operator(vg,pg)

        # Tranform the image to sinogram (it is already a Pytorch tensor)
        sinogram = A(sample_slice)
        
        return {'ground_truth': sample, 'sinogram': sinogram}