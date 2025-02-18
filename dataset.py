import os
from torch.utils.data import Dataset
import h5py
import torch
from skimage.transform import radon
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
        # Get the file corresponding to the index
        file_path = self.files[idx]
        
        # Read the HDF5 file
        with h5py.File(file_path, 'r') as f:
            # For example, let's assume the ground truth data is stored in 'data' key in the HDF5 file
            ground_truth = f['data'][:]
        
        # Convert to tensor
        sample = torch.tensor(ground_truth, dtype=torch.float32)
        
        # Create Sinogram 
        theta = np.linspace(0.0, 360.0, max(sample.shape), endpoint=False)
        sinogram = radon(sample, theta=theta)
        
        # Convert the sinogram to PyTorch tensor
        sinogram_tensor = torch.tensor(sinogram, dtype=torch.float32)

    
        return {'ground_truth': sample, 'sinogram': sinogram_tensor}