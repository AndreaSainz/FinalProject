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
        # Convert to absolute path
        self.ground_truth_dir = os.path.abspath(ground_truth_dir)
        print(f"Using dataset directory: {self.ground_truth_dir}")  

        # List all HDF5 files
        self.files = [os.path.join(self.ground_truth_dir, f) for f in os.listdir(self.ground_truth_dir) if f.endswith('.hdf5')]
        # Sort the files such that '000' files come first
        self.files.sort()
        print("Found files:", self.files)

        # If no files are found, raise an error
        if len(self.files) == 0:
            raise ValueError(f"No HDF5 files found in directory: {self.ground_truth_dir}")
    
        #Scan parameters from the paper
        self.pixel_size = 26/362 #Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.detector_bin_width = 513/26 #513 equidistant detector bins s spanning the image diameter.
        self.scan_distance = 79 # (cm) This is not specified, this is an assumption
        self.cone_angle = 0.00000001 #np.arctan(26/(2*self.scan_distance)) 
        self.num_detectors = 513

        print("Cone Angle (rads):", self.cone_angle)



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

        #Calculating the file number. The idx will be a number between 0 and 128*number of files (because each file contains 128 images)
        file_number = int(idx/len(self.files))
        print(f"Taking file number: {file_number}")

        # Get the file corresponding to the index
        file_path = self.files[file_number]
        print(f"Using file path: {file_path}")
        
        # Read the HDF5 file
        with h5py.File(file_path, 'r') as f:
            # For example, let's assume the ground truth data is stored in 'data' key in the HDF5 file
            ground_truth = f['data'][:]
        
        # Convert to tensor
        sample = torch.tensor(ground_truth, dtype=torch.float32)
        
        # Compute the local index within the file
        local_idx = idx % 128
        print(f"Taking image number: {local_idx}")
        
        # Extract the specific slice
        sample_slice = sample[local_idx].unsqueeze(0) # .unsqueeze(0) ensures the pytorch tensor is in the form (1,362,362) instead of (362,362)

        # Create Sinogram using tomosipo
        # Volumen
        vg = ts.volume(shape=(1,362,362))

        #angels
        angles = np.linspace(0, np.pi, 1000, endpoint=True)

        # Fan beam structure
        pg = ts.cone(angles = angles, cone_angle = self.cone_angle, shape=(1, self.num_detectors))

        # Operator 
        A = ts.operator(vg,pg)

        # Tranform the image to sinogram (it is already a Pytorch tensor)
        sinogram = A(sample_slice)

        # Print the shape of the sinogram
        print("Sinogram shape:", sinogram.shape)

        #Backprojecting sinogram to check that I can get the same image
        back_projection = A.T(sinogram)
        
        return {'ground_truth': sample, 'sinogram': sinogram, 'backprojection': back_projection}