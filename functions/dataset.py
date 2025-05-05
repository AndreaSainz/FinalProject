import os
from torch.utils.data import Dataset
import tomosipo as ts
import h5py
import torch
import numpy as np

class LoDoPaBDataset(Dataset):

    def __init__(self, ground_truth_dir, n_single_BP= 16, i_0 = 1000, sigma = 5, seed = 29072000, debug = 0):
        """
        Args:
            ground_truth_dir (str): Directory where the ground truth HDF5 files are stored.
            n_single_BP (int): Number of single-back projections to be created.
        """

        #Scan parameters from the paper
        self.pixel_size = 26/362 #Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.detector_bin_width = 513/26 #513 equidistant detector bins s spanning the image diameter.
        self.num_detectors = 513
        self.src_orig_dist = 575
        self.src_det_dist = 1050
        self.n_single_BP = n_single_BP

        # Noise parameter control 
        self.i_0 = i_0 # incident photons
        self.sigma = sigma
        self.seed = seed

        # Debug parameter
        self.debug = debug

        # Fam beam structure
        # Volumen
        self.vg = ts.volume(shape=(1,362,362))

        #angels
        self.angles = np.linspace(0, np.pi, 1000, endpoint=True)

        # Fan beam structure
        self.pg = ts.cone(angles = self.angles, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))

        # Operator
        self.A = ts.operator(self.vg,self.pg)

        # Angles for the views
        # Define angles(equally spaciated indexes) for single-view back projection
        self.angles_SBP = np.linspace(0, len(self.angles) - 1, self.n_single_BP, dtype=int)



        # Convert to absolute path
        self.ground_truth_dir = os.path.abspath(ground_truth_dir)
        if self.debug == 1:
            print(f"Using dataset directory: {self.ground_truth_dir}")

        # List all HDF5 files
        self.files = [os.path.join(self.ground_truth_dir, f) for f in os.listdir(self.ground_truth_dir) if f.endswith('.hdf5')]
        # Sort the files such that '000' files come first
        self.files.sort()
        if self.debug == 1:
            print("Found files:", self.files)

        # If no files are found, raise an error
        if len(self.files) == 0:
            raise ValueError(f"No HDF5 files found in directory: {self.ground_truth_dir}")

        


    def noise(self, sinogram):
        """Add Poisson and Gaussian noise to the sinogram"""
        
        #Initilizing seed for reproducibility porpuses
        torch.manual_seed(self.seed)

        # Simulate measured photons using Poisson noise (counting error)
        measured_photons = torch.poisson(self.i_0 * torch.exp(-sinogram))

        # Avoid log(0) by setting the minimum to 1
        measured_photons = torch.clamp(measured_photons, min=1.0)

        # Convert back to log domain (using Beer–Lambert Law)
        noisy_sinogram = -torch.log(measured_photons / self.i_0)

        # Adding gaussian noise (detector's imperfections)
        gaussian_noise = torch.normal(mean=0, std=self.sigma, size=sinogram.shape, dtype=sinogram.dtype, device=sinogram.device) 
        #other functions do not allow for device

        # Final result 
        noisy_sinogram += gaussian_noise

        return noisy_sinogram


    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.files)


    def _get_image_from_file(self, idx):
        #Calculating the file number. The idx will be a number between 0 and 128*number of files (because each file contains 128 images)
        file_number = int(idx/128)
        if self.debug == 1:
            print(f"Taking file number: {file_number}")

        # Get the file corresponding to the index
        file_path = self.files[file_number]
        if self.debug == 1:
            print(f"Using file path: {file_path}")

        # Read the HDF5 file
        with h5py.File(file_path, 'r') as f:
            # For example, let's assume the ground truth data is stored in 'data' key in the HDF5 file
            ground_truth = f['data'][:]

        # Convert to tensor
        sample = torch.tensor(ground_truth, dtype=torch.float32)

        # Compute the local index within the file
        local_idx = idx - 128*file_number
        if self.debug == 1:
            print(f"Taking image number: {local_idx}")

        # Extract the specific slice
        sample_slice = sample[local_idx].unsqueeze(0) # .unsqueeze(0) ensures the pytorch tensor is in the form (1,362,362) instead of (362,362)
        sample_slice = sample_slice/1000  #rescale the image
        return sample_slice


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to be shown.

        Returns:
            A sample containing the ground truth data images and the sinogram.
        """

        # Get image
        sample_slice = self._get_image_from_file(idx)

        # Tranform the image to sinogram (it is already a Pytorch tensor)
        sinogram = self.A(sample_slice)

        # Add nose to sinogram
        noisy_sinogram = self.noise(sinogram)

        # Print the shape of the sinogram
        if self.debug == 1:
            print("Sinogram shape:", sinogram.shape)

        #Backprojecting sinogram to check that I can get the same image
        #back_projection = self.A.T(sinogram)

        # Print the shape of the back_projection
        #if self.debug == 1:
        #    print("back_projection shape:", back_projection.shape)


        #Create single-back projections
        single_back_projection = []

       
        for angle_SBP in self.angles_SBP:

          # Define Fan Beam Geometry for each angle
          pg_SP = ts.cone(angles = self.angles[angle_SBP], src_orig_dist=self.src_orig_dist , shape=(1, self.num_detectors))
          # Compute Back Projection
          A_SP = ts.operator(self.vg,pg_SP)

          # Extract only the sinogram at this specific angle
          sinogram_SP = sinogram[:, angle_SBP:angle_SBP+1, :]

          # Back projection at single angle
          back_projection_SP = A_SP.T(sinogram_SP)

          # Convert to PyTorch tensor and append
          single_back_projection.append(back_projection_SP)

        # Stack all projections into a single tensor of shape [n_single_BP, 362, 362]
        single_back_projection = torch.stack(single_back_projection).squeeze(1) #remove extra dimension ([16, 1, 362, 362])

        return {'ground_truth': sample_slice, 
        'sinogram': sinogram, 
        'noisy_sinogram':noisy_sinogram, 
        #'backprojection': back_projection, 
        'single_back_projections': single_back_projection}