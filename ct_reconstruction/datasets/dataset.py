import os
from torch.utils.data import Dataset
from ..utils.loggers import configure_logger
import tomosipo as ts
import h5py
import torch
import numpy as np
import gcsfs
import math

class LoDoPaBDataset(Dataset):

    """
    PyTorch Dataset for low-dose CT reconstruction using sinograms from the data base LoDoPaB-CT.
    Simulates noisy measurements and provides input-target pairs for deep learning models.

    Args:
        ground_truth_dir (str): Path to directory with HDF5 files containing ground truth images.
        n_single_BP (int): Number of single-angle backprojections to generate.
        i_0 (float): Incident X-ray photon count (controls Poisson noise).
        sigma (float): Standard deviation of additive Gaussian noise.
        seed (int): Random seed for reproducibility.
        max_len : 
        debug (bool): If True, prints debug information during execution.
    """


    def __init__(self, ground_truth_dir, vg, angles, pg, A, n_single_BP= 16, alpha=5, i_0 = 1000, sigma = 1, seed = 29072000, max_len = None,  debug = False, logger=None):
        
        #Scan parameters from the paper and data
        self.pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.num_angles = 1000
        self.num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
        self.src_orig_dist = 575
        self.src_det_dist = 1050
        self.slices_per_file = 128
        self.n_single_BP = int(n_single_BP)
        self.alpha = alpha
        

        # Noise parameter  
        self.i_0 = i_0                  # Incident photons
        self.sigma = sigma
        self.seed = seed


        # Debug parameter
        self.debug = debug
        self.logger = (logger.getChild("dataset") if logger else logging.getLogger(__name__ + ".dataset")) 
        self.logger.propagate = False  # prevents logs from being sent to the parent logger
        self.max_len = max_len  

        

        # tomosipo volume and projection geometry
        self.vg = vg                                                     
        self.angles = angles 
        self.pg = pg 
        self.A = A 


        # Select subset of angles for sparse-view backprojection
        self.angles_SBP = np.linspace(0, len(self.angles) - 1, self.n_single_BP, dtype=int)


        # Prepare list of data files
        self.ground_truth_dir = ground_truth_dir
        self.fs = gcsfs.GCSFileSystem()
        
        if ground_truth_dir.startswith("gs://"):
            self.files = [f"gs://{f}" for f in self.fs.ls(ground_truth_dir) if f.endswith('.hdf5')]
        else:
            self.files = [os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir) if f.endswith('.hdf5')]

        # Sort files so I can be sure the last entrance is the last patient
        self.files.sort()

        # Remove last patient because we do not know how many samples it has (known from paper)
        if len(self.files) > 0:
            self.files = self.files[:-1]

        # If no files are found, raise an error
        if len(self.files) == 0:
            raise ValueError(f"No HDF5 files found in directory: {self.ground_truth_dir}")


        # determine how many files I need to cover max_len 
        if self.max_len is not None:
            required_files = math.ceil(self.max_len / self.slices_per_file)
            self.files = self.files[:required_files]

        # save logs
        self._log(f"[Dataset] Using directory: {self.ground_truth_dir}")
        self._log(f"[Dataset] Found {len(self.files)} files.")


    def _log(self, msg):
        """
        Logs a message to the logger's file and optionally prints it to the console.

        This method ensures that:
        - All messages are always logged to the file through the logger.
        - Messages are printed to the console only if debug mode is enabled.

        Args:
            msg (str): The message to be logged.
        """
        # Always log to file
        self.logger.info(msg)

        # Only print to console if debug mode is active
        if self.debug:
            print(msg)



    def __len__(self):
        """Return the number of samples in the dataset that are going to be used"""
        total = len(self.files) * self.slices_per_file  
        return min(total, self.max_len) if self.max_len is not None else total




    def __repr__(self):
        return (f"LoDoPaBDataset(num_samples={len(self)}, "
                f"n_single_BP={self.n_single_BP}, "
                f"noise=Poisson+Gaussian(σ={self.sigma}), "
                f"i_0={self.i_0})")


    def noise(self, sinogram):
        """
        Adds Poisson and Gaussian noise to a clean sinogram.

        Args:
            sinogram (torch.Tensor): Clean sinogram.

        Returns:
            torch.Tensor: Noisy sinogram.
        """
        
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



    def _get_image_from_file(self, idx):
        """
        Loads a specific image slice given a global index.

        Args:
            idx (int): Global index across all slices.

        Returns:
            torch.Tensor: Normalized image of shape (1, H, W)
        """

        #Calculating the file number. The idx will be a number between 0 and 128*number of files (because each file contains 128 images)
        file_number = int(idx/self.slices_per_file)

        # Get the file corresponding to the index
        file_path = self.files[file_number]

        # Read the HDF5 file
        if file_path.startswith("gs://"):
            with h5py.File(self.fs.open(file_path, 'rb')) as f:
                ground_truth = f['data'][:]
        else:
            with h5py.File(file_path, 'r') as f:
                ground_truth = f['data'][:]

                

        # Convert to tensor
        sample = torch.tensor(ground_truth, dtype=torch.float32)

        # Compute the local index within the file
        local_idx = idx - self.slices_per_file*file_number

        # Extract the specific slice
        sample_slice = sample[local_idx].unsqueeze(0)      #.unsqueeze(0) ensures the pytorch tensor is in the form (1,362,362) instead of (362,362)
    
        self._log(f"[Dataset] Taking file number: {file_number}")
        self._log(f"[Dataset] Using file path: {file_path}")
        self._log(f"[Dataset] Taking image number: {local_idx}")

        return sample_slice



    def _generate_single_backprojections(self, sinogram):
        """
        Generates sparse-view backprojections at n_single_BP selected angles.

        Args:
            sinogram (torch.Tensor): Clean sinogram.

        Returns:
            torch.Tensor: Tensor of shape (n_single_BP, H, W) with individual backprojections.
        """

        projections = []
        
        for angle_idx in self.angles_SBP:
            # Define Fan Beam Geometry for each angle
            proj_geom_single = ts.cone(angles= self.angles[angle_idx], src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))

            # Compute Back Projection
            A_single = ts.operator(self.vg, proj_geom_single)

            # Extract only the sinogram at this specific angle
            sinogram_angle = sinogram[:, angle_idx:angle_idx+1, :]

            # Back projection at single angle
            projection = A_single.T(sinogram_angle)

            projections.append(projection)

        # Stack all projections into a single tensor of shape [n_single_BP, 362, 362]
        single_back_projection = torch.stack(projections).squeeze(1) #remove extra dimension ([16, 1, 362, 362])

        return single_back_projection


    def __getitem__(self, idx):
        """
        Returns the training sample at the specified index.

        Args:
            idx (int): Index of sample.

        Returns:
            dict: {
                'ground_truth': Ground truth image (1, H, W),
                'sinogram': Clean sinogram,
                'noisy_sinogram': Sinogram with noise,
                'single_back_projections': [n_single_BP, H, W]
            }
        """

        #checking the index when I do not want all the files
        if self.max_len is not None and idx >= self.max_len:
            raise IndexError(f"Index {idx} out of range for max_len={self.max_len}")

        # Get image
        sample_slice = self._get_image_from_file(idx)

        #normalise the image into a physical interval
        sample_slice = sample_slice / sample_slice.max()  
        sample_slice = sample_slice * self.alpha

        # Tranform the image to sinogram (it is already a Pytorch tensor)
        sinogram = self.A(sample_slice)
    

        #self._log(f"[Sinogram] Sinogram - min: {sinogram.min().item():.4f}, max: {sinogram.max().item():.4f}, mean: {sinogram.mean().item():.4f}, std: {sinogram.std().item():.4f}")

        # Add nose to sinogram
        noisy_sinogram = self.noise(sinogram)

        # Print the shape of the sinogram
        #self._log("[Sinogram] Sinogram shape:", sinogram.shape)

                                                                                
        #Create single-back projections
        single_back_projections = self._generate_single_backprojections(noisy_sinogram)

        return {'ground_truth': sample_slice, 
        'sinogram': sinogram, 
        'noisy_sinogram':noisy_sinogram, 
        'single_back_projections': single_back_projections}