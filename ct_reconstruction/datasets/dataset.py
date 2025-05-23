import os
from torch.utils.data import Dataset
from ..utils.loggers import configure_logger
import tomosipo as ts
import h5py
import torch
import numpy as np
import gcsfs
import math
import logging

class LoDoPaBDataset(Dataset):
    """
    PyTorch Dataset for low-dose CT reconstruction using LoDoPaB-CT data.

    This dataset:
    - Loads ground truth slices from HDF5 files (local or GCS).
    - Computes forward projections (sinograms).
    - Simulates realistic noise (Poisson + Gaussian).
    - Produces sparse-view backprojections at predefined angles.

    Designed for training deep learning models in CT image reconstruction.

    Args:
        ground_truth_dir (str): Path to directory containing HDF5 files with CT slices (can be local or 'gs://').
        vg (ts.VolumeGeometry): tomosipo volume geometry.
        angles (np.ndarray): Array of angles used for projection.
        pg (ts.ProjectionGeometry): tomosipo projection geometry.
        A (ts.Operator): tomosipo forward operator.
        n_single_BP (int): Number of single-angle backprojections to simulate.
        alpha (float): Normalization factor for ground truth scaling.
        i_0 (float): Incident photon count (for Poisson noise).
        sigma (float): Standard deviation of Gaussian noise.
        seed (int): Random seed for reproducibility.
        max_len (int, optional): Maximum number of samples to load. If None, uses all slices.
        debug (bool): If True, prints debug logs to console.
        logger (logging.Logger, optional): Logger instance. If None, a default is created.
    """


    def __init__(self, ground_truth_dir, vg, angles, pg, A, single_bp = False, n_single_BP= 16, alpha=5, i_0 = 1000, sigma = 1, seed = 29072000, max_len = None,  debug = False, logger=None):
        
        #Scan parameters from the paper and data
        self.pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.num_angles = 1000
        self.num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
        self.src_orig_dist = 575
        self.src_det_dist = 1050
        self.slices_per_file = 128
        self.n_single_BP = int(n_single_BP)
        self.single_bp = single_bp
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
        """
        Returns the number of samples available in the dataset (or max_len if set).

        Returns:
            int: Number of total accessible samples.
        """
        total = len(self.files) * self.slices_per_file  
        return min(total, self.max_len) if self.max_len is not None else total




    def __repr__(self):
        """
        Returns a string summary of the dataset configuration.

        Returns:
            str: Dataset description.
        """

        return (f"LoDoPaBDataset(num_samples={len(self)}, "
                f"n_single_BP={self.n_single_BP}, "
                f"noise=Poisson+Gaussian(σ={self.sigma}), "
                f"i_0={self.i_0})")

    def minmax_normalize(self, tensor):

        min_val = tensor.min()
        max_val = tensor.max()
        norm = (tensor - min_val) / (max_val - min_val)
        return norm, min_val, max_val

    def minmax_denormalize(self, norm_tensor, min_val, max_val):
        return norm_tensor * (max_val - min_val) + min_val
        
    def noise(self, sinogram):
        """
        Adds Poisson and Gaussian noise to a sinogram.

        Simulates realistic CT measurement noise using:
            - Poisson noise (from X-ray photon counts).
            - Gaussian noise (detector and system noise).

        Args:
            sinogram (torch.Tensor): Clean input sinogram.

        Returns:
            torch.Tensor: Noisy sinogram of the same shape.
        """
        # Nomalised sinogram between [0,1] to add noise
        norm_sino, min_val, max_val =  self.minmax_normalize(sinogram)

        #Initilizing seed for reproducibility porpuses
        torch.manual_seed(self.seed)

        # Simulate measured photons using Poisson noise (counting error)
        measured_photons = torch.poisson(self.i_0 * torch.exp(-norm_sino))

        # Avoid log(0) by setting the minimum to 1
        measured_photons = torch.clamp(measured_photons, min=1.0)

        # Convert back to log domain (using Beer–Lambert Law)
        noisy_sinogram = -torch.log(measured_photons / self.i_0)

        # Adding gaussian noise (detector's imperfections)
        gaussian_noise = torch.normal(mean=0, std=self.sigma, size=sinogram.shape, dtype=sinogram.dtype, device=sinogram.device) 
        #other functions do not allow for device

        # Final result 
        noisy_sinogram += gaussian_noise

        # reaply the normalization
        noisy_sinogram = self.minmax_denormalize(noisy_sinogram, min_val, max_val)

        # no negative 
        noisy_sinogram = torch.clamp(noisy_sinogram, min=0)

        return noisy_sinogram



    def _get_image_from_file(self, idx):
        """
        Recovers a normalized image slice given a global dataset index.

        Loads the appropriate HDF5 file and extracts the corresponding slice.

        Args:
            idx (int): Global index across all available slices.

        Returns:
            torch.Tensor: Image tensor of shape (1, H, W), normalized to [0, alpha].
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
        Generates a set of backprojections from selected angles for sparse-view simulation.

        Each angle is used to generate a single-angle backprojection using tomosipo.

        Args:
            sinogram (torch.Tensor): Noisy sinogram of shape (1, num_angles, num_detectors).

        Returns:
            torch.Tensor: Stack of backprojections of shape (n_single_BP, H, W).
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
        Returns one training sample consisting of:
            - Ground truth image.
            - Clean sinogram.
            - Noisy sinogram.
            - Set of single-angle backprojections.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: {
                'ground_truth' (torch.Tensor): Image tensor (1, H, W),
                'sinogram' (torch.Tensor): Clean sinogram (1, A, D),
                'noisy_sinogram' (torch.Tensor): Noisy sinogram (1, A, D),
                'single_back_projections' (torch.Tensor): Tensor of shape (n_single_BP, H, W)
            }

        Raises:
            IndexError: If idx is out of range for max_len.
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


        # Normalise noisy sinogram for training porpuses
        noisy_sinogram_normalise, _, _ =  self.minmax_normalize(noisy_sinogram)

                                                                                
        #Create single-back projections
        if self.single_bp:
            single_back_projections = self._generate_single_backprojections(sinogram)
            single_back_projections, _, _ =  self.minmax_normalize(single_back_projections)

            return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'noisy_sinogram':noisy_sinogram, 
            'noisy_sinogram_normalise':noisy_sinogram_normalise,
            'single_back_projections': single_back_projections}
        
        return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'noisy_sinogram': noisy_sinogram,
            'noisy_sinogram_normalise': noisy_sinogram_normalise}