import os
from torch.utils.data import Dataset
import tomosipo as ts
import h5py
import torch
import numpy as np
import gcsfs
import math
import logging

class LoDoPaBDataset(Dataset):
    """
    PyTorch Dataset for low-dose CT image reconstruction using LoDoPaB-CT data.

    This dataset class:
    - Loads ground truth slices from HDF5 files (local or GCS).
    - Computes forward projections (sinograms) using a given operator.
    - Adds realistic noise (Poisson and Gaussian) to simulate measurements.
    - Generates sparse-view backprojections using a fixed number of single-angle projections.

    Designed for supervised training of deep learning models in sparse-view CT reconstruction.

    Parameters:
        ground_truth_dir (str): Path to directory or GCS bucket containing .hdf5 CT slices.
        vg (ts.VolumeGeometry): tomosipo volume geometry describing the image domain.
        angles (np.ndarray): Array of angles used for projection.
        pg (ts.ProjectionGeometry): tomosipo projection geometry describing acquisition.
        A (ts.Operator): tomosipo operator performing the forward projection.
        single_bp (bool): If True, generates a stack of single-angle backprojections.
        n_single_BP (int): Number of angles to use for backprojection.
        alpha (float): Scaling factor for normalizing ground truth images.
        i_0 (float): Incident photon count (used in Poisson noise simulation).
        sigma (float): Standard deviation of Gaussian noise added to the sinogram.
        seed (int): Random seed for reproducibility.
        max_len (int, optional): Maximum number of samples to load. If None, all slices are used.
        debug (bool): If True, prints log messages to the console.
        logger (logging.Logger, optional): Custom logger instance. If None, a default is created.

    Attributes:
        files (List[str]): List of HDF5 file paths used for training.
        angles_SBP (np.ndarray): Subset of angles used for sparse backprojection.
        slices_per_file (int): Number of slices stored per HDF5 file (default: 128).
        alpha (float): Scaling factor applied to ground truth images.
        logger (logging.Logger): Logger instance for internal use.

    Example:
        >>> import tomosipo as ts
        >>> import numpy as np
        >>> from ct_reconstruction.datasets import LoDoPaBDataset

        >>> # Define tomosipo geometry
        >>> vg = ts.volume(shape=(362, 362), size=(26, 26))
        >>> angles = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        >>> pg = ts.cone(angles=angles, src_orig_dist=575, src_det_dist=1050, shape=(1000, 513))
        >>> A = ts.operator(vg, pg)

        >>> # Initialize dataset (local directory example)
        >>> dataset = LoDoPaBDataset(
        ...     ground_truth_dir="data/lodopab/train",
        ...     vg=vg,
        ...     angles=angles,
        ...     pg=pg,
        ...     A=A,
        ...     single_bp=True,
        ...     n_single_BP=16,
        ...     alpha=5.0,
        ...     i_0=1e5,
        ...     sigma=0.01,
        ...     seed=42,
        ...     max_len=128,
        ...     debug=True
        ... )

        >>> # Access a sample
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['ground_truth', 'sinogram', 'noisy_sinogram', 'noisy_sinogram_normalise', 'single_back_projections'])
    """


    def __init__(self, ground_truth_dir, vg, angles, pg, A, single_bp = False, n_single_BP= 16, sparse_view = False, indices = None, alpha=5, i_0 = 1000, sigma = 1, seed = 29072000, max_len = None,  debug = False, logger=None, device="cuda"):

        if single_bp and sparse_view:
            ValueError("Sparse-view sinogram and single view backprojections are not compatible now, choose one of them")

        #Scan parameters from the paper and data
        self.pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.num_angles = 1000
        self.num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
        self.src_orig_dist = 575
        self.src_det_dist = 1050
        self.slices_per_file = 128
        self.n_single_BP = int(n_single_BP)
        self.single_bp = single_bp
        self.sparse_view = sparse_view
        self.indices = indices
        self.alpha = alpha
        self.device= device
        

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
            self.files = [f"gs://{f}" for f in self.fs.ls(ground_truth_dir) if f.endswith('.hdf5') and not f.startswith('._')]
        else:
            self.files = [os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir) if f.endswith('.hdf5') and not f.startswith('._')]

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
        """
        Normalizes a tensor to the range [0, 1] using min-max scaling.

        Parameters:
            tensor (torch.Tensor): Input tensor to normalize.

        Returns:
            tuple:
                - norm (torch.Tensor): Normalized tensor.
                - min_val (float): Minimum value of the original tensor.
                - max_val (float): Maximum value of the original tensor.
        """
        # calculatin min and max from tensor
        min_val = tensor.min()
        max_val = tensor.max()

        #Normalising
        norm = (tensor - min_val) / (max_val - min_val)
        return norm, min_val, max_val
    


    def minmax_denormalize(self, norm_tensor, min_val, max_val):
        """
        Reverses min-max normalization using provided min and max values.

        Parameters:
            norm_tensor (torch.Tensor): Normalized tensor.
            min_val (float): Original minimum value.
            max_val (float): Original maximum value.

        Returns:
            torch.Tensor: Denormalized tensor in original scale.
        """
        return norm_tensor * (max_val - min_val) + min_val
    

        
    def noise(self, sinogram, idx):
        """
        Adds realistic noise to a sinogram using Poisson and Gaussian processes.

        Parameters:
            sinogram (torch.Tensor): Clean input sinogram.

        Returns:
            torch.Tensor: Noisy sinogram of the same shape.

        Notes:
            - Poisson noise simulates X-ray photon count fluctuations.
            - Gaussian noise represents detector/system imperfections.
        """
        # Nomalised sinogram between [0,1] to add noise
        norm_sino, min_val, max_val =  self.minmax_normalize(sinogram)

        #Initilizing seed generator for reproducibility porpuses (this is for having more than one coworkers)
        generator = torch.Generator(device=sinogram.device).manual_seed(self.seed + idx)

        # Simulate measured photons using Poisson noise (counting error)
        measured_photons = torch.poisson(self.i_0 * torch.exp(-norm_sino), generator=generator)

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
            torch.Tensor: Image tensor of shape (1, H, W).

        Raises:
            ValueError: If no HDF5 files are found during dataset initialization.
        """

        #Calculating the file number. The idx will be a number between 0 and 128*number of files (because each file contains 128 images)
        file_number = int(idx/self.slices_per_file)

        # Get the file corresponding to the index
        file_path = self.files[file_number]

        # Compute the local index within the file
        local_idx = idx - self.slices_per_file*file_number

        # Read the HDF5 file
        if file_path.startswith("gs://"):
            with h5py.File(self.fs.open(file_path, 'rb')) as f:
                ground_truth = f['data'][local_idx] /self.alpha
                sino = f['sinograms'][local_idx]/self.alpha
        else:
            with h5py.File(file_path, 'r') as f:
                ground_truth = f['data'][local_idx]/self.alpha
                sino = f['sinograms'][local_idx]/self.alpha

                

        # Convert to tensor
        sample = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(0)
        sino = torch.tensor(sino, dtype=torch.float32).unsqueeze(0)


        self._log(f"[Dataset] Taking file number: {file_number}")
        self._log(f"[Dataset] Using file path: {file_path}")
        self._log(f"[Dataset] Taking image number: {local_idx}")

        return sample, sino



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
        Retrieves a single training sample including projections and ground truth.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing:
                - 'ground_truth' (torch.Tensor): Image (1, H, W).
                - 'sinogram' (torch.Tensor): Clean sinogram (1, A, D).
                - 'noisy_sinogram' (torch.Tensor): Noisy sinogram (1, A, D).
                - 'noisy_sinogram_normalise' (torch.Tensor): Normalized noisy sinogram.
                - 'single_back_projections' (torch.Tensor): [Optional] (n_single_BP, H, W) if `single_bp=True`.

        Raises:
            IndexError: If the index exceeds `max_len`.
        """

        #checking the index when I do not want all the files
        if self.max_len is not None and idx >= self.max_len:
            raise IndexError(f"Index {idx} out of range for max_len={self.max_len}")

        # Get image
        sample_slice, sinogram = self._get_image_from_file(idx)
    
        # Add nose to sinogram
        noisy_sinogram = self.noise(sinogram, idx)
                                                                                
        #Create single-back projections
        if self.single_bp:
            single_back_projections = self._generate_single_backprojections(sinogram)
            sinogram_sparse = noisy_sinogram[:, self.angles_SBP, :] 
            return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'sinogram_sparse': sinogram_sparse, 
            'single_back_projections': single_back_projections}
        
        elif self.sparse_view:
            sparse_sinogram = noisy_sinogram[:, self.indices, :]
            return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'noisy_sinogram': noisy_sinogram, 
            'sparse_sinogram': sparse_sinogram}

        return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'noisy_sinogram': noisy_sinogram,}