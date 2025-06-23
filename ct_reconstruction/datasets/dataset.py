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
    PyTorch Dataset for Low-Dose CT image reconstruction using the LoDoPaB-CT dataset.

    This dataset class supports:
    - Loading CT slices and sinograms from local or GCS-based HDF5 files.
    - Simulation of realistic measurement noise using a combination of Poisson and Gaussian models.
    - Sparse-view sinogram generation and single-angle backprojections for learning-based CT reconstruction.

    Args:
        ground_truth_dir (str): Path to the directory or GCS bucket containing HDF5 files.
        vg (ts.VolumeGeometry): tomosipo volume geometry.
        angles (np.ndarray): Full set of projection angles.
        pg (ts.ProjectionGeometry): tomosipo projection geometry.
        A (ts.Operator): tomosipo forward operator.
        single_bp (bool): Whether to generate single-angle backprojections.
        n_single_BP (int): Number of angles to use for single backprojections.
        sparse_view (bool): Whether to use sparse-view sinograms.
        view_angles (int): Number of angles in sparse view.
        indices (list[int] or torch.Tensor, optional): Custom indices for sparse/single view.
        alpha (float): Normalization factor to scale data in [0, 1].
        i_0 (float): Incident photon count for Poisson noise.
        sigma (float): Standard deviation of Gaussian noise.
        seed (int): Random seed for noise reproducibility.
        max_len (int, optional): Max number of samples to load.
        debug (bool): If True, prints debug messages.
        logger (logging.Logger, optional): Custom logger.
        device (str): Device string, e.g., 'cpu' or 'cuda'.

    Attributes:
        files (List[str]): Sorted list of file paths.
        slices_per_file (int): Number of slices in each HDF5 file.
        indices (torch.Tensor): Selected angles for sparse or single-angle processing.

    Notes:
        The dataset expects each HDF5 file to contain the keys 'data' and 'sinograms'.
        The last HDF5 file is excluded automatically due to unknown slice count (as described in the LoDoPaB-CT paper).

    Example:
        >>> import tomosipo as ts
        >>> import numpy as np
        >>> vg = ts.volume(shape=(362, 362))
        >>> angles = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        >>> pg = ts.cone(angles=angles, src_orig_dist=575, shape=(1, 513))
        >>> A = ts.operator(vg, pg)
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
        >>> sample = dataset[0]
        >>> print(sample['ground_truth'].shape, sample['single_back_projections'].shape)
    """


    def __init__(self, ground_truth_dir, vg, angles, pg, A, single_bp = False, n_single_BP= 16, sparse_view = False, view_angles=90, indices = None, alpha=5, i_0 = 1000, sigma = 1, seed = 29072000, max_len = None,  debug = False, logger=None, device="cuda"):

        if single_bp and sparse_view:
            raise ValueError("Sparse-view sinogram and single view backprojections are not compatible now, choose one of them")

        #Scan parameters from the paper and data
        self.pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.num_angles = 1000
        self.num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
        self.src_orig_dist = 575
        self.src_det_dist = 1050
        self.slices_per_file = 128
        self.n_single_BP = int(n_single_BP)
        self.view_angles = int(view_angles)
        self.single_bp = single_bp
        self.sparse_view = sparse_view
        self.alpha = alpha
        self.device= device
        
        if self.single_bp or self.sparse_view:
            if indices is None:
                if self.single_bp:
                    indices = torch.linspace(0, self.num_angles - 1, steps=self.n_single_BP).long()
                elif self.sparse_view:
                    indices = torch.linspace(0, self.num_angles - 1, steps=self.view_angles).long()  
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(indices, dtype=torch.long)
        self.indices = indices

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


        if (self.single_bp or self.sparse_view) and self.indices is None:
            raise TypeError(f"Indices are needed")

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
    


    def max_normalize(self, tensor):
        """
        Scales a tensor to the range [0, 1] using max normalization.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            tuple: (normalized tensor, original max value)
        """
        # calculatin max from tensor
        max_val = tensor.max()

        #Normalising
        norm = tensor  / max_val 
        return norm, max_val
    


    def max_denormalize(self, norm_tensor, max_val):
        """
        Reverses max normalization to restore original scale.

        Args:
            norm_tensor (torch.Tensor): Normalized tensor.
            max_val (float): Max value used in normalization.

        Returns:
            torch.Tensor: Restored tensor.
        """
        return norm_tensor * max_val  
    

        
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
        norm_sino,  max_val =  self.max_normalize(sinogram)

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
        noisy_sinogram = self.max_denormalize(noisy_sinogram, max_val)

        # no negative 
        noisy_sinogram = torch.clamp(noisy_sinogram, min=0)

        return noisy_sinogram



    def _get_data_from_file(self, idx):
        """
        Takes an image slice and the corresponding sinogram given a global dataset index.

        Args:
            idx (int): Global index across all available slices.

        Returns:
            tuple: (ground truth image, sinogram), both as torch.Tensors.

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
        sino = torch.tensor(sino, dtype=torch.float32)
        
        self._log(f"[Dataset] Taking file number: {file_number}")
        self._log(f"[Dataset] Using file path: {file_path}")
        self._log(f"[Dataset] Taking image number: {local_idx}")

        return sample, sino



    def _generate_single_backprojections(self, sinogram):
        """
        Generates single-angle backprojections for each angle specified in `self.indices`.

        Each selected angle is used to construct a 1-view sinogram and perform a backprojection using tomosipo's fan-beam geometry.

        Args:
            sinogram (torch.Tensor): Noisy sinogram of shape (1, num_angles, num_detectors).

        Returns:
            torch.Tensor: A tensor of shape (n_single_BP, H, W) containing one backprojection per selected angle.

        Note:
            - Assumes `self.indices` and `self.angles` are properly initialized.
            - The output contains one image per selected projection angle.
        """

        projections = []
        
        for angle_idx in self.indices:
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
        Gives a sample from the dataset, including the ground truth image, its corresponding sinogram,
        and optional noisy, sparse, or single-backprojection outputs depending on the mode.

        Modes:
        - Default: Returns clean and noisy sinograms.
        - Sparse view (`sparse_view=True`): Additionally returns sparse-view sinogram.
        - Single backprojection (`single_bp=True`): Returns sparse-view sinogram and per-angle backprojections.

        Args:
            idx (int): Index of the sample. Must be < `max_len` if specified.

        Returns:
            dict: Dictionary with the following keys depending on the mode:
                - 'ground_truth' (torch.Tensor): Ground truth image, shape (1, H, W).
                - 'sinogram' (torch.Tensor): Clean sinogram, shape (1, A, D).
                - 'noisy_sinogram' (torch.Tensor): Noisy sinogram, shape (1, A, D). (not returned if `single_bp=True`)
                - 'sparse_sinogram' (torch.Tensor): Subsampled noisy sinogram, shape (1, V, D), where V is the number of selected views.
                - 'single_back_projections' (torch.Tensor): Backprojections from individual angles, shape (n_single_BP, H, W). Returned only if `single_bp=True`.

        Raises:
            IndexError: If `idx` is out of range based on the configured `max_len`.
        """

        #checking the index when I do not want all the files
        if self.max_len is not None and idx >= self.max_len:
            raise IndexError(f"Index {idx} out of range for max_len={self.max_len}")

        # Get image
        sample_slice, sinogram = self._get_data_from_file(idx)
    
        # Add nose to sinogram
        noisy_sinogram = self.noise(sinogram, idx)
                                                                                
        #Create single-back projections
        if self.single_bp:
            single_back_projections = self._generate_single_backprojections(noisy_sinogram)
            sinogram_sparse = noisy_sinogram[:, self.indices, :] 
            return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'sparse_sinogram': sinogram_sparse,
            'single_back_projections': single_back_projections}
        
        elif self.sparse_view:
            sparse_sinogram = noisy_sinogram[:, self.indices, :]
            return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'noisy_sinogram': noisy_sinogram, 
            'sparse_sinogram': sparse_sinogram}

        return {'ground_truth': sample_slice, 
            'sinogram': sinogram, 
            'noisy_sinogram': noisy_sinogram}