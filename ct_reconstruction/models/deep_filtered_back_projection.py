import torch
from torch.nn import Module
from torch.nn import Conv1d, Conv2d, BatchNorm1d, Sequential, PReLU, ReLU, Parameter
from ..models.model import ModelBase
import json
import tomosipo as ts
import matplotlib.pyplot as plt
from tomosipo.torch_support import (to_autograd)

class DeepFBPNetwork(Module):
    """
    Neural network implementation of the Deep Filtered Backprojection (DeepFBP) model for CT reconstruction.

    The architecture consists of:
    - A learnable frequency-domain filter module.
    - Depthwise 1D convolutional interpolation blocks for each angle.
    - A differentiable backprojection layer via Tomosipo.
    - A 2D CNN-based denoising network for image refinement.

    Args:
        num_detectors (int): Number of detector bins.
        num_angles (int): Number of projection angles.
        A (ts.Operator): Tomosipo forward projection operator.
        filter_type (str): Filtering mode. "Filter I" for shared filter, any other for per-angle.

    Attributes:
        learnable_filter (LearnableFilter): Trainable frequency-domain filter module.
        interpolator_{1-3} (Sequential): Depthwise residual blocks for angular interpolation.
        interpolator_conv (Conv1d): Final angular smoothing convolution.
        denoising_conv_1 (Conv2d): Initial conv layer in the denoising CNN.
        denoising_res_{1-3} (Sequential): Intermediate residual conv blocks.
        denoising_conv_2 (Conv2d): Final output conv layer of the denoiser.
    """
    def __init__(self, num_detectors, num_angles, A, filter_type, pixel_size):
        #Scan parameters from the paper and data
        self.pixel_size = pixel_size


        # initilize parameter from parent clase Module
        super().__init__()

        #Initilize all attributes for this class
        self.num_detectors = num_detectors
        self.num_angles_modulo = num_angles
        self.A_modulo = A
        self.filter_type = filter_type
        self.AT = to_autograd(self.A_modulo.T, is_2d=True, num_extra_dims=2)

        # initilize filter as ram-lak filter
        ram_lak = self.ram_lak_filter()

        #create the filter from the class
        if self.filter_type == "Filter I":
            self.learnable_filter = LearnableFilter(ram_lak.clone().float(), per_angle=False)
        else:
            self.learnable_filter = LearnableFilter(ram_lak.clone().float(), per_angle=True, num_angles=self.num_angles_modulo)

        #initilize interpolator
        self.interpolator_1 = intermediate_residual_block(1)
        self.interpolator_2 = intermediate_residual_block(1)
        self.interpolator_3 = intermediate_residual_block(1)
        self.interpolator_conv = Conv1d(1, 1, kernel_size=3, stride=1, padding=1)

        #initilize denoising part
        self.denoising_conv_1 = Conv2d(1, 64, kernel_size=1, stride=1, padding=0)
        self.denoising_res_1 = denoising_residual_block(64)
        self.denoising_res_2 = denoising_residual_block(64)
        self.denoising_res_3 = denoising_residual_block(64)
        self.denoising_conv_2 = Conv2d(64, 1, kernel_size=1, stride=1, padding=0)


    def ram_lak_filter(self):
        """
        Create the Ram-Lak filter directly in the frequency domain as |ω| over the FFT frequencies.
        """
        freqs = torch.fft.fftfreq(self.num_detectors)  # sin especificar d => frecuencias en [-0.5, 0.5)

        # Filtro Ram-Lak: forma de V con máximo 1 en |f| = 0.5
        ram_lak = 2 * torch.abs(freqs)  # Normalizado: de 0 a 1

        self.freqs_normalized = freqs
        return ram_lak


    def forward(self, x):
        """
        Executes the forward pass through the DeepFBPNetwork.

        The pipeline includes:
            - Learnable filtering in the frequency domain
            - Angular interpolation using depthwise convolutions
            - Differentiable backprojection via tomosipo operator
            - Denoising using a 2D CNN

        Args:
            x (torch.Tensor): Input sinogram of shape (B, 1, A, D), where:
                B = batch size,
                A = number of projection angles,
                D = number of detectors.

        Returns:
            torch.Tensor: Reconstructed CT image of shape (B, 1, H, W).
        """
        
        # Apply filter for frequency domain
        x1 = self.learnable_filter(x)
        
        x1 = x1.squeeze(1)
        # Supón que x1: [B, A, D]
        x1 = x1.reshape(-1, 1, self.num_detectors)  # [B*A, 1, D]

        # Aplicar los bloques residuales 1D
        x2 = self.interpolator_1(x1)
        x3 = self.interpolator_2(x2)
        x4 = self.interpolator_3(x3)
        x5 = self.interpolator_conv(x4)

        x5 = x5.view(-1, self.num_angles_modulo, self.num_detectors)
        x5 = x5.unsqueeze(1)               # [B, 1, A, D]

        # A.T() only accepts [1, A, D] so we iterate by batch
        images = self.AT(x5)  

        # apply denoiser to the output image
        x6 = self.denoising_conv_1(images)
        x7 = self.denoising_res_1(x6)
        x8 = self.denoising_res_2(x7)
        x9 = self.denoising_res_3(x8)
        x10 = self.denoising_conv_2(x9)

        return x10


class intermediate_residual_block(Module):
    """
    Create a depthwise 1D residual convolutional block for interpolation.

    Args:
        channels (int): Number of input/output channels (usually equal to num_angles).

    Returns:
        torch.nn.Sequential: A depthwise residual block.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.bn =  BatchNorm1d(channels)
        self.prelu = PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        return x + out
    

class denoising_residual_block(Module):
    """
    Create a residual block used in the denoising stage.

    Args:
        in_channels (int): Number of input and output channels.

    Returns:
        torch.nn.Sequential: A 2-layer convolutional residual block.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu =  ReLU(inplace=True)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out
    
class DeepFBP(ModelBase):
    """
    High-level wrapper for training and managing the Deep Filtered Backprojection (DeepFBP) model.

    This class extends `ModelBase` and encapsulates the DeepFBP-specific functionality, including
    configurable projection geometry (sparse or full view), structured training phases,
    and model configuration serialization.

    DeepFBP enhances classical FBP by integrating:
        - Trainable frequency-domain filtering (shared or per-angle)
        - Learnable 1D angular interpolation modules
        - Residual 2D CNN-based image denoising

    Args:
        model_path (str): Path to save model checkpoints and logs.
        filter_type (str): Filtering strategy. "Filter I" for shared filter, others for per-angle.
        sparse_view (bool): Whether to use sparse-angle projection geometry.
        view_angles (int): Number of angles if using sparse-view mode.
        alpha (float): Log-scaling factor for sinogram preprocessing.
        i_0 (float): Incident photon count for noise simulation.
        sigma (float): Standard deviation for additive Gaussian noise.
        batch_size (int): Number of samples per training batch.
        epochs (int): Total number of training epochs.
        learning_rate (float): Optimizer learning rate.
        debug (bool): If True, enables verbose output and plotting.
        seed (int): Random seed for reproducibility.
        accelerator (torch.device): Device for training.
        scheduler (str): Learning rate scheduler identifier.
        log_file (str): Log file path for training events.

    Attributes:
        model (DeepFBPNetwork): The underlying neural architecture.
        current_phase (int): The training phase currently in use (1–3).
    """

    def __init__(self, model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed,accelerator, scheduler, log_file):
        super().__init__(model_path, "DeepFBP", False, 1, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        self.filter_type = filter_type
        
        self.current_phase = None

        self.A_deepfbp, self.pg_deepfbp, self.num_angles_deepfbp = self._build_projection_operator()


        self.model = DeepFBPNetwork(self.num_detectors, self.num_angles_deepfbp, self.A_deepfbp, self.filter_type, self.pixel_size)


    def _build_projection_operator(self):
        """
        Constructs the appropriate tomosipo projection operator based on view configuration.

        If sparse-view reconstruction is enabled, this method subsamples the available
        projection angles using evenly spaced indices and builds a sparse-view operator.
        Otherwise, it uses the full set of angles and the default projection operator.

        Returns:
            A (ts.Operator): Tomosipo forward projection operator.
            pg (ts.ProjectionGeometry): Associated projection geometry.
            num_angles (int): Number of projection angles used.

        Notes:
            - The sparse-view operator is useful for simulating reduced acquisition scenarios.
            - The selected angles are stored via `self.indices` and reused elsewhere.
            - This method also logs which configuration is applied.

        Example:
            >>> A, pg, n_angles = self._build_projection_operator()
        """
        
        if self.sparse_view:
            self.indices = torch.linspace(0, self.num_angles - 1, steps=self.view_angles).long()
            angles_sparse = self.angles[self.indices]
            pg = ts.cone(angles=angles_sparse, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))
            A = ts.operator(self.vg, pg)
            num_angles = self.view_angles
            self._log(f"[Geometry] Using sparse-view geometry with {num_angles} angles.")
        else:
            pg = self.pg
            A = self.A
            num_angles = self.num_angles
            self._log(f"[Geometry] Using full-view geometry with {num_angles} angles.")
        
        return A, pg, num_angles


    def set_training_phase(self, phase):
        """
        Configures model parameter training for staged learning.

        Phase options:
            1 → Train only the learnable frequency filter.
            2 → Train the filter + interpolator blocks.
            3 → Train the entire model (filter + interpolator + denoiser).

        Args:
            phase (int): Training stage identifier (1, 2, or 3).

        Raises:
            ValueError: If the input phase is not one of [1, 2, 3].
        """
        # change all parameters to no require gradient
        for param in self.model.parameters():
            param.requires_grad = False

        if phase == 1:
            self._log("[TrainPhase] Activating only the learnable filter")
            for param in self.model.learnable_filter.parameters():
                param.requires_grad = True

        elif phase == 2:
            self._log("[TrainPhase] Activating learnable filter and interpolators")
            for param in self.model.learnable_filter.parameters():
                param.requires_grad = True

            for interpolator in [self.model.interpolator_1, self.model.interpolator_2, self.model.interpolator_3, self.model.interpolator_conv]:
                for param in interpolator.parameters():
                    param.requires_grad = True

        elif phase == 3:
            self._log("[TrainPhase] Activating all model parameters")
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid training phase. Use 1, 2 or 3.")
        
        #change the phase parameter to the correct training phase
        self.current_phase = phase


    def train_deepFBP(self, training_path, validation_path, save_path, max_len_train=None, max_len_val=None, patience=10, epochs = None, learning_rate= None, confirm_train=False, show_examples=True, number_of_examples=1, phase=1):
        """
        Conducts training of the DeepFBP model according to a selected training phase.

        Args:
            training_path (str): Path to training data.
            validation_path (str): Path to validation data.
            save_path (str): Where to save the trained model.
            max_len_train (int, optional): Max number of training samples to use.
            max_len_val (int, optional): Max number of validation samples to use.
            patience (int): Early stopping patience.
            epochs (int, optional): Override number of epochs.
            learning_rate (float, optional): Override learning rate.
            confirm_train (bool): Whether to prompt confirmation before training.
            show_examples (bool): If True, visualize sample reconstructions.
            number_of_examples (int): Number of visual samples to show.
            phase (int): Training phase (1, 2, or 3).

        Returns:
            dict: Training history with metrics (loss, PSNR, SSIM).
        """
        # set the training phase
        self.set_training_phase(phase)

        # apply training algorithm from the ModelBase Class
        history = self.train(training_path, validation_path, f"{save_path}_{self.current_phase}", max_len_train, max_len_val, patience,epochs ,learning_rate, confirm_train, show_examples, number_of_examples)
        return history


    def save_config(self):
        """
        Saves model hyperparameters and configuration to a JSON file.

        The file is saved to `{model_path}_config.json`. It includes training parameters,
        model structure, device, and logging path.
        """

        config = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "sparse-view": self.sparse_view,
            "view_angles": self.view_angles,
            "filter_type" : self.filter_type,
            "current_phase" : self.current_phase,
            "accelerator" : str(self.accelerator.device),
            "alpha": self.alpha,
            "i_0": self.i_0,
            "sigma": self.sigma,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "debug": self.debug,
            "seed": self.seed,
            "scheduler": self.scheduler,
            "log_file": self.logger.handlers[0].baseFilename if self.logger.handlers else "training.log"
        }
        #open file in writting mode
        with open(f"{self.model_path}_config.json", "w") as f:
            json.dump(config, f, indent=4)

        self._log(f"[DeepFBP] Configuration saved to: {self.model_path}_config.json")


class LearnableFilter(Module):
    """
    Learnable frequency-domain filter module for CT sinograms.

    This module replaces traditional analytical filters (e.g., Ram-Lak)
    with trainable parameters. Can operate in shared or per-angle mode.

    Args:
        init_filter (torch.Tensor): Initial frequency-domain filter (1D).
        per_angle (bool): If True, create one filter per projection angle.
        num_angles (int, optional): Required if `per_angle=True`.

    Attributes:
        weights (nn.Parameter): Filter weights in the frequency domain.

    Raises:
        AssertionError: If `per_angle=True` and `num_angles` is not provided.
    """

    def __init__(self, init_filter, per_angle=False, num_angles=None):
        super().__init__()
        self.per_angle = per_angle

        if self.per_angle:
            assert num_angles is not None, "num_angles must be provided when per_angle=True"
            filters = torch.stack([init_filter.clone().detach() for _ in range(num_angles)])
            self.register_parameter("weights", Parameter(filters))
        else:
            self.register_parameter("weights", Parameter(init_filter))


    def forward(self, x):
        """
        Apply the learnable frequency-domain filter to the input sinogram.

        Args:
            x (torch.Tensor): Input sinogram of shape (B, A, D).

        Returns:
            torch.Tensor: Filtered sinogram of same shape.

        Example:
            >>> filtered = filter_module(x)
        """

        # convert sinogram to frequency domain
        ftt1d = torch.fft.fft(x, dim=-1)

        # apply de filter
        if self.per_angle:
            filtered = ftt1d * self.weights[None, :, :]
        else:
            filtered = ftt1d * self.weights[None, None, :]
        
        # retorn the filtered sinogram 
        return torch.fft.ifft(filtered, dim=-1).real
