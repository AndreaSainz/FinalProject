import torch
from torch.nn import Module
from torch.nn import Conv1d, Conv2d, BatchNorm1d, Sequential, PReLU, ReLU, Parameter
from ..models.model import ModelBase
import json

class DeepFBPNetwork(Module):
    """
    Internal model for Deep Filtered Backprojection (DeepFBP) CT reconstruction.

    This neural network combines a learnable FBP filtering stage, interpolation blocks,
    and a CNN-based denoiser to reconstruct high-quality CT images from sinograms.

    Components:
        - Learnable filter module (shared or per-angle)
        - Depthwise 1D convolutional interpolators
        - Differentiable backprojection layer
        - 2D CNN denoising module

    Args:
        num_detectors (int): Number of detectors in the CT system.
        num_angles (int): Number of projection angles.
        A (callable): Forward projection operator (Radon transform).
        filter_type (str): Filtering strategy. Use "Filter I" for shared filtering.

    Attributes:
        learnable_filter (LearnableFilter): Trainable frequency-domain filter module.
        interpolator_{1-3} (nn.Sequential): Depthwise 1D convolutional interpolation blocks.
        interpolator_conv (nn.Conv1d): Final 1D convolutional layer for angular smoothing.
        denoising_conv_1 (nn.Conv2d): Initial convolution layer for image denoising.
        denoising_res_{1-3} (nn.Sequential): Intermediate residual blocks for denoising.
        denoising_conv_2 (nn.Conv2d): Final convolution layer to reconstruct output image.
    """

    def __init__(self, num_detectors, num_angles, A, filter_type):
        # initilize parameter from parent clase Module
        super().__init__()

        #Initilize all attributes for this class
        self.num_detectors = num_detectors
        self.num_angles = num_angles
        self.A = A
        self.filter_type = filter_type

        # initilize filter as ram-lak filter
        ram_lak = self.ram_lak_filter()

        #create the filter from the class
        if self.filter_type == "Filter I":
            self.learnable_filter = LearnableFilter(ram_lak.clone().float(), per_angle=False)
        else:
            self.learnable_filter = LearnableFilter(ram_lak.clone().float(), per_angle=True, num_angles=self.num_angles)

        #initilize interpolator
        self.interpolator_1 = self.intermediate_residual_block(num_angles)
        self.interpolator_2 = self.intermediate_residual_block(num_angles)
        self.interpolator_3 = self.intermediate_residual_block(num_angles)
        self.interpolator_conv = Conv1d(num_angles, num_angles, kernel_size=3, stride=1, padding=1)

        #initilize denoising part
        self.denoising_conv_1 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.denoising_res_1 = self.denoising_residual_block(64)
        self.denoising_res_2 = self.denoising_residual_block(64)
        self.denoising_res_3 = self.denoising_residual_block(64)
        self.denoising_conv_2 = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


    def ram_lak_filter(self):
        """
        Compute the discrete Ram-Lak filter in the spatial domain. This filter corresponds 
        to the ideal ramp filter :math:`|\omega|` in the frequency domain, commonly used in 
        Filtered Back Projection (FBP) for CT reconstruction.

        Returns:
            torch.Tensor: Real-valued frequency domain filter of shape (num_detectors,).

        Notes:
            This filter is derived from:
            Kak, A. C., & Slaney, M. (1988). Principles of computerized tomographic imaging.
        """
        #initilize a vector with all zeros
        vector = torch.zeros(self.num_detectors)

        #find the center
        center = self.num_detectors//2
        vector[center] =1 / (4 * self.num_detectors**2)

        # get all the "negative" odd numbers
        for i in range(-center,center):
            if i%2 != 0 and  i!=0:
                vector[center + i] = -1 / (torch.pi ** 2 * i ** 2)   
        
        # the ram-lak filter should be in the frequency domain
        frecuency_filter = torch.fft.fft(vector)
        frecuency_filter =  torch.fft.fftshift(frecuency_filter)

        return frecuency_filter.real


    def intermediate_residual_block(self, channels):
        """
        Create a depthwise 1D residual convolutional block for interpolation.

        Args:
            channels (int): Number of input/output channels (usually equal to num_angles).

        Returns:
            torch.nn.Sequential: A depthwise residual block.
        """
        return Sequential(
            Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            BatchNorm1d(channels),
            PReLU()
        )


    def denoising_residual_block(self, in_channels):
        """
        Create a residual block used in the denoising stage.

        Args:
            in_channels (int): Number of input and output channels.

        Returns:
            torch.nn.Sequential: A 2-layer convolutional residual block.
        """
        return Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        """
        Executes the forward pass through the DeepFBPNetwork.

        The pipeline includes:
            - Learnable filtering in the frequency domain
            - Angular interpolation using depthwise convolutions
            - Differentiable backprojection via custom operator
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
        x = self.learnable_filter(x)
        x = x.squeeze(1)

        # Apply "interpolator", is changing the values of the sinogram directly, is more like a denoiser
        x = x + self.interpolator_1(x)
        x = x + self.interpolator_2(x)
        x = x + self.interpolator_3(x)
        x = self.interpolator_conv(x)

        x = x.unsqueeze(1)               # [B, 1, A, D]

        # A.T() only accepts [1, A, D] so we iterate by batch
        images = [DifferentiableBackprojection.apply(xi, self.A) for xi in x]  
        image = torch.stack(images, dim=0) 

        # apply denoiser to the output image
        x = self.denoising_conv_1(image)
        x = x + self.denoising_res_1(x)
        x = x + self.denoising_res_2(x)
        x = x + self.denoising_res_3(x)
        x = self.denoising_conv_2(x)
        return x


class DeepFBP(ModelBase):
    """
    Main interface for training and managing the Deep Filtered Backprojection (DeepFBP) model.

    This class wraps the `DeepFBPNetwork` and provides a high-level interface for training
    in structured phases, saving configurations, and controlling model behavior
    across the CT reconstruction pipeline.

    The DeepFBP model enhances classical Filtered Backprojection by introducing:
        - Learnable frequency-domain filters
        - 1D angular interpolation modules
        - A 2D CNN denoising stage

    Args:
        model_path (str): Path to save model checkpoints and logs.
        filter_type (str): Type of filter. Options:
            - "Filter I": Shared filter across angles.
            - Any other string: Per-angle filters.
        alpha (float): Log scaling factor for projection values.
        i_0 (float): Incident photon intensity (used for simulating noise).
        sigma (float): Gaussian noise standard deviation.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.
        learning_rate (float): Optimizer learning rate.
        debug (bool): Enable verbose debug output.
        seed (int): Seed for reproducibility.
        accelerator (torch.device): Device for training (e.g., `torch.device("cuda")`).
        scheduler (str): Learning rate scheduler type.
        log_file (str): Path to output log file.

    Attributes:
        model (DeepFBPNetwork): Core model instance.
        current_phase (int): Current training phase (1, 2, or 3).

    Example:
        >>> from ct_reconstruction.models import DeepFBP
        >>> model = DeepFBP(
        ...     model_path="checkpoints/deepfbp",
        ...     filter_type="Filter I",
        ...     alpha=0.001,
        ...     i_0=1e5,
        ...     sigma=0.01,
        ...     batch_size=4,
        ...     epochs=100,
        ...     learning_rate=1e-3,
        ...     debug=True,
        ...     seed=42,
        ...     accelerator=torch.device("cuda"),
        ...     scheduler="StepLR",
        ...     log_file="train.log"
        ... )
        >>> model.train_deepFBP(
        ...     training_path="data/train",
        ...     validation_path="data/val",
        ...     save_path="checkpoints/deepfbp",
        ...     phase=1
        ... )
        >>> model.save_config()
    """

    def __init__(self, model_path, filter_type, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed,accelerator, scheduler, log_file):
        super().__init__(model_path, "DeepFBP", False, 1, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        self.filter_type = filter_type
        self.model = DeepFBPNetwork(self.num_detectors, self.num_angles, self.A, self.filter_type)
        self.current_phase = None

    def set_training_phase(self, phase):
        """
        Set model training phase and selectively activate parts of the network.

        Args:
            phase (int): Training phase (1: filter only, 2: filter + interpolation, 3: all layers).

        Raises:
            ValueError: If an invalid phase is specified.
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
        Trains the DeepFBP model using a specific training phase.

        Args:
            training_path (str): Path to training dataset.
            validation_path (str): Path to validation dataset.
            save_path (str): Path to save model checkpoints.
            max_len_train (int, optional): Max training samples.
            max_len_val (int, optional): Max validation samples.
            patience (int): Early stopping patience.
            confirm_train (bool): Ask for user confirmation before training.
            show_examples (bool): Show reconstructed examples during training.
            number_of_examples (int): Number of examples to show.
            phase (int): Training phase to activate.

        Returns:
            dict: Training history metrics.

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
        # shifted frequencies
        ftt1d_shifted = torch.fft.fftshift(ftt1d)

        # apply de filter
        if self.per_angle:
            filtered = ftt1d_shifted * self.weights.unsqueeze(0)
        else:
            filtered = ftt1d_shifted * self.weights
        
        # retorn the filtered sinogram 
        return torch.fft.ifft(filtered, dim=-1).real


class DifferentiableBackprojection(torch.autograd.Function):
    """
    Differentiable wrapper for the backprojection operator.

    Enables gradient flow through custom CT backprojection routines using PyTorch's autograd.

    Methods:
        forward(x, operator): Applies backprojection using `operator.T(x)`.
        backward(grad_output): Applies forward projection using `operator(grad_output)`.

    Args:
        x (torch.Tensor): Input tensor of shape (1, A, D).
        operator (callable): Custom linear operator (e.g., TomoSipo Radon transform).

    Returns:
        torch.Tensor: Reconstructed image.

    Example:
        >>> image = DifferentiableBackprojection.apply(sinogram, A)
    """
    @staticmethod
    def forward(ctx, x, operator):
        ctx.operator = operator
        ctx.save_for_backward(x)
        return operator.T(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # We assume that the gradient of A.T is simply A
        grad_input = ctx.operator(grad_output)
        return grad_input, None  # second arg is for `operator`, which has no gradient