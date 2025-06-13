import torch
from torch.nn import Module, ModuleList, Sequential, Conv1d, Conv2d, BatchNorm1d, PReLU, ReLU, Parameter, Hardtanh, BatchNorm2d
from torch.nn.functional import pad
import matplotlib.pyplot as plt
import tomosipo as ts
from tomosipo.torch_support import to_autograd
import json
from .model import ModelBase


class LearnableFilter(Module):
    """
    Learnable frequency-domain filter module for CT sinograms in FusionFBP.

    This module replaces traditional filters like Ram-Lak with trainable parameters
    in the frequency domain. It supports shared or per-angle filters.

    Args:
        init_filter (torch.Tensor): Initial 1D filter in the frequency domain.
        per_angle (bool): If True, uses one filter per angle.
        num_angles (int, optional): Required if per_angle=True.
    """

    def __init__(self, init_filter, per_angle=False, num_angles=None):
        super().__init__()
        self.per_angle = per_angle

        if per_angle:
            assert num_angles is not None, "num_angles must be provided when per_angle=True"
            filters = torch.stack([init_filter.clone().detach() for _ in range(num_angles)])
            self.register_parameter("weights", Parameter(filters))
        else:
            filters = torch.stack([init_filter.clone().detach()])  # shape: (1, D)
            self.register_parameter("weights", Parameter(filters))


    def forward(self, x):
        ftt1d = torch.fft.fft(x, dim=-1)
        if self.per_angle:
            filtered = ftt1d * self.weights[None, :, :]
        else:
            filter_shared = self.weights.expand(ftt1d.shape[1], -1)
            filtered = ftt1d * filter_shared[None, :, :]
        return torch.fft.ifft(filtered, dim=-1).real


class IntermediateResidualBlock(Module):
    """
    Depthwise 1D residual block used in angular interpolation.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = Sequential(
            Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            BatchNorm1d(channels),
            PReLU()
        )

    def forward(self, x):
        return x + self.block(x)


class DenoisingBlock(Module):
    """
    Deep CNN block for image denoising used in FusionFBP.

    Consists of an initial conv block, 15 intermediate conv+BN+ReLU blocks,
    and a final conv layer to restore the image. 

    This is slightly change arquitecture base on DBP model but adapted to only one initial image.
    """
    def __init__(self):

        # Initialize the base training infrastructure
        super().__init__()
        
        # initial layer
        self.conv1 = self.initial_layer(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        # middel layer (15 equal layers)
        self.middle_blocks = ModuleList([
            self.conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) 
            for _ in range(15)])

        # last layer
        self.final = self.final_layer(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        #change parameters
        self.model = Sequential(self.conv1,*self.middle_blocks,self.final)

        



    def initial_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds the initial convolutional block of the network.

        This block consists of:
            - Conv2d
            - ReLU activation

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Convolution stride.
            padding (int): Zero-padding to add to each side.

        Returns:
            nn.Sequential: A sequential block with Conv2d and ReLU.
        """
        initial = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ReLU(inplace=True))
        return initial



    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds an intermediate convolutional block for the DBP architecture.

        This block includes:
            - Conv2d
            - BatchNorm2d
            - ReLU activation

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Convolution stride.
            padding (int): Zero-padding to apply.

        Returns:
            nn.Sequential: A sequential block with Conv2d, BatchNorm2d, and ReLU.
        """
        convolution = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), BatchNorm2d(out_channels), ReLU(inplace=True))
        return convolution



    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds the final convolutional layer that produces the output image.

        This layer does not include an activation function.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (typically 1).
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Convolution stride.
            padding (int): Zero-padding to apply.

        Returns:
            nn.Conv2d: Final convolutional layer.
        """
        final = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        return final



    def forward(self, x):
        """
        Defines the forward pass of the DBP network.

        The input is passed sequentially through:
            - Initial convolutional block
            - Fifteen intermediate convolutional blocks
            - Final convolutional layer

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where:
                B = batch size,
                C = number of backprojection channels (`n_single_BP`),
                H, W = spatial dimensions.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, H, W), representing the reconstructed image.
        """
        
        # initial part
        conv1 = self.conv1(x)

        # middle part
        middle = conv1
        for block in self.middle_blocks:
            middle = block(middle)

        #final part
        final_layer = self.final(middle)

        return final_layer


class FusionFBPNetwork(Module):
    """
    FusionFBP Network for CT reconstruction.

    This architecture fuses physics-based filtering and backprojection with
    deep learning modules for interpolation and denoising.

    Pipeline:
        - Learnable frequency-domain filtering (Ram-Lak-based)
        - Angular interpolation via residual 1D convolutions
        - Differentiable backprojection using Tomosipo
        - Deep CNN-based image denoising

    Args:
        num_detectors (int): Number of detector bins.
        num_angles (int): Number of projection angles.
        A (ts.Operator): Tomosipo projection operator.
        filter_type (str): Either 'Filter I' (shared) or per-angle.
        device (torch.device): Computation device.
    """

    def __init__(self, num_detectors, num_angles, A, filter_type, device):
        super().__init__()

        self.num_detectors = num_detectors
        self.num_angles_ = num_angles
        self.device = device
        self.A = A
        self.AT = to_autograd(self.A.T, is_2d=True, num_extra_dims=2)

        # Padding parameters
        self.projection_size_padded = self.compute_projection_size_padded()
        self.padding = self.projection_size_padded - self.num_detectors

        # Initialize Ram-Lak filter in frequency domain
        ram_lak = self.ram_lak_filter(self.projection_size_padded)
        if filter_type == "Filter I":
            self.learnable_filter = LearnableFilter(ram_lak, per_angle=False)
        else:
            self.learnable_filter = LearnableFilter(ram_lak, per_angle=True, num_angles=self.num_angles_)

        # Interpolation blocks
        self.interpolator_1 = IntermediateResidualBlock(1)
        self.interpolator_2 = IntermediateResidualBlock(1)
        self.interpolator_3 = IntermediateResidualBlock(1)
        self.interpolator_conv = Conv1d(1, 1, kernel_size=3, padding=1, bias = False)

        # Tomosipo normalization map (1s projection)
        sinogram_ones = torch.ones((1,1, num_angles, num_detectors), device=self.device)
        self.tomosipo_normalizer = self.AT(sinogram_ones) + 1e-6

        # Denoising blocks
        self.denoiser = DenoisingBlock()
    

    def compute_projection_size_padded(self):
        """
        Computes the next power-of-two padding size to avoid aliasing.
        """
        return 2 ** int(torch.ceil(torch.log2(torch.tensor(self.num_detectors, dtype=torch.float32))).item())


    def ram_lak_filter(self, size):
        """
        Generates Ram-Lak filter directly in frequency domain.
        """
        steps = int(size / 2 + 1)
        ramp = torch.linspace(0, 1, steps, dtype=torch.float32)
        down = torch.linspace(1, 0, steps, dtype=torch.float32)
        f = torch.cat([ramp, down[:-2]])
        return f
            

    def forward(self, x):

        # Initial shape: [B, 1, A, D]
        x = x.squeeze(1)  # [B, A, D]
        x = pad(x, (0, self.padding), mode="constant", value=0)
        x = self.learnable_filter(x)
        x = x[..., :self.num_detectors]  # Remove padding

        # Interpolation network
        x = x.reshape(-1, 1, self.num_detectors)
        x = self.interpolator_1(x)
        x = self.interpolator_2(x)
        x = self.interpolator_3(x)
        x = self.interpolator_conv(x)
        x = x.view(-1, self.num_angles_, self.num_detectors).unsqueeze(1)
        
        # Differentiable backprojection
        img = self.AT(x)

        #normlise images
        img = img / self.tomosipo_normalizer

        # Denoising network
        x = self.denoiser(img)

        return x
    
    



class FusionFBP(ModelBase):
    """
    High-level wrapper for training and managing the FusionFBP model.

    This class integrates a trainable filtered backprojection with a CNN denoiser.
    It supports different projection geometries and training phases.

    FusionFBP combines:
        - A learnable Ram-Lak-like filter in the frequency domain
        - Residual depthwise convolutions for angular interpolation
        - A differentiable backprojection operator (Tomosipo)
        - A deep CNN-based denoising block inspired by DBP architectures

    Args:
        model_path (str): Path to save model checkpoints and logs.
        filter_type (str): Filter strategy ('Filter I' or per-angle).
        sparse_view (bool): Whether to use sparse-angle projection.
        view_angles (int): Number of views for sparse acquisition.
        alpha (float): Log scaling parameter for sinogram preproc.
        i_0 (float): Incident photon count for Poisson noise model.
        sigma (float): Gaussian noise level.
        batch_size (int): Number of samples per batch.
        epochs (int): Total training epochs.
        learning_rate (float): Optimizer learning rate.
        debug (bool): Verbosity toggle.
        seed (int): Random seed.
        accelerator (torch.device): Device used for training.
        scheduler (str): Learning rate scheduler.
        log_file (str): Path to the training log file.
    """

    def __init__(self, model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed,accelerator, scheduler, log_file):
        super().__init__(model_path, "FusionFBP", False, 1, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        self.filter_type = filter_type
        

        self.A_fusionFBP = self._get_operator()
        self.num_angles_fusionFBP = self.view_angles if self.sparse_view else self.num_angles

        self.model = FusionFBPNetwork(self.num_detectors, self.num_angles_fusionFBP, self.A_fusionFBP, self.filter_type, self.device)



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

        self._log(f"[FusionFBP] Configuration saved to: {self.model_path}_config.json")
