import torch
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torch.nn import Sequential
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..datasets.dataset import LoDoPaBDataset
from ..callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import json
import random
import numpy as np
from torchsummary import summary
from ..models.model import ModelBase
from accelerate import Accelerator




class DBP(ModelBase):
    """
    Deep Backprojection (DBP) network for CT image reconstruction.

    This model uses a deep CNN to reconstruct images from multiple single-angle
    backprojections. It inherits utilities from `ModelBase` for training, evaluation, 
    and configuration management.

    The architecture consists of:
        - Initial layer: Conv2d + ReLU
        - Middle layers: 15 repeated blocks of Conv2d + BatchNorm2d + ReLU
        - Final layer: Conv2d (maps to a single channel output)

    Args:
        model_path (str): Directory to save the model weights and logs.
        n_single_BP (int): Number of single-angle backprojections per input sample.
        alpha (float): Scaling factor applied to the log-transformed projections.
        i_0 (float): Incident X-ray intensity, used to simulate Poisson noise.
        sigma (float): Standard deviation of Gaussian noise to simulate.
        batch_size (int): Number of training samples per batch.
        epochs (int): Total number of training epochs.
        learning_rate (float): Initial learning rate for the optimizer.
        debug (bool): If True, enables debug logging and verbose output.
        seed (int): Seed for reproducibility.
        accelerator (torch.device): Device to use (e.g., `torch.device("cuda")`).
        scheduler (str): Name of learning rate scheduler to use.
        log_file (str): Path to the log file.

    Attributes:
        conv1 (nn.Sequential): Initial convolutional block (Conv2d + ReLU).
        middle_blocks (nn.ModuleList): List of 15 residual CNN blocks.
        final (nn.Conv2d): Final convolutional layer (no activation).
        model (nn.Sequential): Full model assembled sequentially.

    Example:
        >>> import torch
        >>> from ct_reconstruction.models import DBP
        >>> model = DBP(
        ...     model_path="models/dbp",
        ...     n_single_BP=10,
        ...     alpha=0.001,
        ...     i_0=1e5,
        ...     sigma=0.01,
        ...     batch_size=4,
        ...     epochs=50,
        ...     learning_rate=1e-3,
        ...     debug=False,
        ...     seed=42,
        ...     accelerator=torch.device("cuda"),
        ...     scheduler="ReduceLROnPlateau",
        ...     log_file="logs/dbp_training.log"
        ... )
        >>> output = model(torch.randn(4, 10, 128, 128))  # forward pass
        >>> model.save_config()  # save hyperparameters
    """


    def __init__(self, model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file):

        # Initialize the base training infrastructure
        super().__init__(model_path, "DBP", True, n_single_BP, alpha, i_0, sigma, batch_size, epochs, "Adam", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        

        # initial layer
        self.conv1 = self.initial_layer(in_channels=self.n_single_BP, out_channels=64, kernel_size=3, stride=1, padding=1)

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
        Constructs the initial convolutional block of the network.

        Applies:
            - Conv2d
            - ReLU activation

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Convolution stride.
            padding (int): Amount of zero-padding added to both sides.

        Returns:
            nn.Sequential: A sequential block with Conv2d and ReLU.
        """
        initial = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ReLU(inplace=True))
        return initial



    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Constructs a middle convolutional block used in the DBP network.

        Applies:
            - Conv2d
            - BatchNorm2d
            - ReLU

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            stride (int): Stride for convolution.
            padding (int): Padding to be added.

        Returns:
            nn.Sequential: A sequential block with Conv2d, BatchNorm2d, and ReLU.
        """
        convolution = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), BatchNorm2d(out_channels), ReLU(inplace=True))
        return convolution



    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Constructs the final convolutional layer that outputs a single-channel image.

        This layer does not apply any activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (typically 1).
            kernel_size (int): Convolution kernel size.
            stride (int): Stride for convolution.
            padding (int): Padding to apply.

        Returns:
            nn.Conv2d: Final convolutional layer.
        """
        final = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        return final



    def forward(self, x):
        """
        Defines the forward pass of the DBP network.

        Applies:
            - Initial convolutional block
            - 15 middle residual CNN blocks
            - Final convolution layer

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where:
                B: batch size,
                C: number of backprojection channels (n_single_BP),
                H, W: height and width of the input image.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, H, W), the reconstructed image.
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


    def save_config(self):
        """
        Saves model hyperparameters and configuration to a JSON file.

        The file is saved to `{model_path}_config.json`. It includes training parameters,
        model structure, device, and logging path.
        """
        config = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "accelerator" : str(self.accelerator.device),
            "n_single_BP": self.n_single_BP,
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

        self._log(f"[DBP] Configuration saved to: {self.model_path}_config.json")