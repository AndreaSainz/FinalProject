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




class DBP(ModelBase):
    """
    Deep Backprojection (DBP) network for CT reconstruction.

    This class defines a CNN-based architecture for reconstructing CT images
    from single backprojection inputs. It inherits the training, evaluation,
    and utility methods from ModelBase.

    Architecture:
        - Initial block: Conv2d + ReLU
        - Middle blocks: 15 × (Conv2d + BatchNorm2d + ReLU)
        - Final layer: Conv2d (1 output channel)

    Args:
        in_channels (int): Number of input channels (e.g., number of BPs per sample).
        training_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset.
        test_path (str): Path to the test dataset.
        model_path (str): Path to save model weights and logs.
        n_single_BP (int): Number of backprojections per sample.
        alpha (float): Normalization factor.
        i_0 (float): Incident X-ray intensity.
        sigma (float): Standard deviation of noise.
        max_len_train (int): Max number of training samples to load.
        max_len_val (int): Max number of validation samples to load.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        debug (bool): Whether to enable debug mode.
        seed (int): Random seed for reproducibility.
        scheduler
        log_file (str): Path to log file.
    """


    def __init__(self, model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, scheduler, log_file):

        # Initialize the base training infrastructure
        super().__init__(model_path, "DBP", True, n_single_BP, alpha, i_0, sigma, batch_size, epochs, "Adam", "MSELoss", learning_rate, debug, seed, scheduler, log_file)
        

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

        This block consists of a Conv2d followed by a ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride for the convolution.
            padding (int): Zero-padding added to both sides.

        Returns:
            nn.Sequential: Sequential block with Conv2d + ReLU.
        """
        initial = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ReLU(inplace=True))
        return initial



    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Constructs a middle convolutional block used in the DBP network.

        This block includes Conv2d, BatchNorm2d, and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride for the convolution.
            padding (int): Zero-padding added to both sides.

        Returns:
            nn.Sequential: Sequential block with Conv2d + BatchNorm2d + ReLU.
        """
        convolution = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), BatchNorm2d(out_channels), ReLU(inplace=True))
        return convolution



    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Constructs the final convolutional layer that maps to a single-channel output.

        This layer does not apply any activation function.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (usually 1).
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride for the convolution.
            padding (int): Zero-padding added to both sides.

        Returns:
            nn.Conv2d: Final convolutional layer.
        """
        final = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        return final



    def forward(self, x):
        """
        Defines the forward pass of the DBP network.

        Applies:
            - Initial convolution
            - 15 repeated conv blocks
            - Final convolutional layer

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Reconstructed image of shape (B, 1, H, W).
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
        Saves model hyperparameters to a JSON config file for later restoration.
        """
        config = {
            "model_type": self.model_type,
            "model_path": self.model_path,
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
        with open(f"{self.model_path}_config.json", "w") as f:
            json.dump(config, f, indent=4)

        self._log(f"[DBP] Configuration saved to: {self.model_path}_config.json")