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
from functions.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import json
import random
import numpy as np
from torchsummary import summary
from functions.model import ModelBase




class DBP(ModelBase):
    """
    Deep Backprojection (DBP) network for CT reconstruction.

    Inherits from ModelBase and implements a CNN architecture.

    Architecture:
        - Initial Conv2d + ReLU layer.
        - 15 repeated Conv2d + BatchNorm2d + ReLU blocks.
        - Final Conv2d layer producing single-channel output.
    """


    def __init__(self, in_channels, training_path, validation_path, test_path, model_path, n_single_BP, alpha, i_0, sigma, max_len, batch_size, epochs, optimizer_type, loss_type, learning_rate, debug, seed, log_file):
        
        # Initialize the base training infrastructure
        super().__init__(training_path, validation_path, test_path, model_path, "DBP", n_single_BP, alpha, i_0, sigma, max_len, batch_size, epochs, optimizer_type, loss_type,learning_rate, debug, seed, log_file)

        self.in_channels = in_channels
        # initial layer
        self.conv1 = self.initial_layer(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)

        # middel layer (15 equal layers)
        self.middle_blocks = ModuleList([
            self.conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) 
            for _ in range(15)])

        # last layer
        self.final = self.final_layer(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)



    def initial_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates the initial convolutional layer with ReLU activation.
        
        Returns:
            Sequential: Sequential model with Conv2d + ReLU.
        """
        initial = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ReLU(inplace=True))
        return initial



    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a convolutional block with Conv2d, BatchNorm2d, and ReLU.
        
        Returns:
            Sequential: Sequential model with Conv2d + BatchNorm2d + ReLU.
        """
        convolution = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), BatchNorm2d(out_channels), ReLU(inplace=True))
        return convolution



    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates the final convolutional layer without activation.

        Returns:
            Conv2d: Output Conv2d layer.
        """
        final = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        return final



    def forward(self, x):
        # initial part
        conv1 = self.conv1(x)

        # middle part
        middle = conv1
        for block in self.middle_blocks:
            middle = block(middle)

        #final part
        final_layer = self.final(middle)

        return final_layer