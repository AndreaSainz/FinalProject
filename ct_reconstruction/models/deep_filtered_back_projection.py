import torch
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Conv1d
from torch.nn import PReLU
from torch.nn import BatchNorm1d
from torch.nn import Sequential
from torch.nn import MSELoss
from torch.nn import Parameter
from torch.optim import AdamW
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
import os


class DeepFBP(ModelBase):
    """
    x: sinogram in this case
    Deep Filtered Back Projection (DeepFBP) network for CT reconstruction.

    Inherits from ModelBase and implements a mixture architecture.

    Architecture:
        - Initial filter learned from data.
        - Interpolation operations.
        - CNN post-processing for denoising.
    """


    def __init__(self, filter_type, model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, scheduler, log_file):
        
        # Initialize the base training infrastructure
        super().__init__(model_path, "DeepFBP", n_single_BP, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, scheduler, log_file)
        self.filter_type

        # for python to know the parameters are learnable they should be define inside an nn.Module in init
        # initialize the filter as the 
        ram_lak = ram_lak_filter()
        # telling pyhton that the filter is learnable 
        self.learnable_filter = Parameter(ram_lak.clone().detach())

        self.interpolator_1 = int_residual_block(channels=self.num_detectors, kernel_size=3, stride=1, padding=1)
        self.interpolator_2 = int_residual_block(channels=self.num_detectors, kernel_size=3, stride=1, padding=1)
        self.interpolator_3 = int_residual_block(channels=self.num_detectors, kernel_size=3, stride=1, padding=1)
        self.interpolator_conv = Conv1d(self.num_detectors, self.num_detectors, kernel_size=3, stride=1, padding=1)
        self.model = 


    def ram_lak_filter(self):
        """
        Compute the discrete Ram-Lak filter in the spatial domain.

        The filter, in the space domain, is defined as follows:

             h[n] = 
                1 / (4 * N^2)               if n == 0
                -1 / (π^2 * n^2)            if n is odd and n ≠ 0
                0                           if n is even and n ≠ 0
            
            @book{kak1988pct,
            title     = {Principles of Computerized Tomographic Imaging},
            author    = {Avinash C. Kak and Malcolm Slaney},
            year      = {1988},
            publisher = {IEEE Press},
            url       = {http://www.slaney.org/pct/}
            }
        Where:
        - n ranges from -N/2 to N/2 (assuming N is even),
        - N is the total number of points in the filter.

        This filter corresponds to the ideal ramp filter |ω| in the frequency domain,
        commonly used in Filtered Back Projection (FBP) for CT reconstruction.
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
        
        frecuency_filter = torch.fft.fft(vector)
        frecuency_filter =  torch.fft.fftshift(frecuency_filter)

        return frecuency_filter


        
    def filter1(self,x):
        # 1D Fast Fourier Transform
        ftt1d = torch.fft.fft(x, dim=-1)

        # Shift transformation
        ftt1d_shiffted = torch.fft.fftshift(ftt1d)

        # filtering values
        filter_ftt1d_shiffted= ftt1d_shiffted*self.learnable_filter

        # transforming back to sinogram
        filter_sinogram = torch.fft.ifft(filter_ftt1d_shiffted, dim=-1).real

        return filter_sinogram


    def filter2(self,x):
        # 1D Fast Fourier Transform (one per angle)
        ftt1d = torch.fft.fft(x, dim=-1)

        # Shift transformation 
        ftt1d_shiffted = torch.fft.fftshift(ftt1d, dim=-1)

        # filtering values, different for each angle (the unsqueeze create a filter per angle [num_angles, detectors])
        filter_ftt1d_shiffted= ftt1d_shiffted*self.learnable_filter.unsqueeze(0)

        # transforming back to sinogram
        filter_sinogram = torch.fft.ifft(filter_ftt1d_shiffted, dim=-1).real

        return filter_sinogram


    def int_residual_block(self, channels, kernel_size, stride, padding):
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
        convolution = Sequential(Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, group =channels), 
        BatchNorm1d(channels), 
        PReLU(inplace=True))
        return convolution


    def linear_interpolation(self, z):

        image = torch.zeros((self.batch_size, self.pixels,self.pixels))

        pxi,yi,θ = (1 − z) p′ (a) + zp′ (a + 1)



    def forward(self, x):
        # initial part, filter
        if self.filter_type == "Filter I":
            x = filter1(self,x)
        else:
            x = filter2(self,x)
        
        #interpolator part
        res1 = self.interpolator_1(x)
        x += res1
        res2 = self.interpolator_2(x)
        x += res2
        res3 = self.interpolator_3(x)
        x += res3

        x = self.interpolator_conv(x)








        return final_layer


    def save_config(self):
        """
        Saves model hyperparameters to a JSON config file for later restoration.
        """
        config = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "filter_type" : self.filter_type,
            "n_single_BP": self.n_single_BP,
            "alpha": self.alpha,
            "i_0": self.i_0,
            "sigma": self.sigma,
            "max_len_train": self.max_len_train,
            "max_len_val": self.max_len_val,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer_type": self.optimizer_type,
            "loss_type": self.loss_type,
            "learning_rate": self.learning_rate,
            "debug": self.debug,
            "seed": self.seed,
            "scheduler": self.scheduler,
            "log_file": self.logger.handlers[0].baseFilename if self.logger.handlers else "training.log"
        }
        with open(f"{self.model_path}_config.json", "w") as f:
            json.dump(config, f, indent=4)

        self._log(f"[DeepFBP] Configuration saved to: {self.model_path}_config.json")