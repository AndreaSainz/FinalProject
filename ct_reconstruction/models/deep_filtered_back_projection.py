import torch
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torch.nn import Sequential
from torch.nn import MSELoss
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


class DeepFBP(ModelBase):
    """
    Deep Filtered Back Projection (DeepFBP) network for CT reconstruction.

    Inherits from ModelBase and implements a mixture architecture.

    Architecture:
        - Initial filter learned from data.
        - Interpolation operations.
        - CNN post-processing for denoising.
    """


    def __init__(self, in_channels, training_path, validation_path, test_path, model_path, n_single_BP, alpha, i_0, sigma, max_len, batch_size, epochs, learning_rate, debug, seed, scheduler, log_file):
        
        # Initialize the base training infrastructure
        super().__init__(training_path, validation_path, test_path, model_path, "DeepFBP", n_single_BP, alpha, i_0, sigma, max_len, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, scheduler, log_file)


    def forward(self, x):
    

        return final_layer