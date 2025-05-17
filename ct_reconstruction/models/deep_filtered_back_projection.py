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
import os


class DeepFBP(ModelBase):
    """
    Deep Filtered Back Projection (DeepFBP) network for CT reconstruction.

    Inherits from ModelBase and implements a mixture architecture.

    Architecture:
        - Initial filter learned from data.
        - Interpolation operations.
        - CNN post-processing for denoising.
    """


    def __init__(self, in_channels, model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, scheduler, log_file):
        
        # Initialize the base training infrastructure
        super().__init__(model_path, "DeepFBP", n_single_BP, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, scheduler, log_file)
        self.model = self

    def forward(self, x):
    
        return final_layer

    def save_config(self):
        """
        Saves model hyperparameters to a JSON config file for later restoration.
        """
        config = {
            "model_path": self.model_path,
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