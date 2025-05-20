import torch
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Conv1d, Conv2d
from torch.nn import PReLU, ReLU
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
        super().__init__(model_path, "DeepFBP", False, n_single_BP, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, scheduler, log_file)
        self.filter_type = filter_type

        # for python to know the parameters are learnable they should be define inside an nn.Module in init
        # initialize the filter as the 
        ram_lak = self.ram_lak_filter()
        # telling pyhton that the filter is learnable 
        if self.filter_type == "Filter I":
            self.learnable_filter = Parameter(ram_lak.clone().detach()) #[Detectors]
        else:
            stacked_filters = torch.stack([ram_lak.clone().detach() for _ in range(self.num_angles)])  # [Angles, Detectors]
            self.learnable_filter = Parameter(stacked_filters)


        self.interpolator_1 = self.intermediate_residual_block(channels=self.num_angles, kernel_size=3, stride=1, padding=1)
        self.interpolator_2 = self.intermediate_residual_block(channels=self.num_angles, kernel_size=3, stride=1, padding=1)
        self.interpolator_3 = self.intermediate_residual_block(channels=self.num_angles, kernel_size=3, stride=1, padding=1)
        self.interpolator_conv = Conv1d(self.num_angles, self.num_angles, kernel_size=3, stride=1, padding=1)

        self.denoising_conv_1 = Conv2d(in_channels= 1, out_channels= 64, kernel_size=3, stride=1, padding=1)
        self.denoising_res_1 = self.denoising_residual_block(in_channels=64, kernel_size=3, stride=1, padding=1)
        self.denoising_res_2 = self.denoising_residual_block(in_channels=64, kernel_size=3, stride=1, padding=1)
        self.denoising_res_3 = self.denoising_residual_block(in_channels=64, kernel_size=3, stride=1, padding=1)
        self.denoising_conv_2 = Conv2d(in_channels= 64, out_channels= 1, kernel_size=3, stride=1, padding=1)

        self.model = self


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
        
        # the ram-lak filter should be in the frequency domain
        frecuency_filter = torch.fft.fft(vector)
        frecuency_filter =  torch.fft.fftshift(frecuency_filter)

        return frecuency_filter


        
    def filter(self,x):
        # 1D Fast Fourier Transform
        ftt1d = torch.fft.fft(x, dim=-1)

        # Shift transformation
        ftt1d_shiffted = torch.fft.fftshift(ftt1d)

        # filtering values
        if self.filter_type == "Filter I":
            # Apply same filter across all angles
            filtered = ftt1d_shifted * self.learnable_filter  # broadcast over [B, A, D]
        else:
            # Apply angle-specific filter: [A, D] → broadcast over B
            filtered = ftt1d_shifted * self.learnable_filter.unsqueeze(0)  # [1, A, D]
        

        # transforming back to sinogram
        filter_sinogram = torch.fft.ifft(filtered, dim=-1).real

        return filter_sinogram



    def intermediate_residual_block(self, channels, kernel_size, stride, padding):
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


    def denoising_residual_block(self, in_channels, kernel_size, stride, padding):
        convolution = Sequential(
            Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU(inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        return convolution


    def set_training_phase(self, phase):
        """
        Configura qué partes del modelo son entrenables según la fase.

        Fase 1: Solo filtro.
        Fase 2: Filtro + interpoladores.
        Fase 3: Todos los módulos.
        """
        for param in self.parameters():
            param.requires_grad = False

        if phase == 1:
            self._log("[TrainPhase] Activating only the learnable filter")
            self.learnable_filter.requires_grad = True

        elif phase == 2:
            self._log("[TrainPhase] Activating learnable filter and interpolators")
            self.learnable_filter.requires_grad = True
            for interpolator in [self.interpolator_1,
                                self.interpolator_2,
                                self.interpolator_3,
                                self.interpolator_conv]:
                for param in interpolator.parameters():
                    param.requires_grad = True

        elif phase == 3:
            self._log("[TrainPhase] Activating all model parameters")
            for param in self.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Fase de entrenamiento no válida. Usa 1, 2 o 3.")


    def train_deepFBP(self, training_path, validation_path, save_path, max_len_train=None, max_len_val=None,
          patience=10, confirm_train=False, show_examples=True, number_of_examples=1, phase=1):

          """
        Performs full training with early stopping, metric logging, and learning rate scheduling.

        Args:
            training_path (str): Path to the training dataset.
            validation_path (str): Path to the validation dataset.
            max_len_train (int, optional): Maximum number of training samples.
            max_len_val (int, optional): Maximum number of validation samples.
            patience (int): Number of epochs to wait for improvement before early stopping.
            confirm_train (bool): Ask for user confirmation before starting training.

        Returns:
            dict: Training history with average loss, PSNR, and SSIM per epoch.
        """
        # set the phase we are in
        self.set_training_phase(phase)

        #changing paths parameters
        history = self.train(training_path, validation_path, save_path, max_len_train, max_len_val,
          patience, confirm_train, show_examples, number_of_examples)

        return history
       

    def forward(self, x):
        # initial part, filter
        x = self.filter(x)
        
        #Neural network for interpolator part
        res1 = self.interpolator_1(x)
        x += res1
        res2 = self.interpolator_2(x)
        x += res2
        res3 = self.interpolator_3(x)
        x += res3
        x = self.interpolator_conv(x)
        #interpolating (creating image)
        image = self.A.T(x)

        #denoising part( 1 conv, 3 residuals, 1 conv)
        image1 = self.denoising_conv_1(image)
        image2 = self.denoising_res_1(image1)
        image3 = image1 +image2 
        image4 = self.denoising_res_2(image3)
        image5 = image3 +image4 
        image6 =self.denoising_res_3(image5)
        image7 = image5 +image6
        image8 = self.denoising_conv_2(image7)

        return image8


    def print_filter(self, save_path=None, angles=None):
        """
        Visualiza el/los filtro(s) aprendidos en el dominio de la frecuencia.

        Args:
            save_path (str or None): Ruta para guardar la figura. Si es None, solo muestra en pantalla.
            angles (list or None): Lista de índices de ángulos a mostrar (solo para Filter II).
        """
        # Convertir a numpy y normalizar para mejor visualización
        if self.filter_type == "Filter I":
            filt = self.learnable_filter.detach().cpu().numpy()
            filt = np.abs(np.fft.fftshift(filt))
            filt /= filt.max()

            plt.figure(figsize=(5, 4))
            plt.plot(filt, color="tab:blue")
            plt.title("Learned Filter I")
            plt.xlabel("frequency")
            plt.ylabel("amplitude")
            plt.grid(True)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()

        elif self.filter_type == "Filter II":
            if angles is None:
                raise ValueError("You must provide a list of angles for Filter II.")

            filt_all = self.learnable_filter.detach().cpu().numpy()  # Shape: (num_angles, num_detectors)
            filt_all = np.abs(np.fft.fftshift(filt_all, axes=1))
            filt_all = filt_all / np.max(filt_all)

            mean_filter = np.mean(filt_all, axis=0)

            num_angles = len(angles)
            cols = min(4, num_angles + 1)  # limit to 4 columns max
            rows = (num_angles + 1 + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3))
            axes = np.atleast_1d(axes).flatten()

            # Plot mean filter
            axes[0].plot(mean_filter, color="orange")
            axes[0].set_title("Mean of Filter II")
            axes[0].set_xlabel("frequency")
            axes[0].set_ylabel("amplitude")
            axes[0].grid(True)

            # Plot selected angles
            for i, idx in enumerate(angles):
                if i + 1 >= len(axes):
                    break
                axes[i + 1].plot(filt_all[idx], color="orange")
                axes[i + 1].set_title(f"Filter II at {idx}°")
                axes[i + 1].set_xlabel("frequency")
                axes[i + 1].set_ylabel("amplitude")
                axes[i + 1].grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()

        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")

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