import torch
from torch.nn import Module
import torch.nn
from torch.nn import Conv1d, Conv2d, BatchNorm1d, Sequential, PReLU, ReLU, Parameter
from ..models.model import ModelBase
from accelerate import Accelerator
import json

class DeepFBPNetwork(Module):
    def __init__(self, num_detectors, num_angles, A, filter_type):
        super().__init__()
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

        return frecuency_filter.real


    def intermediate_residual_block(self, channels):
        return Sequential(
            Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            BatchNorm1d(channels),
            PReLU()
        )


    def denoising_residual_block(self, in_channels):
        return Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        x = self.learnable_filter(x)
        x = x.squeeze(1)

        x = x + self.interpolator_1(x)
        x = x + self.interpolator_2(x)
        x = x + self.interpolator_3(x)
        x = self.interpolator_conv(x)

        x = x.unsqueeze(1)               # [B, 1, A, D]

        # A.T() solo acepta [1, A, D] así que iteramos por batch
        images = [DifferentiableBackprojection.apply(xi, self.A) for xi in x]         # cada xi: [1, A, D]
        image = torch.stack(images, dim=0) 

        x = self.denoising_conv_1(image)
        x = x + self.denoising_res_1(x)
        x = x + self.denoising_res_2(x)
        x = x + self.denoising_res_3(x)
        x = self.denoising_conv_2(x)
        return x


class DeepFBP(ModelBase):
    def __init__(self, model_path, filter_type, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed,accelerator, scheduler, log_file):
        super().__init__(model_path, "DeepFBP", False, 1, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        self.filter_type = filter_type
        self.model = DeepFBPNetwork(self.num_detectors, self.num_angles, self.A, self.filter_type)
        self.current_phase = None

    def set_training_phase(self, phase):
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
        
        self.current_phase = phase


    def train_deepFBP(self, training_path, validation_path, save_path, max_len_train=None, max_len_val=None, patience=10, confirm_train=False, show_examples=True, number_of_examples=1, phase=1):
        self.set_training_phase(phase)
        history = self.train(training_path, validation_path, f"{save_path}_{self.current_phase}", max_len_train, max_len_val, patience, confirm_train, show_examples, number_of_examples)
        return history


    def save_config(self):
        """
        Saves model hyperparameters to a JSON config file for later restoration.
        """
        config = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "filter_type" : self.filter_type,
            "current_phase" : self.current_phase,
            "accelerator" : self.accelerator,
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

        self._log(f"[DeepFBP] Configuration saved to: {self.model_path}_config.json")


class LearnableFilter(Module):
    def __init__(self, init_filter, per_angle=False, num_angles=None):
        super().__init__()
        self.per_angle = per_angle

        if self.per_angle:
            assert num_angles is not None, "num_angles must be provided when per_angle=True"
            filters = torch.stack([init_filter.clone().detach() for _ in range(num_angles)])
            self.register_parameter("weights", Parameter(filters))
        else:
            self.register_parameter("weights", Parameter(init_filter))

        # Verificar que el parámetro requiere gradiente
        print(f"LearnableFilter initialized: per_angle={self.per_angle}, weights.shape={self.weights.shape}, requires_grad={self.weights.requires_grad}")

    def forward(self, x):
        ftt1d = torch.fft.fft(x, dim=-1)
        if self.per_angle:
            filtered = ftt1d * self.weights.unsqueeze(0)
        else:
            filtered = ftt1d * self.weights
            
        return torch.fft.ifft(filtered, dim=-1).real


class DifferentiableBackprojection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, operator):
        ctx.operator = operator
        ctx.save_for_backward(x)
        return operator.T(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Asumimos que gradiente de A.T es simplemente A
        grad_input = ctx.operator(grad_output)
        return grad_input, None  # second arg is for `operator`, which has no gradient