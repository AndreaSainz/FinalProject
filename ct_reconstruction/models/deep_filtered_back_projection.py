import torch
from torch.nn import Module, Sequential, Conv1d, Conv2d, BatchNorm1d, PReLU, ReLU, Parameter, Hardtanh
from torch.nn.functional import pad
import matplotlib.pyplot as plt
import tomosipo as ts
from tomosipo.torch_support import to_autograd
import json
from ..models.model import ModelBase


class LearnableFilter(Module):
    """
    Learnable frequency-domain filter module for CT sinograms.

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


class DenoisingResidualBlock(Module):
    """
    2D residual block for image denoising.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = Sequential(
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            ReLU(inplace=True),
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return torch.clamp(x + self.block(x), 0, 1)


class DeepFBPNetwork(Module):
    """
    Deep Filtered Backprojection Network for CT reconstruction.

    Pipeline:
        - Learnable frequency filter (Ram-Lak based init)
        - Angular interpolation via depthwise convolutions
        - Differentiable backprojection (Tomosipo)
        - Residual CNN-based denoiser

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
        self.A_ = A
        self.AT = to_autograd(self.A_.T, is_2d=True, num_extra_dims=2)

        # Padding parameters
        self.projection_size_padded = self.compute_projection_size_padded()
        self.padding = self.projection_size_padded - self.num_detectors

        # Initialize Ram-Lak filter in frequency domain
        ram_lak = self.ram_lak_filter(self.projection_size_padded)
        if filter_type == "Filter I":
            self.learnable_filter = LearnableFilter(ram_lak, per_angle=False)
        else:
            self.learnable_filter = LearnableFilter(ram_lak, per_angle=True, num_angles=self.num_angles_)

        print(self.learnable_filter.weights.shape)

        # Interpolation blocks
        self.interpolator_1 = IntermediateResidualBlock(1)
        self.interpolator_2 = IntermediateResidualBlock(1)
        self.interpolator_3 = IntermediateResidualBlock(1)
        self.interpolator_conv = Conv1d(1, 1, kernel_size=3, padding=1, bias = False)

        # Tomosipo normalization map (1s projection)
        sinogram_ones = torch.ones((1,1, self.num_angles_, num_detectors), device=self.device)
        self.tomosipo_normalizer = self.AT(sinogram_ones) + 1e-6

        # Denoising blocks
        self.denoising_conv_1 = Conv2d(1, 64, kernel_size=1)
        self.denoising_conv_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoising_res_1 = DenoisingResidualBlock(64)
        self.denoising_res_2 = DenoisingResidualBlock(64)
        self.denoising_res_3 = DenoisingResidualBlock(64)
        self.denoising_output = Sequential(
            Conv2d(64, 1, kernel_size=3, padding=1),
            Hardtanh(0, 1)
        )


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
        x = self.denoising_conv_1(img)
        x = self.denoising_conv_2(x)
        x = self.denoising_res_1(x)
        x = self.denoising_res_2(x)
        x = self.denoising_res_3(x)
        x = self.denoising_output(x)

        return x
    
    



class DeepFBP(ModelBase):
    """
    High-level wrapper for training and managing the Deep Filtered Backprojection (DeepFBP) model.

    This class extends `ModelBase` and encapsulates the DeepFBP-specific functionality, including
    configurable projection geometry (sparse or full view), structured training phases,
    and model configuration serialization.

    DeepFBP enhances classical FBP by integrating:
        - Trainable frequency-domain filtering (shared or per-angle)
        - Learnable 1D angular interpolation modules
        - Residual 2D CNN-based image denoising

    Args:
        model_path (str): Path to save model checkpoints and logs.
        filter_type (str): Filtering strategy. "Filter I" for shared filter, others for per-angle.
        sparse_view (bool): Whether to use sparse-angle projection geometry.
        view_angles (int): Number of angles if using sparse-view mode.
        alpha (float): Log-scaling factor for sinogram preprocessing.
        i_0 (float): Incident photon count for noise simulation.
        sigma (float): Standard deviation for additive Gaussian noise.
        batch_size (int): Number of samples per training batch.
        epochs (int): Total number of training epochs.
        learning_rate (float): Optimizer learning rate.
        debug (bool): If True, enables verbose output and plotting.
        seed (int): Random seed for reproducibility.
        accelerator (torch.device): Device for training.
        scheduler (str): Learning rate scheduler identifier.
        log_file (str): Log file path for training events.

    Attributes:
        model (DeepFBPNetwork): The underlying neural architecture.
        current_phase (int): The training phase currently in use (1–3).
    """

    def __init__(self, model_path, filter_type, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed,accelerator, scheduler, log_file):
        super().__init__(model_path, "DeepFBP", False, 1, sparse_view, view_angles, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        self.filter_type = filter_type
        
        self.current_phase = None
        self.A_deepfbp = self._get_operator()
        self.num_angles_deepfbp = self.view_angles if self.sparse_view else self.num_angles

        self.model = DeepFBPNetwork(self.num_detectors, self.num_angles_deepfbp, self.A_deepfbp, self.filter_type, self.device)



    def set_training_phase(self, phase):
        """
        Configures model parameter training for staged learning.

        Phase options:
            1 → Train only the learnable filter and output normalizer.
            2 → Filter + interpolators + output normalizer.
            3 → All model components.
        """
        # Disable all gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Always train the filter
        for param in self.model.learnable_filter.parameters():
                param.requires_grad = True

        if phase == 1:
            self._log("[TrainPhase] Activating learnable filter ")

        elif phase == 2:
            self._log("[TrainPhase] Activating filter and interpolators")
            for interpolator in [self.model.interpolator_1, self.model.interpolator_2,
                                self.model.interpolator_3, self.model.interpolator_conv]:
                for param in interpolator.parameters():
                    param.requires_grad = True

        elif phase == 3:
            self._log("[TrainPhase] Activating all model parameters")
            for param in self.model.parameters():
                param.requires_grad = True

        else:
            raise ValueError("Invalid training phase. Use 1, 2 or 3.")

        self.current_phase = phase


    def train_deepFBP(self, training_path, validation_path, save_path, max_len_train=None, max_len_val=None, patience=10, epochs = None, learning_rate= None, confirm_train=False, show_examples=True, number_of_examples=1, phase=1):
        """
        Conducts training of the DeepFBP model according to a selected training phase.

        Args:
            training_path (str): Path to training data.
            validation_path (str): Path to validation data.
            save_path (str): Where to save the trained model.
            max_len_train (int, optional): Max number of training samples to use.
            max_len_val (int, optional): Max number of validation samples to use.
            patience (int): Early stopping patience.
            epochs (int, optional): Override number of epochs.
            learning_rate (float, optional): Override learning rate.
            confirm_train (bool): Whether to prompt confirmation before training.
            show_examples (bool): If True, visualize sample reconstructions.
            number_of_examples (int): Number of visual samples to show.
            phase (int): Training phase (1, 2, or 3).

        Returns:
            dict: Training history with metrics (loss, PSNR, SSIM).
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
            "sparse-view": self.sparse_view,
            "view_angles": self.view_angles,
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
