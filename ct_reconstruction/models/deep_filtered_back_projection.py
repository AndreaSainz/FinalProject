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
            self.register_parameter("weights", Parameter(init_filter))

    def forward(self, x):
        ftt1d = torch.fft.fft(x, dim=-1)
        if self.per_angle:
            filtered = ftt1d * self.weights[None, :, :]
        else:
            filtered = ftt1d * self.weights[None, None, :]
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
        self.num_angles = num_angles
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
            self.learnable_filter = LearnableFilter(ram_lak, per_angle=True, num_angles=num_angles)

        # Interpolation blocks
        self.interpolator_1 = IntermediateResidualBlock(1)
        self.interpolator_2 = IntermediateResidualBlock(1)
        self.interpolator_3 = IntermediateResidualBlock(1)
        self.interpolator_conv = Conv1d(1, 1, kernel_size=3, padding=1, bias = False)

        # Tomosipo normalization map (1s projection)
        sinogram_ones = torch.ones((1, num_angles, num_detectors), device=self.device)
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
        return max(64, 2 ** (self.num_detectors * 2).bit_length())

    def ram_lak_filter(self, size):
        """
        Generates Ram-Lak filter directly in frequency domain.
        """
        freqs = torch.fft.fftfreq(size)
        return torch.abs(freqs)
            
    def forward(self, x):


        print(f" shape inicial {x.shape}")
        max_val = x[0,0].max()
        min_val =  x[0,0].min()
        plt.imshow(x[0,0].detach().cpu().numpy(), cmap='gray')
        plt.title(f"Sinograma inicial vmin/vmax (min = {min_val:.4f}, max={max_val:.4f})")
        plt.savefig("sinograma_inicial.png")
        plt.close()

        images1 = self.AT(x)  

        img_np = images1[0].squeeze().detach().cpu().numpy()
        max_val = img_np.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Imagen con vmin/vmax
        axs[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Tomosipo inicial vmin/vmax (max={max_val:.4f})")
        axs[0].axis("off")

        # Imagen sin vmin/vmax
        axs[1].imshow(img_np, cmap='gray')
        axs[1].set_title(f"Tomosipo inicial auto escala (max={max_val:.4f})")
        #axs[1].axis("off")

        plt.tight_layout()
        plt.savefig("imagen_tomosipo_inicial.png")
        plt.close()




        # Initial shape: [B, 1, A, D]
        x = x.squeeze(1)  # [B, A, D]
        x = pad(x, (0, self.padding), mode="constant", value=0)
        x = self.learnable_filter(x)
        x = x[..., :self.num_detectors]  # Remove padding


        x_img = x.unsqueeze(1)
        images1 = self.AT(x_img)  

        img_np = images1[0].squeeze().detach().cpu().numpy()
        max_val = img_np.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Imagen con vmin/vmax
        axs[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Tomosipo filtrado vmin/vmax (max={max_val:.4f})")
        axs[0].axis("off")

        # Imagen sin vmin/vmax
        axs[1].imshow(img_np, cmap='gray')
        axs[1].set_title(f"Tomosipo filtrado auto escala (max={max_val:.4f})")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig("imagen_tomosipo_filtrado.png")
        plt.close()

        max_val = x[0].max()
        min_val =  x[0].min()
        plt.imshow(x[0].detach().cpu().numpy(), cmap='gray')
        plt.title(f"Sinograma filtrado vmin/vmax (min = {min_val:.4f}, max={max_val:.4f})")
        plt.savefig("sinograma_filtrado.png")
        plt.close()




        # Interpolation network
        x = x.reshape(-1, 1, self.num_detectors)
        x = self.interpolator_1(x)
        x = self.interpolator_2(x)
        x = self.interpolator_3(x)
        x = self.interpolator_conv(x)
        x = x.view(-1, self.num_angles, self.num_detectors).unsqueeze(1)






        max_val = x[0,0].max()
        min_val =  x[0,0].min()
        plt.imshow(x[0,0].detach().cpu().numpy(), cmap='gray')
        plt.title(f"Sinograma interpolado vmin/vmax (min = {min_val:.4f}, max={max_val:.4f}")
        plt.savefig("sinograma_interpolado.png")
        plt.close()
    
        # backpropagation with tomosipo
        images = self.AT(x)  





        img_np = images[0].squeeze().detach().cpu().numpy()
        max_val = img_np.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Imagen con vmin/vmax
        axs[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Tomosipo vmin/vmax (max={max_val:.4f})")
        axs[0].axis("off")

        # Imagen sin vmin/vmax
        axs[1].imshow(img_np, cmap='gray')
        axs[1].set_title(f"Tomosipo auto escala (max={max_val:.4f})")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig("imagen_tomosipo.png")
        plt.close()






        # Differentiable backprojection
        img = self.AT(x)
        img = img / self.tomosipo_normalizer





        img_np = images[0].squeeze().detach().cpu().numpy()
        max_val = img_np.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Imagen con vmin/vmax
        axs[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Tomosipo normalised vmin/vmax (max={max_val:.4f})")
        axs[0].axis("off")

        # Imagen sin vmin/vmax
        axs[1].imshow(img_np, cmap='gray')
        axs[1].set_title(f"Tomosipo auto escala normalised (max={max_val:.4f})")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig("imagen_tomosipo_normalised.png")
        plt.close()







        # Denoising network
        x = self.denoising_conv_1(img)
        x = self.denoising_conv_2(x)
        x = self.denoising_res_1(x)
        x = self.denoising_res_2(x)
        x = self.denoising_res_3(x)





        img_np = x[0].squeeze().detach().cpu().numpy()
        max_val = img_np.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Imagen con vmin/vmax
        axs[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Denoiser vmin/vmax (max={max_val:.4f})")
        axs[0].axis("off")

        # Imagen sin vmin/vmax
        axs[1].imshow(img_np, cmap='gray')
        axs[1].set_title(f"Denoiser auto escala (max={max_val:.4f})")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig("imagen_denoiser.png")
        plt.close()





        return self.denoising_output(x)
    
    



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

        self.A_deepfbp, self.pg_deepfbp, self.num_angles_deepfbp = self._build_projection_operator()


        self.model = DeepFBPNetwork(self.num_detectors, self.num_angles_deepfbp, self.A_deepfbp, self.filter_type, self.device)


    def _build_projection_operator(self):
        """
        Constructs the appropriate tomosipo projection operator based on view configuration.

        If sparse-view reconstruction is enabled, this method subsamples the available
        projection angles using evenly spaced indices and builds a sparse-view operator.
        Otherwise, it uses the full set of angles and the default projection operator.

        Returns:
            A (ts.Operator): Tomosipo forward projection operator.
            pg (ts.ProjectionGeometry): Associated projection geometry.
            num_angles (int): Number of projection angles used.

        Notes:
            - The sparse-view operator is useful for simulating reduced acquisition scenarios.
            - The selected angles are stored via `self.indices` and reused elsewhere.
            - This method also logs which configuration is applied.

        Example:
            >>> A, pg, n_angles = self._build_projection_operator()
        """
        
        if self.sparse_view:
            self.indices = torch.linspace(0, self.num_angles - 1, steps=self.view_angles).long()
            angles_sparse = self.angles[self.indices]
            pg = ts.cone(angles=angles_sparse, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))
            A = ts.operator(self.vg, pg)
            num_angles = self.view_angles
            self._log(f"[Geometry] Using sparse-view geometry with {num_angles} angles.")
        else:
            pg = self.pg
            A = self.A
            num_angles = self.num_angles
            self._log(f"[Geometry] Using full-view geometry with {num_angles} angles.")
        
        return A, pg, num_angles


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
