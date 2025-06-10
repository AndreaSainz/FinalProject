import torch
from torch.nn import Module, ModuleList, Sequential, Conv1d, Conv2d, BatchNorm1d, PReLU, ReLU, Parameter, Hardtanh, BatchNorm2d
from torch.nn.functional import pad
import matplotlib.pyplot as plt
import tomosipo as ts
from tomosipo.torch_support import to_autograd
import json
from .model import ModelBase



class LearnableFilter(Module):
    """
    Implements a learnable frequency-domain filter for sinograms in CT reconstruction.

    This module replaces traditional analytic filters (e.g., Ram-Lak) with learnable weights
    in the frequency domain, optionally per projection angle.

    Args:
        init_filter (torch.Tensor): Initial filter in frequency domain.
        per_angle (bool): If True, uses one filter per angle.
        num_angles (int, optional): Required when per_angle=True.
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
    Depthwise 1D convolutional residual block.

    Used in the angular interpolation subnetwork. Applies a depthwise Conv1d,
    followed by BatchNorm1d and PReLU, with a residual connection.

    Args:
        channels (int): Number of input/output channels.
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


class single_back_projections(Module):
    """
    Applies single-angle differentiable backprojections using Tomosipo.

    Each angle in the sparse set is used to compute a separate backprojection,
    forming a stack of angle-specific reconstructions.

    Args:
        angles_sparse (torch.Tensor): Sparse angles to backproject.
        src_orig_dist (float): Source-to-origin distance.
        num_detectors (int): Number of detector bins.
        vg (ts.VolumeGeometry): Volume geometry for tomosipo operator.
    """
    def __init__(self, angles_sparse, src_orig_dist, num_detectors, vg):

        super().__init__()
        self.angles_sparse = angles_sparse
        self.src_orig_dist = src_orig_dist
        self.num_detectors = num_detectors
        self.vg = vg

        self.tomosipo_geometries = []
        for angle in self.angles_sparse:
            # Define Fan Beam Geometry for each angle
            proj_geom_single = ts.cone(angles= angle, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))

            # Compute Back Projection
            A_single = ts.operator(self.vg, proj_geom_single)

            # make operator diferenciable
            self.AT = to_autograd(A_single.T, is_2d=True, num_extra_dims=2)

            self.tomosipo_geometries.append(self.AT)


    def forward(self, sinogram):
        """
        Generates a set of backprojections from  sparse-view sinogram.

        Each angle is used to generate a single-angle backprojection using tomosipo.

        Args:
            sinogram (torch.Tensor): Noisy sinogram of shape (1, num_angles, num_detectors).

        Returns:
            torch.Tensor: Stack of backprojections of shape (n_single_BP, H, W).
        """

        projections = []
        
        for i, operator in enumerate(self.tomosipo_geometries):
            
            # Extract only the sinogram at this specific angle
            sinogram_angle = sinogram[:, i:i+1, :]

            # Back projection at single angle
            projection = operator(sinogram_angle)

            projections.append(projection)

        # Stack all projections into a single tensor of shape [view_angles, 362, 362]
        single_back_projection = torch.stack(projections).squeeze(1) 

        return single_back_projection


class DBP_block(Module):
    """
    Deep CNN for denoising stacked single-angle backprojections.

    Composed of an initial Conv2d+ReLU, 15 intermediate Conv2d+BN+ReLU layers,
    and a final Conv2d output layer. Adapted from the DBP architecture.

    Args:
        channels (int): Number of input channels (i.e., number of angles).
    """
    def __init__(self, channels):

        # Initialize the base training infrastructure
        super().__init__()
        
        # initial layer
        self.conv1 = self.initial_layer(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)

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
        Builds the initial convolutional block of the network.

        This block consists of:
            - Conv2d
            - ReLU activation

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Convolution stride.
            padding (int): Zero-padding to add to each side.

        Returns:
            nn.Sequential: A sequential block with Conv2d and ReLU.
        """
        initial = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), ReLU(inplace=True))
        return initial



    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds an intermediate convolutional block for the DBP architecture.

        This block includes:
            - Conv2d
            - BatchNorm2d
            - ReLU activation

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Convolution stride.
            padding (int): Zero-padding to apply.

        Returns:
            nn.Sequential: A sequential block with Conv2d, BatchNorm2d, and ReLU.
        """
        convolution = Sequential(Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), BatchNorm2d(out_channels), ReLU(inplace=True))
        return convolution



    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds the final convolutional layer that produces the output image.

        This layer does not include an activation function.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (typically 1).
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Convolution stride.
            padding (int): Zero-padding to apply.

        Returns:
            nn.Conv2d: Final convolutional layer.
        """
        final = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        return final



    def forward(self, x):
        """
        Defines the forward pass of the DBP network.

        The input is passed sequentially through:
            - Initial convolutional block
            - Fifteen intermediate convolutional blocks
            - Final convolutional layer

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where:
                B = batch size,
                C = number of backprojection channels (`n_single_BP`),
                H, W = spatial dimensions.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1, H, W), representing the reconstructed image.
        """
        print(x.shape)
        
        # initial part
        conv1 = self.conv1(x)

        # middle part
        middle = conv1
        for block in self.middle_blocks:
            middle = block(middle)

        #final part
        final_layer = self.final(middle)

        return final_layer
    



class DeepFusionBPNetwork(Module):
    """
    High-level model manager for Deep Fusion Backprojection (DeepFusionBP).

    Handles geometry setup, model instantiation, training configuration, and logging.

    Args:
        model_path (str): Path to save model artifacts.
        filter_type (str): Type of frequency-domain filter ('Filter I' or per-angle).
        view_angles (int): Number of angles in sparse-view mode.
        alpha (float): Log transform scaling for preprocessing.
        i_0 (float): Incident photon count (used in noise simulation).
        sigma (float): Std. dev. for additive Gaussian noise.
        batch_size (int): Training batch size.
        epochs (int): Total training epochs.
        learning_rate (float): Optimizer learning rate.
        debug (bool): Enables verbose/debug plotting if True.
        seed (int): Random seed.
        accelerator (torch.device): Computation device.
        scheduler (str): LR scheduler identifier.
        log_file (str): Path to training log file.

    Example:
        >>> model = DeepFusionBP(
        >>>     model_path="checkpoints/deepfbp",
        >>>     filter_type="Filter I",
        >>>     view_angles=60,
        >>>     alpha=0.001,
        >>>     i_0=1e5,
        >>>     sigma=0.01,
        >>>     batch_size=4,
        >>>     epochs=100,
        >>>     learning_rate=1e-4,
        >>>     debug=True,
        >>>     seed=42,
        >>>     accelerator=torch.device("cuda"),
        >>>     scheduler="cosine",
        >>>     log_file="training.log"
        >>> )
        >>> model.train(train_loader, val_loader)
        >>> model.save_config()
    """
    def __init__(self, angles_sparse, src_orig_dist, num_detectors, num_angles, vg, A, filter_type, device):
        super().__init__()

        self.num_detectors = num_detectors
        self.num_angles = num_angles
        self.device = device
        self.A = A
        self.angles_sparse = angles_sparse
        self.src_orig_dist = src_orig_dist

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

        # Single views backprojection
        self.back_projections = single_back_projections(angles_sparse, src_orig_dist, num_detectors, vg)


        # DBP_model for ct reconstructions
        self.dbp_layer = DBP_block(len(self.angles_sparse))



    def compute_projection_size_padded(self):
        """
        Computes the next power-of-two padding size to avoid aliasing.
        """
        return 2 ** int(torch.ceil(torch.log2(torch.tensor(self.num_detectors, dtype=torch.float32))).item())

    def ram_lak_filter(self, size):
        """
        Generates Ram-Lak filter directly in frequency domain.
        """
        freqs = torch.fft.fftfreq(size)
        ramp = torch.abs(freqs)
        ramp = ramp / ramp.max()
        return torch.abs(freqs)
            
    def forward(self, x):


        print(f" shape inicial {x.shape}")
        max_val = x[0,0].max()
        min_val =  x[0,0].min()
        plt.imshow(x[0,0].detach().cpu().numpy(), cmap='gray')
        plt.title(f"Sinograma inicial vmin/vmax (min = {min_val:.4f}, max={max_val:.4f})")
        plt.savefig("sinograma_inicial.png")
        plt.close()



        # Initial shape: [B, 1, A, D]
        x = x.squeeze(1)  # [B, A, D]
        x = pad(x, (0, self.padding), mode="constant", value=0)


        max_val = x[0].max()
        min_val =  x[0].min()
        plt.imshow(x[0].detach().cpu().numpy(), cmap='gray')
        plt.title(f"Sinograma padding vmin/vmax (min = {min_val:.4f}, max={max_val:.4f})")
        plt.savefig("sinograma_padding.png")
        plt.close()


        x = self.learnable_filter(x)
        x = x[..., :self.num_detectors]  # Remove padding


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
    




        # Differentiable backprojection
        x = x.squeeze(1) #needs shape [B,A,D]
        projections = self.back_projections(x)





        img_np = projections[0,0].squeeze().detach().cpu().numpy()
        max_val = img_np.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Imagen con vmin/vmax
        axs[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Una projeccion Tomosipo vmin/vmax (max={max_val:.4f})")
        axs[0].axis("off")

        # Imagen sin vmin/vmax
        axs[1].imshow(img_np, cmap='gray')
        axs[1].set_title(f"Una projeccion Tomosipo auto escala (max={max_val:.4f})")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig("imagen_projecion.png")
        plt.close()



        #reconstruction with DBP model
        img = self.dbp_layer(projections)



        img_np = img[0].squeeze().detach().cpu().numpy()
        max_val = img_np.max()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Imagen con vmin/vmax
        axs[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Reconstruccion vmin/vmax (max={max_val:.4f})")
        axs[0].axis("off")

        # Imagen sin vmin/vmax
        axs[1].imshow(img_np, cmap='gray')
        axs[1].set_title(f"Reconstruccion auto escala (max={max_val:.4f})")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig("imagen_reconstruccion.png")
        plt.close()


        return img
    
    



class DeepFusionBP(ModelBase):
    """
    High-level wrapper for training and managing the DeepFusionBP model.

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

    def __init__(self, model_path, filter_type, view_angles, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed,accelerator, scheduler, log_file):
        super().__init__(model_path, "DeepFusionBP", False, 1, True, view_angles, alpha, i_0, sigma, batch_size, epochs, "AdamW", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        self.filter_type = filter_type

        self._log(f"[Geometry] Using sparse-view geometry with {view_angles} angles.")
        self.indices = torch.linspace(0, self.num_angles - 1, steps=self.view_angles).long()
        self.angles_sparse = self.angles[self.indices]
        self.pg = ts.cone(angles=self.angles_sparse, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))
        self.A = ts.operator(self.vg, self.pg)
        self.num_angles = self.view_angles


        self.model = DeepFusionBPNetwork(self.angles_sparse, self.src_orig_dist, self.num_detectors, self.view_angles, self.vg, self.A, self.filter_type, self.device)


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

        self._log(f"[DeepFusionBP] Configuration saved to: {self.model_path}_config.json")