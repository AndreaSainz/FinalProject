from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torch.nn import Sequential
import json
from ..models.model import ModelBase





class DBP(ModelBase):
    """
    Deep Backprojection (DBP) network for sparse-view CT image reconstruction.

    This model reconstructs high-quality CT images from multiple single-angle
    backprojections using a deep convolutional neural network. It extends
    `ModelBase`, inheriting functionality for training, evaluation, and configuration
    management.

    The architecture is composed of:
        - One initial convolutional block (Conv2d + ReLU).
        - Fifteen intermediate convolutional blocks (Conv2d + BatchNorm2d + ReLU).
        - One final convolutional layer without activation.

    Args:
        model_type (str): Identifier string, set to "DBP".
        model_path (str): Directory path to save model checkpoints and logs.
        n_single_BP (int): Number of single-angle backprojections per sample.
        alpha (float): Scaling factor applied to log-transformed sinograms.
        i_0 (float): Incident X-ray intensity for simulating Poisson noise.
        sigma (float): Standard deviation of added Gaussian noise.
        batch_size (int): Number of samples per training batch.
        epochs (int): Total number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        debug (bool): If True, enables verbose logging and debugging output.
        seed (int): Random seed for reproducibility.
        accelerator (torch.device): Computation device (e.g., `torch.device("cuda")`).
        scheduler (str): Learning rate scheduler name (e.g., "ReduceLROnPlateau").
        log_file (str): Path to the log file for recording training progress.

    Attributes:
        conv1 (nn.Sequential): Initial convolutional block.
        middle_blocks (nn.ModuleList): Intermediate convolutional layers.
        final (nn.Conv2d): Final convolution layer producing single-channel output.
        model (nn.Sequential): Entire model as a sequential block.

    Example:
        >>> import torch
        >>> from ct_reconstruction.models import DBP
        >>> model = DBP(
        ...     model_path="models/dbp",
        ...     n_single_BP=10,
        ...     alpha=0.001,
        ...     i_0=1e5,
        ...     sigma=0.01,
        ...     batch_size=4,
        ...     epochs=50,
        ...     learning_rate=1e-3,
        ...     debug=False,
        ...     seed=42,
        ...     accelerator=torch.device("cuda"),
        ...     scheduler="ReduceLROnPlateau",
        ...     log_file="logs/dbp_training.log"
        ... )
        >>> x = torch.randn(4, 10, 128, 128)  # input tensor
        >>> output = model(x)  # forward pass
        >>> model.save_config()  # save model configuration
    """


    def __init__(self, model_path, n_single_BP, alpha, i_0, sigma, batch_size, epochs, learning_rate, debug, seed, accelerator, scheduler, log_file):

        # Initialize the base training infrastructure
        super().__init__(model_path, "DBP", True, n_single_BP, False, 0, alpha, i_0, sigma, batch_size, epochs, "Adam", "MSELoss", learning_rate, debug, seed, accelerator, scheduler, log_file)
        

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
        Saves the model configuration and hyperparameters to a JSON file.

        The configuration file is saved at `{model_path}_config.json` and includes
        details such as model type, training settings, device, and logging information.
        """
        config = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "accelerator" : str(self.accelerator.device),
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
        #open file in writting mode
        with open(f"{self.model_path}_config.json", "w") as f:
            json.dump(config, f, indent=4)

        self._log(f"[DBP] Configuration saved to: {self.model_path}_config.json")