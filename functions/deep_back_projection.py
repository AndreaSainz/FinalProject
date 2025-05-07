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
from torchmetrics.functional import structural_similarity_index_measure as ssim
from functions.dataset import LoDoPaBDataset
import time
from tqdm import tqdm




class DBP(Module):
    """
    Deep Backprojection (DBP) network for CT reconstruction.
    
    Args:
        in_channels (int): Number of input channels.

    Architecture:
        - Initial Conv2d + ReLU layer.
        - 15 repeated Conv2d + BatchNorm2d + ReLU blocks.
        - Final Conv2d layer producing single-channel output.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Initial layer
        self.conv1 = self.initial_layer(in_channels = in_channels , out_channels = 64, kernel_size = 3, stride= 1, padding = 1)

        # 15 layer, all equals in dimensions (we need to define each so they have different weights)
        self.middle_blocks = ModuleList([
            self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1) 
            for _ in range(15)])
        
        #Final layer
        self.final = self.final_layer(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)


    def initial_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates the initial convolutional layer with ReLU activation.
        
        Returns:
            Sequential: Sequential model with Conv2d + ReLU.
        """

       initial = Sequential(
                    Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    ReLU(inplace=True))
       return initial

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a convolutional block with Conv2d, BatchNorm2d, and ReLU.
        
        Returns:
            Sequential: Sequential model with Conv2d + BatchNorm2d + ReLU.
        """

       convolution = Sequential(
                    Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    BatchNorm2d(out_channels),
                    ReLU(inplace=True))
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



# metric functions
def compute_psnr(reconstructed, reference, max_val=1.0):
    mse = torch.mean((reconstructed - reference) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.item()

def compute_ssim(reconstructed, reference):
    # both inputs must be shape [N, C, H, W]
    return ssim(reconstructed, reference).item()

    
#training functions
def training_dbp(in_channels, training_path, validtion_path, model_path, n_single_BP, i_0 , sigma, seed, debug, batch_size, epochs, learning_rate):
    """
    Trains the DBP model on provided training and validation datasets.

    Args:
        in_channels (int): Number of input channels.
        training_path (str): Path to training dataset.
        validtion_path (str): Path to validation dataset.
        model_path (str): Path to save the trained model.
        n_single_BP (int): Number of single backprojections.
        i_0 (float): Incident intensity (for dataset).
        sigma (float): Noise standard deviation (for dataset).
        seed (int): Random seed.
        debug (bool): Whether to print debug info.
        batch_size (int): Batch size for DataLoader.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for optimizer.

    Saves:
        - Trained model state dictionary.
    """

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load training and validation datasets
    train_data = LoDoPaBDataset(training_path, n_single_BP, i_0, sigma, seed, debug)
    val_data = LoDoPaBDataset(validtion_path, n_single_BP, i_0, sigma, seed, debug)

    # create dataloader for both
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, seed = seed)
    val_dataloader = DataLoader(val_data, batch_size) # validation and test do not need shuffle

    # calculate steps per epoch for training and validation set
    train_steps = len(train_dataloader.dataset) // batch_size
    val_steps = len(val_dataloader.dataset) // batch_size

    # initialize the DBP model
    if debug:
        print("[INFO] initializing the DBP model...")

    model = DBP(in_channels=in_channels).to(device)

    # initialize Adam optimizer and MSE loss function
    opt = Adam(model.parameters(), lr=learning_rate)
    loss = MSELoss()

    # initialize a dictionary to store training history
    losses = {
        "train_loss": [],
        "val_loss": []
    }

    # measure how long training is going to take
    if debug:
        print("[INFO] training the network...")
    start_time = time.time()

    t = tqdm(range(epochs), desc="Epochs")

    # loop over our epochs
    for e in t:

        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        # loop over the training set
        for (ground_truth, sinogram, noisy_sinogram, single_back_projections) in tqdm(train_dataloader, desc=f"Training Epoch {e+1}"):

            # send the input to the device
            (ground_truth, single_back_projections) = (ground_truth.to(device), single_back_projections.to(device))

            # perform a forward pass and calculate the training loss
            pred = model(single_back_projections)
            los = loss(pred, ground_truth)

            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            total_train_loss += los.item()
            


        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # loop over the validation set
            for (ground_truth, sinogram, noisy_sinogram, single_back_projections) in tqdm(val_dataloader, desc=f"Validation Epoch {e+1}"):

                # send the input to the device
                (ground_truth, single_back_projections) = (ground_truth.to(device), single_back_projections.to(device))

                # make the predictions and calculate the validation loss
                pred = model(single_back_projections)
                total_val_loss += loss(pred, ground_truth)

        # update our training history
        avg_train_loss = total_train_loss / len(train_dataloader)
        losses["train_loss"].append(total_train_loss.cpu().detach().numpy())
        losses["val_loss"].append(total_val_loss.cpu().detach().numpy())
    
        # print the model training and validation information
        t.set_postfix({
            "train_loss": f"{total_train_loss:.6f}",
            "val_loss": f"{total_val_loss:.6f}"
        })

    # finish measuring how long training took
    end_time = time.time()
    if debug:
        print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

    # Save the final model
    torch.save(model.state_dict(), f'{model_path}.pth')
    if debug:
        print(f"[INFO] Model saved as '{model_path}.pth'")


    


