import torch
from torch.nn import Module
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functions.dataset import LoDoPaBDataset
from functions.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import json
import random
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt

class ModelBase(Module):
    """
    Base class for training and validating deep learning models.

    This class provides a reusable structure to train models on datasets
    using PyTorch, with methods for training, validation, and metric computation.
    It is intended to be subclassed for specific architectures (e.g., CNNs, NNs, classifiers).

    Attributes:
        training_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset.
        test_path (str): Path to the test dataset.
        model_path (str): Path to save the trained model.
        n_single_BP (int): Number of single backprojections in the dataset.
        i_0 (float): Incident intensity (specific to dataset).
        sigma (float): Noise standard deviation (specific to dataset).
        batch_size (int): Number of samples per batch.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
        debug (bool): Flag to enable debug mode (prints extra info).
        device (torch.device): Computation device (CPU or GPU).
    """

    def __init__(self, training_path, validtion_path, test_path, model_path, n_single_BP, i_0, sigma, batch_size, epochs, optimizer_type, loss_type,learning_rate, seed, debug):

        self.training_path = training_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.model_path = model_path
        self.n_single_BP = n_single_BP
        self.i_0 = i_0
        self.sigma = sigma
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.debug = debug
        self.trained =  False
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.optimizer = None
        self.loss_fn = None

        # set the device once for the whole class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





    @staticmethod
    def compute_psnr(mse, max_val=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        mse (torch.Tensor or float): The mean squared error between reconstructed and reference images.
        max_val (float, optional): The maximum possible pixel value of the images (default: 1.0).

    Returns:
        float: PSNR value in decibels (dB). Returns infinity if MSE is zero.
    """
    # if mse is a tensor, extract the value
    if isinstance(mse, torch.Tensor):
        mse = mse.item()
    
    # if MSE is zero, return infinite PSNR (perfect match)
    if mse == 0:
        return float('inf')
    
    # calculate PSNR using the standard formula
    psnr = 10 * math.log10(max_val ** 2 / mse)

    return psnr





    @staticmethod
    def compute_ssim(reconstructed, reference):
        """
        Computes the Structural Similarity Index (SSIM) between two images.

        Args:
            reconstructed (torch.Tensor): The reconstructed or predicted image tensor with shape [Batch size, Channels, Height, Weight].
            reference (torch.Tensor): The ground truth or reference image tensor with shape [B, C, H, W].

        Returns:
            float: SSIM value between -1 and 1, where 1 means perfect similarity.
        """
        # both inputs must be shape [B, C, H, W]
        return ssim(reconstructed, reference).item()


    def setup_optimizer_and_loss(self, model):
        """Initialize optimizer and loss function based on config strings."""
        if self.optimizer_type == "Adam":
            self.optimizer = Adam(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        if self.loss_type == "MSELoss":
            self.loss_fn = MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")



    def train_one_epoch(self, model, train_dataloader, opt, loss):
        """
        Perform one training epoch over the training dataset.

        Args:
            model (torch.nn.Module): The PyTorch model being trained.
            train_dataloader (DataLoader): DataLoader for the training dataset.
            opt (torch.optim.Optimizer): Optimizer for updating model weights.
            loss_fn (torch.nn.Module): Loss function to compute the training loss.


        Returns:
            float: Total accumulated training loss over the epoch.
        """

        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        # loop over the training set
        for (ground_truth, sinogram, noisy_sinogram, single_back_projections) in tqdm(train_dataloader, desc=f"Training Epoch {e+1}"):

            # send the input to the device
            (ground_truth, single_back_projections) = (ground_truth.to(self.device), single_back_projections.to(self.device))

            # perform a forward pass and calculate the training loss
            pred = model(single_back_projections)
            loss_value = loss(pred, ground_truth)

            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss_value.backward()
            opt.step()

            # add the loss to the total training loss so far
            total_train_loss += loss_value.item()
        
        return total_train_loss





    
    def validation(self, model, val_dataloader, loss):
        """
        Perform validation over the validation dataset.

        Args:
            model (torch.nn.Module): The PyTorch model being evaluated.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            loss_fn (torch.nn.Module): Loss function to compute the validation loss.

        Returns:
            tuple:
                float: Total validation loss.
                float: Accumulated PSNR over the validation set.
                float: Accumulated SSIM over the validation set.
        """

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # initialize metrics
            total_psnr = 0
            total_ssim = 0

            # loop over the validation set
            for (ground_truth, sinogram, noisy_sinogram, single_back_projections) in tqdm(val_dataloader, desc=f"Validation Epoch {e+1}"):

                # send the input to the device
                (ground_truth, single_back_projections) = (ground_truth.to(self.device), single_back_projections.to(self.device))

                # make the predictions and calculate the validation loss
                pred = model(single_back_projections)
                mse_val = loss(pred, ground_truth).item()
                total_val_loss += mse_val

                # compute metrics
                total_psnr += compute_psnr(mse_val)
                total_ssim += compute_ssim(pred, ground_truth)
        
        return total_val_loss, total_psnr, total_ssim





    def training_model(self, model, model_type, optimizer, loss_func):
        """
        Full training loop for the model, including early stopping, metric tracking,
        and learning rate scheduling.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            model_type (str): Description or type of the model (for logging).
            optimizer_class (type): Optimizer class (e.g., torch.optim.Adam).
            loss_class (type): Loss function class (e.g., torch.nn.MSELoss).

        Returns:
            tuple:
                dict: Dictionary containing training history with keys:
                      'train_loss', 'val_loss', 'psnr', 'ssim'.
                torch.nn.Module: The trained model.
        """

        # fix seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # load training and validation datasets
        train_data = LoDoPaBDataset(self.training_path, self.n_single_BP, self.i_0, self.sigma, self.seed, self.debug)
        val_data = LoDoPaBDataset(self.validtion_path, self.n_single_BP, self.i_0, self.sigma, self.seed, self.debug)

        # create dataloader for both and a generator for reproducibility
        g = torch.Generator() 
        g.manual_seed(seed)
        train_dataloader = DataLoader(train_data, self.batch_size, shuffle=True, generator=g, worker_init_fn=lambda _: np.random.seed(self.seed))
        val_dataloader = DataLoader(val_data, self.batch_size) # validation and test do not need shuffle

        # initialize the DBP model
        if debug:
            print(f"[INFO] initializing the {model_type} model...")

        
        model = model.to(device)
        # show model summary
        sample = next(iter(train_dataloader))[3]  # single_back_projections
        summary(model, input_size=sample.size)

        # confirmation for the model to be train 
        confirm = input("Is this the architecture you want to train? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("[INFO] Training aborted.")
            return  

        # initialize  optimizer and loss function
        self.setup_optimizer_and_loss(model)
        opt = self.optimizer
        loss = self.loss_fn

        # initialize early stopping
        early_stopping = EarlyStopping(patience=10, debug=True, path=f'{self.model_path}_best.pth')

        # initialize scheduler for learning rate
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

        # initialize a dictionary to store training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "psnr": [],
            "ssim":[]
        }

        # measure how long training is going to take
        if debug:
            print("[INFO] training the network...")
        start_time = time.time()

        t = tqdm(range(self.epochs), desc="Epochs")

        # loop over our epochs
        for e in t:

            # call the training function
            total_train_loss = self.train_one_epoch(model, train_dataloader, opt, loss, device)

                
            # call the validation function
            total_val_loss, total_psnr, total_ssim = self.validation(model, val_dataloader, loss, device)

            
            # update our training history
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_val_loss = total_val_loss / len(val_dataloader)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            avg_psnr = total_psnr / len(val_dataloader)
            avg_ssim = total_ssim / len(val_dataloader)
            history["psnr"].append(avg_psnr)
            history["ssim"].append(avg_ssim)

            # update the learning rate scheduler
            scheduler.step(avg_train_loss)

            # check ealy stopping
            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop:
                print(f"[INFO] Early stopping stopped at epoch {e+1}")
                break
            else:
                torch.save(model.state_dict(), f'{self.model_path}_final.pth')
            

            # print the model training and validation information
            t.set_postfix({
                "train_loss": f"{avg_train_loss:.6f}",
                "val_loss": f"{avg_val_loss:.6f}"
            })

        # finish measuring how long training took
        end_time = time.time()
        if debug:
            print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

        # Save final metrics
        with open(f'{self.model_path}_metrics.json', 'w') as f:
            json.dump(history, f)

        if debug:
            print(f"[INFO] Metrics saved as '{self.model_path}_metrics.json'")
        
        # change training state to True
        self.trained =  True
        return history, model


        def testing(self, model, loss_func):
            """
            Perform testing on the provided test dataset and report final metrics.

            Args:
                model (torch.nn.Module): The trained PyTorch model to evaluate.
                model_type (str): Description or type of the model (for logging).
                loss_class (type): Loss function class (e.g., torch.nn.MSELoss).

            Returns:
                dict: Dictionary containing average test loss, PSNR, SSIM, ground truth images and reconstructed ones.
            """
        # fix seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load test dataset
        test_data = LoDoPaBDataset(self.test_path, self.n_single_BP, self.i_0, self.sigma, self.seed, self.debug)
        test_dataloader = DataLoader(test_data, self.batch_size)

        # Move model to device
        model = model.to(self.device)

        if self.debug:
            print(f"[INFO] Testing the {model_type} model...")

        # Initialize loss function
        if loss_func == MSE :
            loss = MSELoss()

        # Save ground truth and predictions
        ground_truth = []
        predictions = []

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # initialize metrics
            total_psnr = 0
            total_ssim = 0

            # loop over the validation set
            for (ground_truth, sinogram, noisy_sinogram, single_back_projections) in tqdm(test_dataloader):

                ground_truth.append(ground_truth)

                # send the input to the device
                (ground_truth, single_back_projections) = (ground_truth.to(device), single_back_projections.to(device))

                # make the predictions and calculate the validation loss
                pred = model(single_back_projections)
                mse_val = loss(pred, ground_truth).item()
                total_val_loss += mse_val

                # save predictions
                predictions.append(pred)

                # compute metrics
                total_psnr += compute_psnr(mse_val)
                total_ssim += compute_ssim(pred, ground_truth)
        

        # training history
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_psnr = total_psnr / len(test_dataloader)
        avg_ssim = total_ssim / len(test_dataloader)

        # change tensors to list before save them
        ground_truth.cpu().tolist()
        predictions.cpu().tolist()

        results = {
            "test_loss": avg_test_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "ground_truth": ground_truth,
            "predictions" : predictions
        }

        # save metrics to file
            with open(f'{self.model_path}_test_metrics.json', 'w') as f:
                json.dump(results, f)

        if self.debug:
            print(f"[INFO] Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
            print(f"[INFO] Test metrics saved to '{self.model_path}_test_metrics.json'")

        return results



    @staticmethod
    def show_example(output_img, ground_truth):
        """
        Display side-by-side images of the output and ground truth.

        Args:
            output_img (torch.Tensor): Model output image.
            ground_truth (torch.Tensor): Ground truth image.
        """

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[1].imshow(output_img.squeeze().cpu(), cmap='gray')
        axes[1].set_title('Reconstruction')
        axes[2].imshow(ground_truth.squeeze().cpu(), cmap='gray')
        axes[2].set_title('Ground Truth')
        plt.show()


    def plot_metric(self, x, y, title, label, xlabel, ylabel, test_value=None, save_path=None):
        plt.figure()
        plt.plot(x, y, label=label)
        if test_value is not None:
            plt.axhline(y=test_value, color='red', linestyle='--', label='Test')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            if self.debug:
                print(f"[INFO] Saved plot to {save_path}")
        plt.show()



    def results(self, mode, example_number, save_path=None):
        """
        Display or plot training and/or testing results.

        Args:
            mode (str): 'training', 'testing', or 'both' to choose which results to show.
            example_number (int): Number of random test examples to display.
        """

        if self.trained:
            if mode == training:
               with open(f'{self.model_path}_metrics.json', 'r') as f:
                    history = json.load(f)
                
                # the plot functions need a list of epochs
                epochs = range(1, len(history['train_loss']) + 1)

                #plot Loss
                self.plot_metric(self, epochs, history['train_loss'], 'Loss over Epochs', 'Epoch', 'Val Loss', test_value=None, save_path=save_path)
                

                # plot PSNR
                self.plot_metric(self, epochs, history['psnr'], 'Loss over Epochs', 'Train Loss', 'Val Loss', test_value=None, save_path=save_path)
                plt.figure()
                plt.plot(epochs, history['psnr'], label='Val PSNR')
                plt.xlabel('Epoch')
                plt.ylabel('PSNR (dB)')
                plt.title('Validation PSNR over Epochs')
                plt.legend()
                plt.show()

                # plot SSIM
                plt.figure()
                plt.plot(epochs, history['ssim'], label='Val SSIM')
                plt.xlabel('Epoch')
                plt.ylabel('SSIM')
                plt.title('Validation SSIM over Epochs')
                plt.legend()
                plt.show()


            elif mode == testing:
                with open(f'{self.model_path}_test_metrics.json', 'r') as f:
                    test_results = json.load(f)

                print("\n=== Testing Results ===")
                print(f"Test Loss: {test_results['test_loss']:.6f}")
                print(f"Test PSNR: {test_results['psnr']:.2f} dB")
                print(f"Test SSIM: {test_results['ssim']:.4f}")

                # get random samples
                random.seed(self.seed)
                samples = random.sample(range(0, len(test_results['predictions'])), example_number)

                for example in samples:
                    show_example(test_results['predictions'][example], test_results['ground_truth'][example])


            elif mode == both:
                with open(f'{self.model_path}_metrics.json', 'r') as f:
                    history = json.load(f)
                with open(f'{self.model_path}_test_metrics.json', 'r') as f:
                    test_results = json.load(f)

                # the plot functions need a list of epochs
                epochs = range(1, len(history['train_loss']) + 1)

                # plot Loss
                plt.figure()
                plt.plot(epochs, history['train_loss'], label='Train Loss')
                plt.plot(epochs, history['val_loss'], label='Val Loss')
                plt.axhline(y=test_results['test_loss'], color='red', linestyle='--', label='Test Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss over Epochs')
                plt.legend()
                plt.show()

                # plot PSNR
                plt.figure()
                plt.plot(epochs, history['psnr'], label='Val PSNR')
                plt.axhline(y=test_results['psnr'], color='red', linestyle='--', label='Test PSNR')
                plt.xlabel('Epoch')
                plt.ylabel('PSNR (dB)')
                plt.title('Validation PSNR over Epochs')
                plt.legend()
                plt.show()

                # plot SSIM
                plt.figure()
                plt.plot(epochs, history['ssim'], label='Val SSIM')
                plt.axhline(y=test_results['ssim'], color='red', linestyle='--', label='Test SSIM')
                plt.xlabel('Epoch')
                plt.ylabel('SSIM')
                plt.title('Validation SSIM over Epochs')
                plt.legend()
                plt.show()

                
        else:
            print(f"[WARNING] This model is not trained yet.")





