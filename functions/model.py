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
import logging
import math
from pytorch_msssim import ssim
from accelerate import Accelerator

class ModelBase(Module):
    """
    Base class for training and validating deep learning models.

    This class provides a reusable structure to train models on datasets
    using PyTorch, with methods for training, validation, testing, metric computation and ploting.
    It is intended to be subclassed for specific architectures (e.g., CNNs, NNs, classifiers).

    Attributes:
        training_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset.
        test_path (str): Path to the test dataset.
        model_path (str): Path to save trained models and logs.
        model_type (str): Name or identifier of the model architecture.
        n_single_BP (int): Number of single backprojections used per sample.
        i_0 (float): Incident X-ray intensity (used for noise modeling).
        sigma (float): Standard deviation of the noise.
        batch_size (int): Number of samples per training batch.
        epochs (int): Number of full passes through the training dataset.
        optimizer_type (str): Optimizer name (e.g., 'Adam').
        loss_type (str): Loss function name (e.g., 'MSELoss').
        learning_rate (float): Learning rate for optimizer.
        seed (int): Random seed for reproducibility.
        debug (bool): Whether to print debug information.
    """

    def __init__(self, training_path, validation_path, test_path, model_path, model_type, n_single_BP, alpha, i_0, sigma, batch_size, epochs, optimizer_type, loss_type,learning_rate, debug, seed, log_file='training.log'):
        super().__init__()
        self.training_path = training_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.model_path = model_path
        self.model_type = model_type
        self.n_single_BP = n_single_BP
        self.alpha = alpha
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

        # accelerator for faster code
        self.accelerator = Accelerator()

        # logger configuration
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(
                level=logging.INFO, 
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ],
                format='%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # set the device once for the whole class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.accelerator.device



    def _set_seed(self):
        """
        Sets the random seed for reproducibility across numpy, Python, and PyTorch (CPU & GPU).
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    def _log(self, msg, level='info'):

        if self.debug and hasattr(self.logger, level):
            getattr(self.logger, level)(msg)




    def _get_dataloaders(self):
        """
        Loads the training and validation datasets and returns the corresponding DataLoaders.

        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        is_gcs = self.training_path.startswith("gs://")

        # load training and validation datasets
        train_data = LoDoPaBDataset(self.training_path, self.n_single_BP, self.alpha, self.i_0, self.sigma, self.seed, False)
        val_data = LoDoPaBDataset(self.validation_path, self.n_single_BP, self.alpha,  self.i_0, self.sigma, self.seed, False)

        # create dataloader for both and a generator for reproducibility
        g = torch.Generator() 
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 if is_gcs else 4,
            pin_memory=True,
            persistent_workers=not is_gcs,
            generator=g,
            worker_init_fn=lambda _: np.random.seed(self.seed)
        )

        val_dataloader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False, # validation and test do not need shuffle
            num_workers=0 if is_gcs else 4,
            pin_memory=True,
            persistent_workers=not is_gcs,
        ) 

        return (train_dataloader, val_dataloader)
            




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



    def train_one_epoch(self, model, train_dataloader, opt, loss, e):
        """
        Runs a single epoch of training on the model.

        Args:
            model (torch.nn.Module): Model to be trained.
            train_dataloader (DataLoader): Dataloader for training set.
            opt (torch.optim.Optimizer): Optimizer instance.
            loss (nn.Module): Loss function.
            e (int): Epoch index for logging.

        Returns:
            float: Total training loss for the epoch.
        """

        # set the model in training mode
        model.train()

        # initialize the total training loss
        total_train_loss = 0

        # loop over the training set
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {e+1}"):

            # send the input to the device
            ground_truth = batch["ground_truth"]
            single_back_projections = batch["single_back_projections"]

            # perform a forward pass and calculate the training loss
            pred = model(single_back_projections)
            loss_value = loss(pred, ground_truth)

            # checking that predictions and ground truth images have same shape
            assert ground_truth.shape == pred.shape, f"[ERROR] Shape mismatch: predicted {pred.shape}, ground truth {ground_truth.shape}"

            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            self.accelerator.backward(loss_value)
            opt.step()

            # add the loss to the total training loss so far
            total_train_loss += loss_value.item()
        
        return total_train_loss





    
    def validation(self, model, val_dataloader, loss, e):
        """
        Evaluates the model on the validation set and computes PSNR, SSIM, and loss.

        Args:
            model (torch.nn.Module): Model to evaluate.
            val_dataloader (DataLoader): Validation data loader.
            loss (nn.Module): Loss function.
            e (int): Epoch index for logging.

        Returns:
            tuple: Total validation loss, accumulated PSNR, accumulated SSIM.
        """

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # initialize validation metrics
            total_val_loss = 0
            total_psnr = 0
            total_ssim = 0

            # loop over the validation set
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {e+1}"):

                # send the input to the device
                ground_truth = batch["ground_truth"]
                single_back_projections = batch["single_back_projections"]
            
                # make the predictions and calculate the validation loss
                pred = model(single_back_projections)
                mse_val = loss(pred, ground_truth).item()
                total_val_loss += mse_val

                # checking that predictions and ground truth images have same shape
                assert ground_truth.shape == pred.shape, f"[ERROR] Shape mismatch: predicted {pred.shape}, ground truth {ground_truth.shape}"

                # compute metrics
                total_psnr += self.compute_psnr(mse_val, self.alpha)
                total_ssim += self.compute_ssim(pred, ground_truth)
        
        return total_val_loss, total_psnr, total_ssim

        

    def training_model(self, model, patience, confirm_train=False):
        """
        Full training loop for the model, including early stopping, metric tracking,
        and learning rate scheduling.

        Args:
            model (torch.nn.Module): Model to train.
            patience (int): Number of epochs to wait without improvement before stopping.
            confirm_train (bool): Ask for manual confirmation before training (default True).

        Returns:
            tuple: Training history dictionary and trained model.
        """

        # fix seed
        self._set_seed()

    
        # load training and validation datasets
        train_dataloader, val_dataloader = self._get_dataloaders()

        
        # show model summary
        model.to(self.device)
        sample = next(iter(train_dataloader))["single_back_projections"]  # single_back_projections
        summary(model, input_size=tuple(sample.shape[1:]))

        # confirmation for the model to be train 
        if confirm_train:
            confirm = input("Is this the architecture you want to train? (yes/no): ")
            if confirm.strip().lower() != "yes":
                self._log("Training aborted.")
                return  

        # initialize  optimizer and loss function
        self.setup_optimizer_and_loss(model)
        loss = self.loss_fn

        # accelerates training
        model, opt, train_dataloader, val_dataloader = self.accelerator.prepare(model, self.optimizer, train_dataloader, val_dataloader)
        


        # initialize early stopping
        early_stopping = EarlyStopping(patience=patience, debug=self.debug, path=f'{self.model_path}_best.pth',  logger=self.logger)

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
        self._log("Training the network...")
        start_time = time.time()

        t = tqdm(range(self.epochs), desc="Epochs")

        # loop over our epochs
        for e in t:

            # call the training function
            total_train_loss = self.train_one_epoch(model, train_dataloader, opt, loss, e)

                
            # call the validation function
            total_val_loss, total_psnr, total_ssim = self.validation(model, val_dataloader, loss, e)

            
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
                self.trained = True
                self._log(f"Early stopping stopped at epoch {e+1}")
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
        self._log("Total time taken to train the model: {:.2f}s".format(end_time - start_time))


        # Save final metrics
        with open(f'{self.model_path}_metrics.json', 'w') as f:
            json.dump(history, f)

        self._log(f"Metrics saved as '{self.model_path}_metrics.json'")
        

        return history, model


    def testing(self, model):
        """
        Perform testing on the provided test dataset and report final metrics.

        Args:
            model (torch.nn.Module): The trained PyTorch model to evaluate.

        Returns:
            dict: Dictionary containing average test loss, PSNR, SSIM, ground truth images and reconstructed ones.
        """
        # fix seed for reproducibility
        self._set_seed()

        # Load test dataset
        test_data = LoDoPaBDataset(self.test_path, self.n_single_BP, self.alpha,  self.i_0, self.sigma, self.seed, False)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        # save ground truth and predictions
        gt_images = []
        predictions = []

        
        self._log(f"Testing the {self.model_type} model...")

        # initialize  optimizer and loss function
        self.setup_optimizer_and_loss(model)
        model, loss, test_dataloader = self.accelerator.prepare(model, self.loss_fn, test_dataloader)


        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # initialize metrics
            total_test_loss = 0
            total_psnr = 0
            total_ssim = 0

            # loop over the validation set
            for batch in tqdm(test_dataloader):
                # send the input to the device
                ground_truth = batch['ground_truth']
                single_back_projections = batch['single_back_projections']

                # make the predictions and calculate the validation loss
                pred = model(single_back_projections)
                mse_val = loss(pred, ground_truth).item()
                total_test_loss += mse_val

                # compute metrics
                total_psnr += self.compute_psnr(mse_val, self.alpha)
                total_ssim += self.compute_ssim(pred, ground_truth)

                # save gound truth and prediction
                predictions.append(pred.cpu())
                gt_images.append(ground_truth.cpu())
                
        

        # training history
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_psnr = total_psnr / len(test_dataloader)
        avg_ssim = total_ssim / len(test_dataloader)


        results = {
            "test_loss": avg_test_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim
        }

        # save predictions and ground truth
        torch.save(torch.cat(predictions), f"{self.model_path}_predictions_images.pt")
        torch.save(torch.cat(gt_images), f"{self.model_path}_ground_truth_images.pt")
        self._log(f"Predictions saved to '{self.model_path}_predictions_images.pt")
        self._log(f"Predictions saved to '{self.model_path}_ground_truth_images.pt")

        # save metrics to file
        with open(f'{self.model_path}_test_metrics.json', 'w') as f:
            json.dump(results, f)

        self._log(f"Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        self._log(f"Test metrics saved to '{self.model_path}_test_metrics.json'")

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
        axes[0].imshow(output_img.squeeze().cpu(), cmap='gray')
        axes[0].set_title('Reconstruction')
        axes[1].imshow(ground_truth.squeeze().cpu(), cmap='gray')
        axes[1].set_title('Ground Truth')
        plt.show()


    def plot_metric(self, x, y_dict, title, xlabel, ylabel, test_value=None, save_path=None):
        """
        Plots metrics over epochs with optional test reference line.

        Args:
            x (list or range): X-axis values (e.g., epochs).
            y_dict (dict): Dictionary with keys as labels and values as lists of y-values.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            test_value (float, optional): Value to draw a horizontal reference line.
            save_path (str, optional): If provided, saves the plot to this path.
        """
        plt.figure()
        for label, y in y_dict.items():
            plt.plot(x, y, label=label)

        if test_value is not None:
            plt.axhline(y=test_value, color='red', linestyle='--', label='Test')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            self._log(f"Saved plot to {save_path}")
            
        plt.show()



    def results(self, mode, example_number, save_path=None):
        """
        Display or plot training and/or testing results.

        Args:
            mode (str): 'training', 'testing', or 'both' to choose which results to show.
            example_number (int): Number of random test examples to display.
        """

        if self.trained:

            
            if mode == "training":
                # handling file path error
                if not os.path.exists(f"{self.model_path}_metrics.json"):
                    self._log(f"File not found: {self.model_path}_metrics.json", level='error')
                    return None
                else:
                    with open(f'{self.model_path}_metrics.json', 'r') as f:
                        history = json.load(f)
                
                # the plot functions need a list of epochs
                epochs = range(1, len(history['train_loss']) + 1)

                #plot Loss
                loss_dict = {'Train Loss': history['train_loss'], 'Val Loss': history['val_loss']}
                self.plot_metric(epochs,loss_dict, title='Loss over Epochs', xlabel='Epoch', ylabel='Loss', test_value=None, save_path=save_path )
                

                # plot PSNR
                psnr_dict = {'PSNR metric':  history['psnr']}
                self.plot_metric(epochs, psnr_dict, 'Validation PSNR over Epochs', 'Epoch', 'PSNR (dB)', test_value=None, save_path=save_path)
                

                # plot SSIM
                ssim_dict = {'SSIM metric':  history['ssim']}
                self.plot_metric(epochs, ssim_dict, 'Validation SSIM over Epochs', 'Epoch', 'SSIM', test_value=None, save_path=save_path)
            


            elif mode == "testing":
                # handling file path error
                if not os.path.exists(f"{self.model_path}_test_metrics.json"):
                    self._log(f"File not found: {self.model_path}_test_metrics.json", level='error')
                    return None
                elif not os.path.exists(f"{self.model_path}_predictions_images.pt") or not os.path.exists(f"{self.model_path}_ground_truth_images.pt"):
                    self._log("Prediction or ground truth .pt files not found", level="error")
                    return

                else:
                    with open(f'{self.model_path}_test_metrics.json', 'r') as f:
                        test_results = json.load(f)
                    predictions = torch.load(f"{self.model_path}_predictions_images.pt")
                    ground_truths = torch.load(f"{self.model_path}_ground_truth_images.pt")

                #plot results
                print("\n=== Testing Results ===")
                print(f"Test Loss: {test_results['test_loss']:.6f}")
                print(f"Test PSNR: {test_results['psnr']:.2f} dB")
                print(f"Test SSIM: {test_results['ssim']:.4f}")

                # get random samples
                random.seed(self.seed)
                samples = random.sample(range(0, len(predictions)), example_number)

                for example in samples:
                    self.show_example(predictions[example], ground_truths[example])


            elif mode == "both":
                # handling file path error
                if not os.path.exists(f"{self.model_path}_test_metrics.json"):
                    
                    self._log(f"File not found: {self.model_path}_test_metrics.json", level='error')
                    return None

                elif not os.path.exists(f"{self.model_path}_metrics.json"):
                    self._log(f"File not found: {self.model_path}_metrics.json", level='error')
                    return None

                else:
                    with open(f'{self.model_path}_test_metrics.json', 'r') as f:
                        test_results = json.load(f)
                    with open(f'{self.model_path}_metrics.json', 'r') as f:
                        history = json.load(f)


                # the plot functions need a list of epochs
                epochs = range(1, len(history['train_loss']) + 1)

                # plot Loss
                loss_dict = {'Train Loss': history['train_loss'], 'Val Loss': history['val_loss']}
                self.plot_metric(epochs,loss_dict, title='Loss over Epochs', xlabel='Epoch', ylabel='Loss', test_value=test_results['test_loss'], save_path=save_path )
                

                # plot PSNR
                psnr_dict = {'Val PSNR': history['psnr']}
                self.plot_metric(epochs, psnr_dict, title='PSNR over Epochs', xlabel='Epoch', ylabel='PSNR (dB)', test_value=test_results['psnr'], save_path=save_path )
                

                # plot SSIM
                ssim_dict = {'Val SSIM': history['ssim']}
                self.plot_metric(epochs, ssim_dict, title='SSIM over Epochs', xlabel='Epoch', ylabel='SSIM', test_value=test_results['ssim'], save_path=save_path )

                
        else:
            self._log(f"This model is not trained yet.",  level='warning')