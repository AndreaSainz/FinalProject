import tomosipo as ts
from ts_algorithms import fbp, fdk, sirt, em, tv_min2d, nag_ls
import torch
from torch.nn import MSELoss, L1Loss, Module
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..datasets.dataset import LoDoPaBDataset
from ..callbacks.early_stopping import EarlyStopping
from ..utils.metrics import compute_psnr, compute_ssim, compute_psnr_results
from ..utils.plotting import show_example, plot_metric, plot_different_reconstructions, show_example_epoch
from ..utils.loggers import configure_logger
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import json
import random
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import logging
from accelerate import Accelerator
import os
import gc
import pandas as pd


class ModelBase(Module):
    """
    Abstract base class for CT reconstruction model training, evaluation, and analysis.

    This class provides a unified framework for:
    - Setting up volume and projection geometries using Tomosipo package.
    - Loading CT datasets and preprocessing them.
    - Training deep learning models with optional early stopping and scheduler.
    - Evaluating model performance using PSNR, SSIM, and MSE loss.
    - Visualizing metrics and reconstructed outputs.
    - Running classical CT reconstruction algorithms (e.g., FBP, SIRT, TV-Min).

    This class is intended to be subclassed with a specific model defined in `self.model`.

    Attributes:
        model_path (str): Path prefix to save checkpoints, logs, and metrics.
        model_type (str): Name/identifier for the model architecture (used for logging and plotting).
        n_single_BP (int): Number of single backprojections per sample.
        alpha (float): Normalization factor for PSNR computation.
        i_0 (float): Incident X-ray intensity used in noise modeling.
        sigma (float): Standard deviation of Gaussian noise applied to sinograms.
        batch_size (int): Batch size for training and evaluation.
        epochs (int): Maximum number of training epochs.
        optimizer_type (str): Optimizer type, e.g., "Adam", "AdamW".
        loss_type (str): Loss function type, e.g., "MSELoss".
        learning_rate (float): Learning rate for optimizer.
        debug (bool): Enables verbose logging.
        seed (int): Random seed for full reproducibility.
        scheduler (bool): Enables learning rate scheduling with ReduceLROnPlateau.
    """

    def __init__(self, model_path, model_type, single_bp , n_single_BP, alpha, i_0, sigma, batch_size, epochs, optimizer_type, loss_type, learning_rate, debug, seed, scheduler = True, log_file='training.log'):
        super().__init__()

        #Scan parameters from the paper and data
        self.pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.num_angles = 1000
        self.num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
        self.src_orig_dist = 575
        self.src_det_dist = 1050

        # Create tomosipo volume and projection geometry
        self.vg = ts.volume(shape=(1,self.pixels,self.pixels))                                                       # Volumen
        self.angles = np.linspace(0, np.pi, self.num_angles, endpoint=True)                                          # Angles
        self.pg = ts.cone(angles = self.angles, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))     # Fan beam structure
        self.A = ts.operator(self.vg,self.pg)                                                                        # Operator
                                                                              
        # dataset parameters
        self.model = None
        self.training_path = None
        self.validation_path = None
        self.test_path = None
        self.single_bp = single_bp
        self.n_single_BP = n_single_BP
        self.alpha = alpha
        self.i_0 = i_0
        self.sigma = sigma
        self.max_len_train = None
        self.max_len_val = None
        self.max_len_test = None

        # model parameters
        self.model_path = model_path
        self.model_type = model_type
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
        self.scheduler = scheduler

        # accelerator for faster code
        self.accelerator = Accelerator()
        self.trainable_params = None

        # logger configuration
        self.logger = configure_logger("ct_reconstruction.models.model", log_file, debug=self.debug)
        self.dataset_logger = configure_logger("ct_reconstruction.dataset", log_file, debug=False)
        
        # set the device once for the whole class
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
        """
        Logs a message to the logger with the specified level.

        Always logs to the file, and optionally prints to console if self.debug is True.
        Uses the logging level (e.g., 'info', 'warning', 'error') to determine severity.

        Args:
            msg (str): The message to log.
            level (str): Logging level (default is 'info'). Can be 'debug', 'info', 'warning', 'error', or 'critical'.
        """
        # Get the logging method based on the provided level (defaults to .info)
        level_func = getattr(self.logger, level.lower(), self.logger.info)
        # Call the logging method with the message
        level_func(msg)



    def _get_dataloaders(self):
        """
        Loads training and validation datasets and creates corresponding DataLoaders.

        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        # looking if we are using data from google storage
        is_gcs = self.training_path.startswith("gs://")

        
        # load training and validation datasets
        train_data = LoDoPaBDataset(self.training_path,
            self.vg,
            self.angles,
            self.pg,
            self.A,
            self.single_bp,
            self.n_single_BP,
            self.alpha,
            self.i_0,
            self.sigma,
            self.seed,
            self.max_len_train, 
            False,
            self.dataset_logger)

        val_data = LoDoPaBDataset(self.validation_path, 
            self.vg, 
            self.angles, 
            self.pg, 
            self.A, 
            self.single_bp,
            self.n_single_BP, 
            self.alpha,  
            self.i_0, 
            self.sigma, 
            self.seed, 
            self.max_len_val, 
            False, 
            self.dataset_logger)

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
            



    def setup_optimizer_and_loss(self):
        """
        Initializes the optimizer and loss function based on configuration.
        
        Raises:
            ValueError: If the optimizer or loss type is unsupported.
        """
        # get trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        #optimiser
        if self.optimizer_type == "Adam":
            self.optimizer = Adam(trainable_params, lr=self.learning_rate)

        elif self.optimizer_type == "AdamW":
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Loss function
        if self.loss_type == "MSELoss":
            self.loss_fn = MSELoss()
        elif self.loss_type == "L1Loss":
            self.loss_fn = L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")



    def train_one_epoch(self, train_dataloader, opt, loss, e, save_path, show_examples, number_of_examples):
        """
        Trains the model for a single epoch.

        Args:
            train_dataloader (DataLoader): Dataloader for training data.
            opt (torch.optim.Optimizer): Optimizer used for weight updates.
            loss (nn.Module): Loss function.
            e (int): Epoch index (for logging or debugging).

        Returns:
            float: Total accumulated loss over the epoch.
        """
        # set the model in training mode
        self.model.train()

        # initialize the total training loss
        total_train_loss = 0

        # loop over the training set
        for batch_idx, batch in enumerate(train_dataloader):

            # send the input to the device
            ground_truth = batch["ground_truth"]
            if self.single_bp:
                input_data = batch["single_back_projections"]
            else:
                input_data = batch["noisy_sinogram"]

            # perform a forward pass and calculate the training loss
            pred = self.model(input_data)
            loss_value = loss(pred, ground_truth)

            if show_examples:
                if batch_idx == 0:
                    for i in range(min(number_of_examples, pred.shape[0])):
                        show_example_epoch(pred[i], ground_truth[i], e, f"{save_path}_epochs_{i}")
                        print(f"[INFO] Saved example figure: {save_path}_epochs_{i}_{e}.png")

            # checking that predictions and ground truth images have same shape
            assert ground_truth.shape == pred.shape, f"[ERROR] Shape mismatch: predicted {pred.shape}, ground truth {ground_truth.shape}"

            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            self.accelerator.backward(loss_value)
            opt.step()

            # add the loss to the total training loss so far
            total_train_loss += loss_value.item()
        
        return total_train_loss


    def validation(self, val_dataloader, loss, mse_fn, e):
        """
        Evaluates the model on the validation dataset.

        Args:
            val_dataloader (DataLoader): Dataloader for validation data.
            loss (nn.Module): Primary loss function used for training.
            mse_fn (nn.Module or None): Optional MSELoss for PSNR calculation.
            e (int): Epoch index (used in progress bar).

        Returns:
            tuple: (total_val_loss, total_psnr, total_ssim)
        """

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            self.model.eval()

            # initialize validation metrics
            total_val_loss = 0
            total_psnr = 0
            total_ssim = 0

            # loop over the validation set
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {e+1}", leave=False):

                # send the input to the device
                ground_truth = batch["ground_truth"]
                if self.single_bp:
                    input_data = batch["single_back_projections"]
                else:
                    input_data = batch["noisy_sinogram"]
            
                # make the predictions and calculate the validation loss
                pred = self.model(input_data)
                loss_value = loss(pred, ground_truth).item()
                total_val_loss += loss_value

                # checking that predictions and ground truth images have same shape
                assert ground_truth.shape == pred.shape, f"[ERROR] Shape mismatch: predicted {pred.shape}, ground truth {ground_truth.shape}"

                # compute metrics
                if self.loss_type == "MSELoss" or mse_fn is None:
                    mse_val = loss_value
                else:
                    mse_val = mse_fn(pred, ground_truth).item()

                total_psnr += compute_psnr(mse_val, self.alpha)
                total_ssim += compute_ssim(pred, ground_truth, self.alpha)
        
        return total_val_loss, total_psnr, total_ssim

        

    def train(self, training_path, validation_path, save_path, max_len_train = None, max_len_val=None, patience=10, confirm_train=False, show_examples=True, number_of_examples=1):
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
        #changing paths parameters
        self.training_path = training_path
        self.validation_path = validation_path
        self.max_len_train = max_len_train
        self.max_len_val = max_len_val

        # fix seed
        self._set_seed()

    
        # load training and validation datasets
        train_dataloader, val_dataloader = self._get_dataloaders()

    
        # show model summary
        self.model.to(self.device)
        if self.single_bp:
            sample = next(iter(train_dataloader))["single_back_projections"]  # single_back_projections
        else:
            sample = next(iter(train_dataloader))["noisy_sinogram"]
        summary(self.model, input_size=tuple(sample.shape[1:]))

        # confirmation for the model to be train 
        if confirm_train:
            confirm = input("Is this the architecture you want to train? (yes/no): ")
            if confirm.strip().lower() != "yes":
                self._log("Training aborted.")
                return  

        # initialize  optimizer and loss function
        self.setup_optimizer_and_loss()
        loss = self.loss_fn

        # for mse value calculation in case the loss is not MSE
        mse_fn = MSELoss() if self.loss_type != "MSELoss" else None


        # accelerates training
        self.model, opt, train_dataloader, val_dataloader = self.accelerator.prepare(self.model, self.optimizer, train_dataloader, val_dataloader)
        
        # initialize early stopping
        early_stopping = EarlyStopping(patience=patience, debug=self.debug, path=f'{self.model_path}_best.pth',  logger=self.logger)

        # initialize scheduler for learning rate
        if self.scheduler:
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
            total_train_loss = self.train_one_epoch(train_dataloader, opt, loss, e, save_path, show_examples, number_of_examples)

                
            # call the validation function
            total_val_loss, total_psnr, total_ssim = self.validation(val_dataloader, loss, mse_fn, e)

            
            # update our training history
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_val_loss = total_val_loss / len(val_dataloader)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            avg_psnr = total_psnr / len(val_dataloader)
            avg_ssim = total_ssim / len(val_dataloader)
            history["psnr"].append(avg_psnr)
            history["ssim"].append(avg_ssim)

            # update the learning rate scheduler (when the validation loss is not improving)
            if self.scheduler:
                scheduler.step(avg_val_loss)

            # check ealy stopping
            early_stopping(avg_val_loss, self.model)

        
            if early_stopping.early_stop:
                self._log(f"Early stopping stopped at epoch {e+1}")
                break
            
        
            # print the model training and validation information
            t.set_postfix({
                "train_loss": f"{avg_train_loss:.6f}",
                "val_loss": f"{avg_val_loss:.6f}"
            })

            #Clean memory
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()


        # change model training 
        self.trained = True

        # finish measuring how long training took
        end_time = time.time()
        self._log("Total time taken to train the model: {:.2f}s".format(end_time - start_time))


        # Save final metrics
        with open(f'{self.model_path}_metrics.json', 'w') as f:
            json.dump(history, f)

        self._log(f"Metrics saved as '{self.model_path}_metrics.json'")
        
        return history


    def _get_test_dataloaders(self):
        """
        Loads test dataset and returns DataLoader.

        Returns:
            DataLoader: Dataloader for the test dataset.
        """

        # Load test dataset
        is_gcs = self.test_path.startswith("gs://")

        test_data = LoDoPaBDataset(self.test_path, 
        self.vg, 
        self.angles, 
        self.pg, 
        self.A, 
        self.single_bp,
        self.n_single_BP, 
        self.alpha,  
        self.i_0, 
        self.sigma, 
        self.seed, 
        self.max_len_test, 
        False, 
        self.dataset_logger)

        test_dataloader = DataLoader(test_data, 
        batch_size=self.batch_size, 
        shuffle=False, 
        num_workers=0 if is_gcs else 4,
        pin_memory=True,
        persistent_workers=not is_gcs) 

        return test_dataloader



    def evaluate(self, test_dataloader, loss, mse_fn):
        """
        Evaluates the model on the test dataset.

        Args:
            test_dataloader (DataLoader): Dataloader for the test set.
            loss (nn.Module): Loss function used during testing.
            mse_fn (nn.Module or None): MSE loss function for computing PSNR if needed.

        Returns:
            tuple: (predictions, ground_truths, total_loss, total_psnr, total_ssim)
        """

        # save ground truth and predictions
        gt_images = []
        predictions = []
        noisy_sinograms = []

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            self.model.eval()

            # initialize metrics
            total_test_loss = 0
            total_psnr = 0
            total_ssim = 0

            # loop over the validation set
            for batch in tqdm(test_dataloader):
                # send the input to the device
                ground_truth = batch['ground_truth']
                noisy_sino = batch['noisy_sinogram']
                if self.single_bp:
                    input_data = batch["single_back_projections"]
                else:
                    input_data = noisy_sino

                

                # make the predictions and calculate the validation loss
                pred = self.model(input_data)
                loss_val = loss(pred, ground_truth).item()
                total_test_loss += loss_val

                # checking that predictions and ground truth images have same shape
                assert ground_truth.shape == pred.shape, f"[ERROR] Shape mismatch: predicted {pred.shape}, ground truth {ground_truth.shape}"

                # compute metrics
                if self.loss_type == "MSELoss" or mse_fn is None:
                    mse_val = loss_val
                else:
                    mse_val = mse_fn(pred, ground_truth).item()

                total_psnr += compute_psnr(mse_val, self.alpha)
                total_ssim += compute_ssim(pred, ground_truth, self.alpha)

                # save gound truth and prediction
                predictions.append(pred.cpu())
                gt_images.append(ground_truth.cpu())
                noisy_sinograms.append(noisy_sino.cpu())

                #Clean memory
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        return predictions, gt_images, noisy_sinograms, total_test_loss, total_psnr, total_ssim


    def test(self, test_path, max_len_test=None):
        """
        Runs evaluation on the test dataset and saves predictions, metrics, and sinograms.

        Args:
            test_path (str): Path to the test dataset.
            max_len_test (int, optional): Max number of test samples to use.

        Returns:
            dict: Dictionary with average test loss, PSNR, and SSIM.
        """

        #changing paths parameters
        self.test_path = test_path
        self.max_len_test = max_len_test

        # Check if the model has been trained
        if not self.trained:
            self._log("This model is not trained yet.", level='warning')
            return

        # fix seed for reproducibility
        self._set_seed()

        # get test data loader
        test_dataloader = self._get_test_dataloaders()
        
        # for mse value calculation in case the loss is not MSE
        mse_fn = MSELoss() if self.loss_type != "MSELoss" else None


        self._log(f"Testing the {self.model_type} model...")

        # initialize  optimizer and loss function
        self.setup_optimizer_and_loss()
        model, loss, test_dataloader = self.accelerator.prepare(self.model, self.loss_fn, test_dataloader)

        # call evaliation function
        predictions, gt_images, noisy_sinograms, total_test_loss, total_psnr, total_ssim = self.evaluate(test_dataloader, loss, mse_fn)
        
    
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
        torch.save(torch.cat(noisy_sinograms), f"{self.model_path}_noisy_sinograms.pt")
        self._log(f"Predictions saved to {self.model_path}_predictions_images.pt")
        self._log(f"Ground Truth images saved to {self.model_path}_ground_truth_images.pt")
        self._log(f"Noisy sinograms saved to {self.model_path}_noisy_sinograms.pt")

        # save metrics to file
        with open(f'{self.model_path}_test_metrics.json', 'w') as f:
            json.dump(results, f)

        self._log(f"Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        self._log(f"Test metrics saved to {self.model_path}_test_metrics.json")

        return results
        




    def results(self, mode, example_number = 0, save_path=None):
        """
        Visualizes and logs training and/or test results.

        This includes:
        - Loss, PSNR, SSIM over epochs for training.
        - Quantitative and visual results from testing.
        - Optional combination of both in one analysis.

        Args:
            mode (str): One of ['training', 'testing', 'both'].
            example_number (int): Number of test examples to visualize.
            save_path (str, optional): Base path for saving generated plots.

        Returns:
            list[int] or None: Indices of plotted test examples (only for testing mode).
        """
        # checking if the model has been trained before
        if not self.trained:
            self._log(f"This model is not trained yet.",  level='warning')
            return
        
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
            plot_metric(epochs,loss_dict, title='Loss over Epochs', xlabel='Epoch', ylabel='Loss', test_value=None, save_path=f"{save_path}_training_loss.png" )
            self._log(f"Saved plot to {save_path}_training_loss.png")
            

            # plot PSNR
            psnr_dict = {'PSNR metric':  history['psnr']}
            plot_metric(epochs, psnr_dict, 'Validation PSNR over Epochs', 'Epoch', 'PSNR (dB)', test_value=None, save_path=f"{save_path}_training_psnr.png")
            self._log(f"Saved plot to {save_path}_training_psnr.png")
            

            # plot SSIM
            ssim_dict = {'SSIM metric':  history['ssim']}
            plot_metric(epochs, ssim_dict, 'Validation SSIM over Epochs', 'Epoch', 'SSIM', test_value=None, save_path=f"{save_path}_training_ssim.png")
            self._log(f"Saved plot to {save_path}_training_ssim.png")
        


        elif mode == "testing":
            # handling file path error
            if not os.path.exists(f"{self.model_path}_test_metrics.json"):
                self._log(f"File not found: {self.model_path}_test_metrics.json", level='error')
                return None
            elif not os.path.exists(f"{self.model_path}_predictions_images.pt"):
                self._log("Predictions .pt files not found", level="error")
                return
            elif not os.path.exists(f"{self.model_path}_ground_truth_images.pt"):
                self._log("Ground truth .pt files not found", level="error")
                return
            else:
                with open(f'{self.model_path}_test_metrics.json', 'r') as f:
                    test_results = json.load(f)
                predictions = torch.load(f"{self.model_path}_predictions_images.pt")
                ground_truths = torch.load(f"{self.model_path}_ground_truth_images.pt")

            #plot results
            print("\n=== Testing Results ===")
            print(f"Test Loss: {test_results['test_loss']:.6f}")
            print(f"Test PSNR: {test_results['psnr']:.2f}")
            print(f"Test SSIM: {test_results['ssim']:.4f}")

            # get random samples
            random.seed(self.seed)
            samples = random.sample(range(0, len(predictions)), example_number)

            for example in samples:
                show_example(predictions[example], ground_truths[example])

            return samples
                


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
            plot_metric(epochs,loss_dict, title='Loss over Epochs', xlabel='Epoch', ylabel='Loss', test_value=test_results['test_loss'], save_path=f"{save_path}_train_test_loss.png")
            self._log(f"Saved plot to {save_path}_train_test_loss")

            # plot PSNR
            psnr_dict = {'Val PSNR': history['psnr']}
            plot_metric(epochs, psnr_dict, title='PSNR over Epochs', xlabel='Epoch', ylabel='PSNR (dB)', test_value=test_results['psnr'], save_path=f"{save_path}_train_test_psnr.png" )
            self._log(f"Saved plot to {save_path}_train_test_psnr.png")

            # plot SSIM
            ssim_dict = {'Val SSIM': history['ssim']}
            plot_metric(epochs, ssim_dict, title='SSIM over Epochs', xlabel='Epoch', ylabel='SSIM', test_value=test_results['ssim'], save_path=f"{save_path}_train_test_ssim.png" )
            self._log(f"Saved plot to {save_path}_train_test_ssim.png")

        else:
            raise ValueError(f"Result mode not supported. Choose when of 'training', 'testing', or 'both")




    def other_ct_reconstruction(self, sinogram, num_iterations_sirt=100, num_iterations_em= 100, num_iterations_tv_min=100, num_iterations_nag_ls = 100, lamda = 0.0001):
        """
        Performs CT image reconstruction using multiple algorithms provided by TomoSipo.

        Args:
            sinogram (torch.Tensor): The measured projection data (sinogram).
            num_iterations_sirt (int): Number of iterations for SIRT reconstruction.
            num_iterations_em (int): Number of iterations for Expectation Maximization.
            num_iterations_tv_min (int): Number of iterations for Total Variation minimization.
            num_iterations_nag_ls (int): Number of iterations for NAG-LS method.
            lamda (float): Regularization parameter for TV minimization (TV-L2 model).

        Returns:
            dict: Dictionary containing reconstructed images from each method:
                {   "fbp": <Tensor>,
                    "sirt": <Tensor>,
                    "em": <Tensor>,
                    "tv_min": <Tensor>,
                    "nag_ls": <Tensor> }
        
        Note:
            FDK is not included in this method as it is designed for 3D cone-beam geometry,
            which is not compatible with the current fan-beam setup.
        """
        # Filtered Backprojection (FBP) reconstruction
        rec_fbp = fbp(self.A, sinogram)
        #Simultaneous Iterative Reconstruction Technique (SIRT)
        rec_sirt = sirt(self.A, sinogram, num_iterations_sirt)
        # Expectation Maximization (EM) reconstruction
        rec_em = em(self.A, sinogram, num_iterations_em)
        #Total Variation regularized least squares using Chambolle-Pock algorithm
        rec_tv_min = tv_min2d(self.A, sinogram, lamda, num_iterations_tv_min)
        #Nesterov Accelerated Gradient for Least Squares reconstruction 
        rec_nag_ls = nag_ls(self.A, sinogram, num_iterations_nag_ls)

        return {"fbp": rec_fbp,
            "sirt": rec_sirt,
            "em": rec_em,
            "tv_min": rec_tv_min,
            "nag_ls": rec_nag_ls}




    def report_results_images(self, save_path, samples, num_iterations_sirt=100, num_iterations_em= 100, num_iterations_tv_min=100, num_iterations_nag_ls = 100, lamda = 0.0001):
        """
        Generates and saves visual comparisons between model predictions, ground truth, 
        and multiple classical CT reconstruction methods.

        Args:
            save_path (str): Path prefix where the output images will be saved.
            samples (list of int): Indices of test samples to visualize.
            num_iterations_sirt (int): Iterations for SIRT reconstruction.
            num_iterations_em (int): Iterations for EM reconstruction.
            num_iterations_tv_min (int): Iterations for TV minimization.
            num_iterations_nag_ls (int): Iterations for NAG-LS.
            lamda (float): Regularization parameter for TV minimization.

        Raises:
            RuntimeError: If required .pt files (predictions, sinograms, etc.) are missing.
        """
        
        if not self.trained:
            self._log(f"This model is not trained yet.",  level='warning')
            return

        # handling file path error
        if not os.path.exists(f"{self.model_path}_predictions_images.pt"):
            self._log("Prediction .pt files not found", level="error")
            return
        elif not os.path.exists(f"{self.model_path}_ground_truth_images.pt"):
            self._log("Ground truth .pt files not found", level="error")
            return

        elif not os.path.exists(f"{self.model_path}_noisy_sinograms.pt"):
            self._log("Noisy sinograms .pt files not found", level="error")
            return
        else:
            predictions = torch.load(f"{self.model_path}_predictions_images.pt")
            ground_truths = torch.load(f"{self.model_path}_ground_truth_images.pt")
            sinograms = torch.load(f"{self.model_path}_noisy_sinograms.pt")

        # generate and save all plots (model reconstructed image, gt image and classical methods reconstrucstions)
        for sample in samples:
            reconstructions_dict = self.other_ct_reconstruction(sinograms[sample], num_iterations_sirt, num_iterations_em, num_iterations_tv_min, num_iterations_nag_ls, lamda)
            plot_different_reconstructions(self.model_type, sample, reconstructions_dict, predictions[sample], ground_truths[sample], save_path)



    def report_results_table(self, save_path, num_iterations_sirt=100, num_iterations_em=100,
                         num_iterations_tv_min=100, num_iterations_nag_ls=100, lamda=0.0001):
        if not self.trained:
            self._log(f"This model is not trained yet.", level='warning')
            return

        # File existence checks
        if not os.path.exists(f"{self.model_path}_ground_truth_images.pt"):
            self._log("Ground truth .pt files not found", level="error")
            return
        if not os.path.exists(f"{self.model_path}_noisy_sinograms.pt"):
            self._log("Noisy sinograms .pt files not found", level="error")
            return

        # Load data
        ground_truths = torch.load(f"{self.model_path}_ground_truth_images.pt")
        sinograms = torch.load(f"{self.model_path}_noisy_sinograms.pt")

        metrics = {
            "Algorithm": ["FBP", "SIRT", "EM", "TV-Min", "NAG-LS"],
            "PSNR": [0] * 5,
            "SSIM": [0] * 5
        }

        n_samples = len(sinograms)

        for i in range(n_samples):
            gt_image = ground_truths[i]
            recon_dict = self.other_ct_reconstruction(
                sinograms[i],
                num_iterations_sirt=num_iterations_sirt,
                num_iterations_em=num_iterations_em,
                num_iterations_tv_min=num_iterations_tv_min,
                num_iterations_nag_ls=num_iterations_nag_ls,
                lamda=lamda
            )

            metrics["PSNR"][0] += compute_psnr_results(recon_dict["fbp"], gt_image, self.alpha)
            metrics["PSNR"][1] += compute_psnr_results(recon_dict["sirt"], gt_image, self.alpha)
            metrics["PSNR"][2] += compute_psnr_results(recon_dict["em"], gt_image, self.alpha)
            metrics["PSNR"][3] += compute_psnr_results(recon_dict["tv_min"], gt_image, self.alpha)
            metrics["PSNR"][4] += compute_psnr_results(recon_dict["nag_ls"], gt_image, self.alpha)

            metrics["SSIM"][0] += compute_ssim(recon_dict["fbp"], gt_image, self.alpha)
            metrics["SSIM"][1] += compute_ssim(recon_dict["sirt"], gt_image, self.alpha)
            metrics["SSIM"][2] += compute_ssim(recon_dict["em"], gt_image, self.alpha)
            metrics["SSIM"][3] += compute_ssim(recon_dict["tv_min"], gt_image, self.alpha)
            metrics["SSIM"][4] += compute_ssim(recon_dict["nag_ls"], gt_image, self.alpha)

        # Average metrics
        metrics["PSNR"] = [val / n_samples for val in metrics["PSNR"]]
        metrics["SSIM"] = [val / n_samples for val in metrics["SSIM"]]

        # Create DataFrame
        df = pd.DataFrame(metrics)

        # Save as CSV
        df.to_csv(f"{save_path}_reconstruction_metrics.csv", index=False)

        self._log(f"Results table saved to {save_path}_reconstruction_metrics.csv", level="info")



    def load_model(self, path=None):
        """
        Loads model weights from the specified file.

        Args:
            path (str, optional): Path to the model checkpoint. If None, uses self.model_path + '_best.pth'.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        # we first need to check if we create correctly the model
        if self.model is None:
            raise ValueError("Model instance is not initialized. Set self.model before loading weights.")

        if path is None:
            path = f"{self.model_path}_best.pth"

        if not os.path.exists(path):
            self._log(f"Model file not found: {path}", level="error")
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self._log(f"Model weights loaded from {path}")

        #change training state 
        self.trained = True