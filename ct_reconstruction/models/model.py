import tomosipo as ts
import torch
from torch.nn import Module
from torch.nn import MSELoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..datasets.dataset import LoDoPaBDataset
from ..callbacks.early_stopping import EarlyStopping
from ..utils.metrics import compute_psnr, compute_ssim
from ..utils.plotting import show_example, plot_metric
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


class ModelBase(Module):
    """
    Abstract base class for CT reconstruction model training, evaluation, and testing.

    This class provides a general framework to:
    - Load CT datasets and reconstruction operators
    - Train deep learning models using PyTorch
    - Evaluate performance via PSNR, SSIM, and loss
    - Save/load predictions and metrics
    - Plot performance metrics over epochs

    Attributes:
        training_path (str): Path to the training dataset.
        validation_path (str): Path to the validation dataset.
        test_path (str): Path to the test dataset.
        model_path (str): Path to save trained models and logs.
        model_type (str): Name or identifier of the model architecture.
        n_single_BP (int): Number of single backprojections per sample.
        alpha (float): Normalization factor used for scaling.
        i_0 (float): Incident X-ray intensity for noise modeling.
        sigma (float): Standard deviation of the noise.
        max_len_train (int): Max number of training samples to load.
        max_len_val (int): Max number of validation samples to load.
        batch_size (int): Number of samples per batch.
        epochs (int): Total number of training epochs.
        optimizer_type (str): Optimizer type (e.g., "Adam", "AdamW").
        loss_type (str): Loss function name (e.g., "MSELoss").
        learning_rate (float): Initial learning rate.
        debug (bool): Whether to print debug messages.
        seed (int): Random seed for reproducibility.
        scheduler (bool): Whether to use a learning rate scheduler.
        log_file (str): Path to the model logger.
    """

    def __init__(self, training_path, validation_path, test_path, model_path, model_type, n_single_BP, alpha, i_0, sigma,  max_len_train, max_len_val, batch_size, epochs, optimizer_type, loss_type, learning_rate, debug, seed, scheduler = True, log_file='training.log'):
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
        self.training_path = training_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.n_single_BP = n_single_BP
        self.alpha = alpha
        self.i_0 = i_0
        self.sigma = sigma
        self.max_len_train = max_len_train
        self.max_len_val = max_len_val

        # model parameters
        self.model = self
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

        # logger configuration
        self.logger = configure_logger("ct_reconstruction.models.model", log_file, debug=self.debug)
        self.dataset_logger = configure_logger("ct_reconstruction.dataset", log_file, debug=False)
        
        # set the device once for the whole class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.accelerator.device

        # model
        self.model = None


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

        #optimiser
        if self.optimizer_type == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        elif self.optimizer_type == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Loss function
        if self.loss_type == "MSELoss":
            self.loss_fn = MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")



    def train_one_epoch(self, train_dataloader, opt, loss, e):
        """
        Performs one full training epoch on the dataset.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            opt (Optimizer): Optimizer used for model updates.
            loss (nn.Module): Loss function.
            e (int): Current epoch index (for logging).

        Returns:
            float: Total accumulated training loss.
        """
        # set the model in training mode
        self.model.train()

        # initialize the total training loss
        total_train_loss = 0

        # loop over the training set
        for batch in train_dataloader:

            # send the input to the device
            ground_truth = batch["ground_truth"]
            single_back_projections = batch["single_back_projections"]

            # perform a forward pass and calculate the training loss
            pred = self.model(single_back_projections)
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


    def validation(self, val_dataloader, loss, mse_fn, e):
        """
        Evaluates model performance on validation data.

        Args:
            val_dataloader (DataLoader): Validation DataLoader.
            loss (nn.Module): Loss function.
            mse_fn (nn.Module or None): MSE function, used for PSNR calculation.
            e (int): Current epoch index.

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
                single_back_projections = batch["single_back_projections"]
            
                # make the predictions and calculate the validation loss
                pred = self.model(single_back_projections)
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
                total_ssim += compute_ssim(pred, ground_truth)
        
        return total_val_loss, total_psnr, total_ssim

        

    def train(self, patience, confirm_train=False):
        """
        Full training loop including early stopping, metric logging, and scheduler.

        Args:
            patience (int): Early stopping patience in epochs.
            confirm_train (bool): Whether to require user confirmation before training.

        Returns:
            dict: Training history with loss and metrics per epoch.
        """

        # fix seed
        self._set_seed()

    
        # load training and validation datasets
        train_dataloader, val_dataloader = self._get_dataloaders()

    
        # show model summary
        self.model.to(self.device)
        sample = next(iter(train_dataloader))["single_back_projections"]  # single_back_projections
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
            total_train_loss = self.train_one_epoch(train_dataloader, opt, loss, e)

                
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
        is_gcs = self.training_path.startswith("gs://")

        test_data = LoDoPaBDataset(self.test_path, 
        self.vg, 
        self.angles, 
        self.pg, 
        self.A, 
        self.n_single_BP, 
        self.alpha,  
        self.i_0, 
        self.sigma, 
        self.seed, 
        max_len_test, 
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
                single_back_projections = batch['single_back_projections']

                # make the predictions and calculate the validation loss
                pred = self.model(single_back_projections)
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
                total_ssim += compute_ssim(pred, ground_truth)

                # save gound truth and prediction
                predictions.append(pred.cpu())
                gt_images.append(ground_truth.cpu())

        return predictions, gt_images, total_test_loss, total_psnr, total_ssim


    def test(self, max_len_test):
        """
        Tests the trained model on the test set and saves predictions and metrics.

        Args:
            max_len_test (int): Maximum number of test samples to load.

        Returns:
            dict: Dictionary containing average test loss, PSNR, and SSIM.
        """

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
        predictions, gt_images, total_test_loss, total_psnr, total_ssim = self.evaluate(test_dataloader, loss, mse_fn)
        
    
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
        self._log(f"Predictions saved to {self.model_path}_predictions_images.pt")
        self._log(f"Predictions saved to {self.model_path}_ground_truth_images.pt")

        # save metrics to file
        with open(f'{self.model_path}_test_metrics.json', 'w') as f:
            json.dump(results, f)

        self._log(f"Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        self._log(f"Test metrics saved to {self.model_path}_test_metrics.json")

        return results
        




    def results(self, mode, example_number = 0, save_path=None):
        """
        Visualizes and logs training and/or test results using stored metrics and predictions.

        This function can generate plots for:
        - Training: loss, PSNR, and SSIM over epochs.
        - Testing: numerical results and qualitative visual comparisons.
        - Both: combines training and test metrics in the same plots (with test metric as a horizontal line).

        For testing mode, random image samples are displayed for visual inspection.

        Args:
            mode (str): One of 'training', 'testing', or 'both'.
            example_number (int): Number of test examples to visualize.
            save_path (str, optional): Path to save result plots. Defaults to current directory.

        Raises:
            ValueError: If `mode` is not one of the accepted values.
            FileNotFoundError: If required metrics or predictions files are missing.
        """

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



    def load_model(self, path=None):
        """
        Loads model weights from the specified file.

        Args:
            path (str, optional): Path to the model checkpoint. If None, uses self.model_path + '_best.pth'.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if path is None:
            path = f"{self.model_path}_best.pth"

        if not os.path.exists(path):
            self._log(f"Model file not found: {path}", level="error")
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self._log(f"Model weights loaded from {path}")