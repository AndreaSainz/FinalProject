import tomosipo as ts
from ts_algorithms import fbp,sirt, em, tv_min2d, nag_ls
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
import os
import gc
import pandas as pd



class ModelBase(Module):
    """
    Abstract base class for CT reconstruction training, evaluation, and analysis.

    This class provides a flexible interface for deep learning and classical CT reconstruction algorithms.
    It supports multiple data acquisition modes (full sinogram, sparse-view, single backprojection), integrates
    noise modeling, angle indexing via offsets, and defines full training, evaluation, visualization, and testing workflows.

    Attributes:
        model_path (str): Path to save checkpoints, metrics, predictions, etc.
        model_type (str): Identifier name for the model architecture.
        single_bp (bool): Whether to use single backprojection mode.
        sparse_view (bool): Whether to use sparse view sinogram mode.
        n_single_BP (int): Number of backprojections used per sample (if single_bp is True).
        view_angles (int): Number of angles used in sparse-view mode.
        alpha (float): Scaling factor for PSNR computation.
        i_0 (float): Incident X-ray intensity for noise modeling.
        sigma (float): Standard deviation of added Gaussian noise.
        batch_size (int): Batch size for training/validation/testing.
        epochs (int): Number of training epochs.
        optimizer_type (str): Optimizer type ('Adam', 'AdamW'). 
        loss_type (str): Loss function ('MSELoss', 'L1Loss').
        learning_rate (float): Learning rate for the optimizer.
        seed (int): Seed for reproducibility.
        debug (bool): If True, enables verbose logging.
        scheduler (bool): If True, uses learning rate scheduler.
        offset (int): Integer offset applied to angular indices. Only for testing.
        accelerator (Accelerator): accelerate.Accelerator object for mixed precision and device handling.
        trained (bool): Whether the model has been trained. This is an internal flag.
        indices_base (torch.Tensor): Base angular indices for sparse_view or single_bp mode. This are calculated inside the class.
        current_indices (torch.Tensor): Angular indices including offset. This are calculated inside the class if the offset is changed during testing.
        A (ts.Operator): Full tomosipo projection operator.
        A_sparse (ts.Operator): Sparse-view or backprojection projection operator. This is internally calculated for sparse-view CT.
        vg (ts.VolumeGeometry): Tomosipo volume geometry.
        pg (ts.ProjectionGeometry): Full-angle projection geometry.
        pg_sparse (ts.ProjectionGeometry): Projection geometry corresponding to selected indices. This is internally calculated for sparse-view CT.
        device (torch.device): Torch device derived from accelerator.

    Example:
        >>> from accelerate import Accelerator
        >>> model = MyModel(
        ...     model_path='checkpoints/my_model',
        ...     model_type='UNet',
        ...     single_bp=True,
        ...     n_single_BP=10,
        ...     sparse_view=False,
        ...     view_angles=50,
        ...     alpha=0.001,
        ...     i_0=1e5,
        ...     sigma=0.01,
        ...     batch_size=4,
        ...     epochs=100,
        ...     optimizer_type='AdamW',
        ...     loss_type='MSELoss',
        ...     learning_rate=1e-4,
        ...     debug=True,
        ...     seed=123,
        ...     accelerator=Accelerator(),
        ...     scheduler=True
        ... )
        >>> model.train("data/train", "data/val", save_path="outputs/recon")
        >>> results = model.test("data/test")
        >>> model.results("both", example_number=3, save_path="outputs/plots")
    """


    def __init__(self, model_path, model_type, single_bp , n_single_BP, sparse_view, view_angles,  alpha, i_0, sigma, batch_size, epochs, optimizer_type, loss_type, learning_rate, debug, seed, accelerator,  scheduler = True, log_file='training.log'):
        super().__init__()

        #Scan parameters from the paper and data
        self.pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.num_angles = 1000
        self.pixel_size = 26.0
        self.num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
        self.src_orig_dist = 575
        self.src_det_dist = 1050
                                                                              
        # dataset parameters
        self.model = None
        self.training_path = None
        self.validation_path = None
        self.test_path = None
        self.single_bp = single_bp
        self.n_single_BP = n_single_BP
        self.sparse_view = sparse_view
        self.view_angles = view_angles
        self.alpha = alpha
        self.i_0 = i_0
        self.sigma = sigma
        self.max_len_train = None
        self.max_len_val = None
        self.max_len_test = None
        self.offset = 0

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
        self.indices = None

        # Create tomosipo volume and projection geometry
        self.vg = ts.volume(shape=(1,self.pixels,self.pixels))                                                       # Volumen
        self.angles = np.linspace(0, np.pi, self.num_angles, endpoint=True)                                          # Angles
        self.pg = ts.cone(angles = self.angles, src_orig_dist=self.src_orig_dist,  shape=(1, self.num_detectors) )     # Fan beam structure
        self.A = ts.operator(self.vg,self.pg)     
   

        if self.sparse_view:
            self.indices_base = torch.linspace(0, self.num_angles - 1, steps=self.view_angles).long()
            # so the angles match the subset that was taken from the original angles
            angles_sparse = self.angles[self.indices_base] 
            self.pg_sparse = ts.cone(angles=angles_sparse, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors) )
            self.A_sparse = ts.operator(self.vg, self.pg_sparse)

        elif self.single_bp:
            self.indices_base = torch.linspace(0, self.num_angles - 1, steps=self.n_single_BP).long()
            # so the angles match the subset that was taken from the original angles
            angles_sparse = self.angles[self.indices_base] 
            self.pg_sparse = ts.cone(angles=angles_sparse, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors))
            self.A_sparse = ts.operator(self.vg, self.pg_sparse)
        else:
            self.indices_base = torch.arange(self.num_angles)
        

        # accelerator for faster code
        self.accelerator = accelerator
        self.trainable_params = None

        # logger configuration
        self.logger = configure_logger("ct_reconstruction.models.model", log_file, debug=self.debug)
        self.dataset_logger = configure_logger("ct_reconstruction.dataset", log_file, debug=False)
        
        # set the device once for the whole class
        self.device = self.accelerator.device


    @property
    def current_indices(self):
        return (self.indices_base + self.offset) % self.num_angles
    


    def _set_seed(self):
        """
        Sets the global random seed for reproducibility.

        This includes seeds for:
        - Python's built-in `random` module
        - NumPy operations
        - PyTorch CPU and CUDA (if available)

        It also configures PyTorch to use deterministic algorithms and disables benchmarking.

        Example:
            >>> self._set_seed()
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
        Logs a message with the specified severity level.

        This function writes to both the internal logger and optionally prints
        the message to the console based on the debug flag.

        Args:
            msg (str): The message to log.
            level (str): Logging severity ('debug', 'info', 'warning', 'error', 'critical').

        Example:
            >>> self._log("Training started", level="info")
        """
        # Get the logging method based on the provided level (defaults to .info)
        level_func = getattr(self.logger, level.lower(), self.logger.info)
        # Call the logging method with the message
        level_func(msg)



    def _get_dataloaders(self):
        """
        Loads training and validation datasets and creates corresponding DataLoaders.

        This method loads LoDoPaB datasets for both training and validation, then constructs
        torch DataLoaders with consistent seed initialization for reproducibility.

        Returns:
            tuple: (train_dataloader, val_dataloader)

        Example:
            >>> train_loader, val_loader = self._get_dataloaders()
        """
        
        # load training and validation datasets
        train_data = LoDoPaBDataset(self.training_path,
            self.vg,
            self.angles,
            self.pg,
            self.A,
            self.single_bp,
            self.n_single_BP,
            self.sparse_view, 
            self.current_indices,
            self.alpha,
            self.i_0,
            self.sigma,
            self.seed,
            self.max_len_train, 
            False,
            self.dataset_logger,
            self.accelerator.device)

        val_data = LoDoPaBDataset(self.validation_path, 
            self.vg, 
            self.angles, 
            self.pg, 
            self.A, 
            self.single_bp,
            self.n_single_BP, 
            self.sparse_view, 
            self.current_indices,
            self.alpha,  
            self.i_0, 
            self.sigma, 
            self.seed, 
            self.max_len_val, 
            False, 
            self.dataset_logger,
            self.accelerator.device)

        # create dataloader for both and a generator for reproducibility
        g = torch.Generator() 
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=g,
            worker_init_fn= lambda _: np.random.seed(self.seed)
        )

        val_dataloader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False, # validation and test do not need shuffle
            num_workers=0,
            pin_memory=True
        ) 

        return (train_dataloader, val_dataloader)
            



    def setup_optimizer_and_loss(self, learning_rate = None):
        """
        Initializes the optimizer and loss function for training.

        This method configures the optimizer and loss function based on the values 
        specified in `self.optimizer_type` and `self.loss_type`. Only model parameters 
        with `requires_grad=True` are passed to the optimizer.

        Supported optimizers:
            - "Adam"
            - "AdamW"

        Supported loss functions:
            - "MSELoss"
            - "L1Loss"

        Args:
            learning_rate (float, optional): Learning rate to use. If not provided, 
                uses `self.learning_rate`.

        Raises:
            ValueError: If an unsupported optimizer or loss type is provided.

        Example:
            >>> self.optimizer_type = "Adam"
            >>> self.loss_type = "MSELoss"
            >>> self.setup_optimizer_and_loss(learning_rate=1e-3)
        """

        #change parameters if neccesary
        learning_rate = learning_rate or self.learning_rate

        # Get only parameters that have requires_grad=True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Optimizer
        if self.optimizer_type == "Adam":
            self.optimizer = Adam(trainable_params, lr=learning_rate)
        elif self.optimizer_type == "AdamW":
            self.optimizer = AdamW(trainable_params, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Loss function
        if self.loss_type == "MSELoss":
            self.loss_fn = MSELoss()
        elif self.loss_type == "L1Loss":
            self.loss_fn = L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")



    def train_one_epoch(self, train_dataloader, opt, loss, e, save_path, show_examples, number_of_examples, fixed_input, fixed_gt):
        """
        Train the model for a single epoch.

        Performs forward passes, loss computation, backpropagation, and optimizer steps
        for each batch. Optionally saves output predictions for a fixed batch every 5 epochs.

        Args:
            train_dataloader (DataLoader): Dataloader for training data.
            opt (torch.optim.Optimizer): Optimizer instance.
            loss (nn.Module): Loss function.
            epoch_idx (int): Current epoch number.
            save_path (str): Path prefix for saving visual outputs.
            show_examples (bool): If True, visualizes model predictions every 5 epochs.
            number_of_examples (int): Number of samples to visualize.
            fixed_input (Tensor): Batch of fixed inputs for visualization.
            fixed_gt (Tensor): Corresponding ground truth for `fixed_input`.

        Returns:
            float: Total accumulated training loss for the epoch.

        Example:
            >>> loss = nn.MSELoss()
            >>> train_loss = self.train_one_epoch(loader, optimizer, loss, 0, 'out/epoch0', True, 2, x_fixed, y_fixed)
        """

        # set the model in training mode
        self.model.train()

        # initialize the total training loss
        total_train_loss = 0

        # loop over the training set
        for batch_id, batch in enumerate(train_dataloader):

            # send the input to the device
            ground_truth = batch["ground_truth"]

            if self.single_bp:
                input_data = batch["single_back_projections"]
            elif self.sparse_view:
                input_data = batch["sparse_sinogram"]
            else:
                input_data = batch["noisy_sinogram"]

            # perform a forward pass and calculate the training loss
            pred = self.model(input_data)
            loss_value = loss(pred, ground_truth)

            # checking that predictions and ground truth images have same shape
            assert ground_truth.shape == pred.shape, f"[ERROR] Shape mismatch: predicted {pred.shape}, ground truth {ground_truth.shape}"

            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            self.accelerator.backward(loss_value)
            opt.step()

            # add the loss to the total training loss so far
            total_train_loss += loss_value.item()

            #Clean memory
            del input_data, ground_truth, pred, loss_value
            if batch_id % 10 == 0:
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()


        if show_examples and fixed_input is not None and fixed_gt is not None and e % 5 == 0:
            with torch.no_grad():  # Asegura que no guarda gradientes
                fixed_pred = self.model(fixed_input)
                for i in range(min(number_of_examples, fixed_pred.shape[0])):
                    show_example_epoch(fixed_pred[i], fixed_gt[i], e, f"{save_path}_epochs_{i}")
                    self._log(f"[INFO] Saved example figure: {save_path}_epochs_{i}_{e}.png")
                
        return total_train_loss


    def validation(self, val_dataloader, loss, mse_fn, e):
        """
        Evaluate model performance on the validation set.

        Computes total validation loss, PSNR, and SSIM. If the training loss is not MSE,
        a separate MSE function is used for PSNR computation.

        Args:
            val_dataloader (DataLoader): Dataloader for validation data.
            loss (nn.Module): Loss function used during training.
            mse_fn (nn.Module or None): MSE loss function used to compute PSNR, if needed.
            e (int): Current epoch index (used for progress display).

        Returns:
            tuple: (total_val_loss, total_psnr, total_ssim)

        Example:
            >>> val_loss, psnr, ssim = self.validation(val_loader, loss_fn, mse_fn, e=1)
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
            for batch_id, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {e+1}", leave=False)):

                # send the input to the device
                ground_truth = batch["ground_truth"]
                if self.single_bp:
                    input_data = batch["single_back_projections"]
                elif self.sparse_view:
                    input_data = batch["sparse_sinogram"]
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

                total_psnr += compute_psnr(mse_val, 1)
                total_ssim += compute_ssim(pred, ground_truth, 1)

                 #Clean memory
                if batch_id == 10:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        return total_val_loss, total_psnr, total_ssim

        

    def train(self, training_path, validation_path, save_path, max_len_train = None, max_len_val=None, patience=10, epochs=None, learning_rate=None, confirm_train=False, show_examples=True, number_of_examples=1):
        """
        Train the model with validation, early stopping, and optional learning rate scheduling.

        Performs full training pipeline: model setup, dataloader preparation, training/validation loop,
        checkpointing, metric tracking, and optional visualization of predictions.

        Args:
            training_path (str): Path to the training dataset directory.
            validation_path (str): Path to the validation dataset directory.
            save_path (str): Prefix path for saving checkpoints and visual outputs.
            max_len_train (int, optional): Max number of training samples to use.
            max_len_val (int, optional): Max number of validation samples to use.
            patience (int): Number of epochs to wait for improvement before early stopping.
            epochs (int, optional): Total number of training epochs. Overrides default.
            learning_rate (float, optional): Learning rate for optimizer. Overrides default.
            confirm_train (bool): If True, prompts user to confirm model architecture before training.
            show_examples (bool): Whether to visualize and save sample predictions.
            number_of_examples (int): Number of examples to visualize when `show_examples` is True.

        Returns:
            dict: Dictionary with history of losses, PSNR, and SSIM per epoch.

        Example:
            >>> history = model.train(\"data/train\", \"data/val\", save_path=\"checkpoints/my_model\")
        """

        #changing paths parameters
        self.offset = 0 # So if I want to retraine I am sure I am using the same angles
        epochs = epochs or self.epochs
        learning_rate = learning_rate or self.learning_rate
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
        elif self.sparse_view:
            sample = next(iter(train_dataloader))["sparse_sinogram"]
        else:
            sample = next(iter(train_dataloader))["noisy_sinogram"]
        #summary(self.model, input_size=tuple(sample.shape[1:])) #just for debugging

        # confirmation for the model to be train 
        if confirm_train:
            confirm = input("Is this the architecture you want to train? (yes/no): ")
            if confirm.strip().lower() != "yes":
                self._log("Training aborted.")
                return  

        # initialize  optimizer and loss function
        self.setup_optimizer_and_loss(learning_rate)
        loss = self.loss_fn

        # for mse value calculation in case the loss is not MSE
        mse_fn = MSELoss() if self.loss_type != "MSELoss" else None


        # accelerates training
        self.model, opt, train_dataloader, val_dataloader = self.accelerator.prepare(self.model, self.optimizer, train_dataloader, val_dataloader)
        
        # fixed one batch if show_example is require
        if show_examples:
            fixed_batch = next(iter(train_dataloader))

            if self.single_bp:
                fixed_input = fixed_batch["single_back_projections"] 
            elif self.sparse_view:
                fixed_input = fixed_batch["sparse_sinogram"] 
            else:
                fixed_input = fixed_batch["noisy_sinogram"]

            fixed_gt = fixed_batch["ground_truth"]

        # initialize early stopping
        early_stopping = EarlyStopping(patience=patience, debug=self.debug, path=f'{self.model_path}',  logger=self.logger, accelerator = self.accelerator)

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

        t = tqdm(range(epochs), desc="Epochs")

        # loop over our epochs
        for e in t:

            # call the training function
            total_train_loss = self.train_one_epoch(train_dataloader, opt, loss, e, save_path, show_examples, number_of_examples, fixed_input, fixed_gt)

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
        Constructs the test DataLoader using the LoDoPaBDataset.

        Initializes the test dataset with all relevant geometrical and noise parameters.
        Uses `self.current_indices` to define angular subsets if `sparse_view` or `single_bp` is active.

        Returns:
            DataLoader: PyTorch DataLoader for the test dataset (no shuffle, pin_memory=True).

        Example:
            >>> self.test_path = "data/ground_truth_test"
            >>> self.max_len_test = 200
            >>> test_loader = self._get_test_dataloaders()
        """

        test_data = LoDoPaBDataset(self.test_path, 
        self.vg, 
        self.angles, 
        self.pg, 
        self.A, 
        self.single_bp,
        self.n_single_BP, 
        self.sparse_view, 
        self.current_indices,
        self.alpha,  
        self.i_0, 
        self.sigma, 
        self.seed, 
        self.max_len_test, 
        False, 
        self.dataset_logger,
        self.accelerator.device)

        test_dataloader = DataLoader(test_data, 
        batch_size=self.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
        ) 

        return test_dataloader



    def evaluate(self, test_dataloader, loss, mse_fn):
        """
        Evaluates the model on the test set using loss, PSNR, and SSIM.

        Performs forward passes without gradients and accumulates metrics across all batches.
        Also returns predictions, ground truths, and input sinograms for optional analysis.

        Args:
            test_dataloader (DataLoader): DataLoader containing test samples.
            loss (nn.Module): Loss function used for evaluation.
            mse_fn (nn.Module or None): MSE function used to compute PSNR if loss is not MSE.

        Returns:
            tuple: (
                predictions (List[Tensor]): List of prediction batches (on CPU),
                ground_truths (List[Tensor]): List of ground truth batches (on CPU),
                sinograms (List[Tensor]): List of input sinograms (sparse or noisy, on CPU),
                total_loss (float): Accumulated test loss,
                total_psnr (float): Accumulated PSNR across batches,
                total_ssim (float): Accumulated SSIM across batches
            )

        Example:
            >>> preds, ground_truth, sinograms, loss, psnr, ssim = model.evaluate(test_loader, loss_fn, mse_fn)
        """

        # save ground truth and predictions
        gt_images = []
        predictions = []
        sinograms = []

        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            self.model.eval() 
            
            # initialize metrics
            total_test_loss = 0
            total_psnr = 0
            total_ssim = 0

            # loop over the validation set
            for batch_id, batch in enumerate(tqdm(test_dataloader)):
                # send the input to the device
                ground_truth = batch['ground_truth']

                if self.single_bp:
                    input_data = batch["single_back_projections"]
                    sino = batch['sparse_sinogram']
                elif self.sparse_view:
                    input_data = batch["sparse_sinogram"]
                    sino = input_data
                else:
                    input_data = batch["noisy_sinogram"]
                    sino = input_data

                

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

                total_psnr += compute_psnr(mse_val, 1)
                total_ssim += compute_ssim(pred, ground_truth, 1)

                # save gound truth and prediction
                predictions.append(pred.cpu())
                gt_images.append(ground_truth.cpu())
                sinograms.append(sino.cpu())

                #Clean memory
                if batch_id == 10:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        return predictions, gt_images, sinograms, total_test_loss, total_psnr, total_ssim


    def test(self, test_path, max_len_test=None, offset = 0):
        """
        Tests the trained model on a designated test dataset.

        Loads the test dataset, runs forward inference, computes evaluation metrics (loss, PSNR, SSIM),
        and saves predictions, ground truths, and input sinograms to disk.

        Args:
            test_path (str): Path to the test dataset.
            max_len_test (int, optional): Max number of test samples to load.
            offset (int, optional): Angular index offset to apply if `sparse_view` or `single_bp` is used.

        Returns:
            dict: Dictionary with average test loss, PSNR, and SSIM scores.

        Side Effects:
            - Modifies `self.test_path` and `self.max_len_test`.
            - Saves .pt files for predictions, ground truth images, and sinograms.
            - Saves metrics to a .json file.

        Example:
            >>> results = model.test("data/ground_truth_test", max_len_test=200)
        """

        if (self.single_bp or self.sparse_view) and self.offset != 0:
            self.offset = offset

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
        self.model, loss, test_dataloader = self.accelerator.prepare(self.model, self.loss_fn, test_dataloader)

        # call evaliation function
        predictions, gt_images, sinograms, total_test_loss, total_psnr, total_ssim = self.evaluate(test_dataloader, loss, mse_fn)
        
        
    
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
        self._log(f"Ground Truth images saved to {self.model_path}_ground_truth_images.pt")
        if self.sparse_view or self.single_bp:
            torch.save(torch.cat(sinograms), f"{self.model_path}_sparse_sinograms.pt")
            self._log(f"Sparse-view sinograms saved to {self.model_path}_sparse_sinograms.pt")
        else : 
            torch.save(torch.cat(sinograms), f"{self.model_path}_noisy_sinograms.pt")
            self._log(f"Noisy sinograms saved to {self.model_path}_noisy_sinograms.pt")

        # save metrics to file
        with open(f'{self.model_path}_test_metrics.json', 'w') as f:
            json.dump(results, f)

        self._log(f"Test Loss: {avg_test_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        self._log(f"Test metrics saved to {self.model_path}_test_metrics.json")

        return results
        




    def results(self, mode, example_number = 0, save_path=None):
        """
        Generate visual summaries of training and/or testing results.

        Depending on the mode, this function will:
        - Plot training loss, PSNR, and SSIM over epochs.
        - Print test metrics and visualize prediction vs ground truth images.
        - Optionally save all generated plots.

        Args:
            mode (str): One of ['training', 'testing', 'both']. Determines which plots to produce.
            example_number (int): Number of test samples to visualize in 'testing' mode.
            save_path (str, optional): Path prefix for saving plots.

        Returns:
            list[int] or None: List of indices of test samples visualized (only if mode='testing').

        Raises:
            ValueError: If `mode` is invalid.

        Example:
            >>> model.results(mode='training', save_path='outputs/plots')
            >>> model.results(mode='testing', example_number=5)
            >>> model.results(mode='both')
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


    def _get_operator(self):
        """
        Selects and returns the appropriate tomography operator.

        Returns the standard operator (`self.A`) for full-view scenarios, or constructs
        and returns a sparse-view operator (`self.A_sparse`) when `sparse_view` or `single_bp`
        is enabled. If `offset` is non-zero, the operator is redefined using
        `self.current_indices` to apply angular shifting.

        Returns:
            ts.Operator: Tomosipo operator configured for the current acquisition mode.

        Notes:
            - If `offset == 0`, a previously cached sparse operator is reused.
            - If `offset != 0`, a new sparse operator is constructed with shifted angles.
            - Assumes `self.angles` and `self.current_indices` are up to date.
        
        Example:
            >>> A = model._get_operator()
        """
        
        if (self.sparse_view or self.single_bp) and self.offset == 0:
            return self.A_sparse
        elif (self.single_bp or self.sparse_view) and self.offset != 0:
            angles_sparse = self.angles[self.current_indices]
            self.pg_sparse = ts.cone(angles=angles_sparse, src_orig_dist=self.src_orig_dist, shape=(1, self.num_detectors) )
            self.A_sparse = ts.operator(self.vg, self.pg_sparse)
            return self.A_sparse
        else:
            return self.A
        
    

    def other_ct_reconstruction(self, sinogram, A, num_iterations_sirt=100, num_iterations_em= 100, num_iterations_tv_min=100, num_iterations_nag_ls = 100, lamda = 0.0001):
        """
        Performs classical CT reconstruction algorithms on a given sinogram.

        Applies several reconstruction techniques: FBP, SIRT, EM, TV-minimization, and NAG-LS.

        Args:
            sinogram (Tensor): Input sinogram of shape (1, A, D).
            A (ts.Operator): Tomosipo operator used for reconstruction.
            num_iterations_sirt (int): Iterations for SIRT algorithm.
            num_iterations_em (int): Iterations for EM algorithm.
            num_iterations_tv_min (int): Iterations for TV-minimization.
            num_iterations_nag_ls (int): Iterations for NAG-LS.
            lamda (float): Regularization parameter for TV-minimization.

        Returns:
            dict: Reconstructions from each method with keys: ['fbp', 'sirt', 'em', 'tv_min', 'nag_ls'].
        """
        #Ensure the sinogram has correct shape
        if sinogram.ndim == 4:
            sinogram = sinogram[0]  # From [1, 1, 1000, 513] or [10, 1, 1000, 513] → [1, 1000, 513]
        elif sinogram.ndim == 3 and sinogram.shape[0] > 1:
            sinogram = sinogram[0:1]  # From [10, 1000, 513] → [1, 1000, 513]

        # Filtered Backprojection (FBP) reconstruction
        rec_fbp = fbp(A, sinogram)
        #Simultaneous Iterative Reconstruction Technique (SIRT)
        rec_sirt = sirt(A, sinogram, num_iterations_sirt)
        # Expectation Maximization (EM) reconstruction
        rec_em = em(A, sinogram, num_iterations_em)
        #Total Variation regularized least squares using Chambolle-Pock algorithm
        rec_tv_min = tv_min2d(A, sinogram, lamda, num_iterations_tv_min)
        #Nesterov Accelerated Gradient for Least Squares reconstruction 
        rec_nag_ls = nag_ls(A, sinogram, num_iterations_nag_ls)

        return {"fbp": rec_fbp,
            "sirt": rec_sirt,
            "em": rec_em,
            "tv_min": rec_tv_min,
            "nag_ls": rec_nag_ls}




    def report_results_images(self, save_path, samples, num_iterations_sirt=100, num_iterations_em= 100, num_iterations_tv_min=100, num_iterations_nag_ls = 100, lamda = 0.0001):
        """
        Generates and saves visual comparisons of reconstructions across methods.

        For each selected test sample, displays the ground truth, model prediction,
        and results from classical methods (FBP, SIRT, EM, TV-Min, NAG-LS).

        Args:
            save_path (str): Prefix path to save output figures.
            samples (list[int]): Indices of test samples to visualize.
            num_iterations_sirt (int): Iterations for SIRT.
            num_iterations_em (int): Iterations for EM.
            num_iterations_tv_min (int): Iterations for TV minimization.
            num_iterations_nag_ls (int): Iterations for NAG-LS.
            lamda (float): TV regularization parameter.

        Raises:
            RuntimeError: If required prediction, GT or sinogram files are missing.

        Example:
            >>> model.report_results_images('out/visuals', samples=[0, 5, 10])
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
        elif not (self.sparse_view or self.single_bp) and not os.path.exists(f"{self.model_path}_noisy_sinograms.pt"):
            self._log("Noisy sinograms .pt files not found", level="error")
            return
        elif (self.sparse_view or self.single_bp) and not os.path.exists(f"{self.model_path}_sparse_sinograms.pt"):
            self._log("Sparse-view sinograms .pt files not found", level="error")
        else:
            predictions = torch.load(f"{self.model_path}_predictions_images.pt")
            ground_truths = torch.load(f"{self.model_path}_ground_truth_images.pt")
            if self.sparse_view or self.single_bp:
                sinograms = torch.load(f"{self.model_path}_sparse_sinograms.pt") 
            else:
                sinograms = torch.load(f"{self.model_path}_noisy_sinograms.pt")

        # calculate de A operator for tomosipo 
        A = self._get_operator()

        # generate and save all plots (model reconstructed image, gt image and classical methods reconstrucstions)
        for sample in samples:
            reconstructions_dict = self.other_ct_reconstruction(sinograms[sample], A, num_iterations_sirt, num_iterations_em, num_iterations_tv_min, num_iterations_nag_ls, lamda)
            plot_different_reconstructions(self.model_type, sample, reconstructions_dict, predictions[sample], ground_truths[sample], save_path)



    def report_results_table(self, save_path, test_path, max_len_test, num_iterations_sirt=100, num_iterations_em=100,
                         num_iterations_tv_min=100, num_iterations_nag_ls=100, lamda=0.0001, only_results = False):
        """
        Computes average PSNR and SSIM for classical reconstruction algorithms on the test set.

        If `only_results=True`, recomputes from raw test sinograms and GT using reconstruction algorithms.
        Otherwise, loads saved prediction and sinogram files from disk.

        Args:
            save_path (str): Path prefix for saving the CSV file.
            test_path (str): Test dataset path (required if only_results=True).
            max_len_test (int): Number of test samples (required if only_results=True).
            num_iterations_sirt (int): Iterations for SIRT algorithm.
            num_iterations_em (int): Iterations for EM algorithm.
            num_iterations_tv_min (int): Iterations for TV minimization.
            num_iterations_nag_ls (int): Iterations for NAG-LS.
            lamda (float): Regularization weight for TV minimization.
            only_results (bool): If True, re-evaluate directly from dataset instead of loading saved .pt files.

        Raises:
            ValueError: If required arguments are missing when only_results=True.

        Example:
            >>> model.report_results_table('results/classical', test_path='data/test', max_len_test=256, only_results=True)
        """

        metrics = {
            "Algorithm": ["FBP", "SIRT", "EM", "TV-Min", "NAG-LS"],
            "PSNR": [0] * 5,
            "SSIM": [0] * 5
        }

        if only_results:
            if test_path is None:
                raise ValueError("When only_results=True, test_path must be provided.")

            n_samples = max_len_test

            #changing paths parameters
            self.test_path = test_path
            self.max_len_test = max_len_test

            # get test data loader
            test_dataloader = self._get_test_dataloaders()
            test_dataloader = self.accelerator.prepare(test_dataloader)

            # calculate de A operator for tomosipo 
            A = self._get_operator()

            for batch in tqdm(test_dataloader):
                # send the input to the device
                ground_truth = batch['ground_truth']
                if self.sparse_view or self.single_bp:
                    sino = batch['sparse_sinogram']
                else:
                    sino = batch['noisy_sinogram']

                recon_dict = self.other_ct_reconstruction(sino, A, num_iterations_sirt=num_iterations_sirt, num_iterations_em=num_iterations_em, num_iterations_tv_min=num_iterations_tv_min, num_iterations_nag_ls=num_iterations_nag_ls, lamda=lamda)

                metrics["PSNR"][0] += compute_psnr_results(recon_dict["fbp"], ground_truth, 1)
                metrics["PSNR"][1] += compute_psnr_results(recon_dict["sirt"], ground_truth, 1)
                metrics["PSNR"][2] += compute_psnr_results(recon_dict["em"], ground_truth, 1)
                metrics["PSNR"][3] += compute_psnr_results(recon_dict["tv_min"], ground_truth, 1)
                metrics["PSNR"][4] += compute_psnr_results(recon_dict["nag_ls"], ground_truth, 1)

                metrics["SSIM"][0] += compute_ssim(recon_dict["fbp"], ground_truth, 1)
                metrics["SSIM"][1] += compute_ssim(recon_dict["sirt"], ground_truth, 1)
                metrics["SSIM"][2] += compute_ssim(recon_dict["em"], ground_truth, 1)
                metrics["SSIM"][3] += compute_ssim(recon_dict["tv_min"], ground_truth, 1)
                metrics["SSIM"][4] += compute_ssim(recon_dict["nag_ls"], ground_truth, 1)


        else:
            # File existence checks
            if not os.path.exists(f"{self.model_path}_ground_truth_images.pt"):
                self._log("Ground truth .pt files not found", level="error")
                return
            if not  (self.sparse_view or self.single_bp) and not os.path.exists(f"{self.model_path}_noisy_sinograms.pt"):
                self._log("Noisy sinograms .pt files not found", level="error")
                return
            if  (self.sparse_view or self.single_bp) and not os.path.exists(f"{self.model_path}_sparse_sinograms.pt"):
                self._log("Sparse-view sinograms .pt files not found", level="error")
                return
            
            # Load data
            ground_truths = torch.load(f"{self.model_path}_ground_truth_images.pt")
            if self.sparse_view or self.single_bp:
                sinograms = torch.load(f"{self.model_path}_sparse_sinograms.pt")
            else:
                sinograms = torch.load(f"{self.model_path}_noisy_sinograms.pt")
            n_samples = len(sinograms)

            # calculate de A operator for tomosipo 
            A = self._get_operator()

            for i in range(n_samples):
                gt_image = ground_truths[i][0] if ground_truths[i].dim() == 4 else ground_truths[i]
                recon_dict = self.other_ct_reconstruction( sinograms[i], A, num_iterations_sirt=num_iterations_sirt, num_iterations_em=num_iterations_em, num_iterations_tv_min=num_iterations_tv_min, num_iterations_nag_ls=num_iterations_nag_ls, lamda=lamda
                )

                metrics["PSNR"][0] += compute_psnr_results(recon_dict["fbp"], gt_image, 1)
                metrics["PSNR"][1] += compute_psnr_results(recon_dict["sirt"], gt_image, 1)
                metrics["PSNR"][2] += compute_psnr_results(recon_dict["em"], gt_image, 1)
                metrics["PSNR"][3] += compute_psnr_results(recon_dict["tv_min"], gt_image, 1)
                metrics["PSNR"][4] += compute_psnr_results(recon_dict["nag_ls"], gt_image, 1)

                metrics["SSIM"][0] += compute_ssim(recon_dict["fbp"], gt_image, 1)
                metrics["SSIM"][1] += compute_ssim(recon_dict["sirt"], gt_image, 1)
                metrics["SSIM"][2] += compute_ssim(recon_dict["em"], gt_image, 1)
                metrics["SSIM"][3] += compute_ssim(recon_dict["tv_min"], gt_image, 1)
                metrics["SSIM"][4] += compute_ssim(recon_dict["nag_ls"], gt_image, 1)

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
        Loads a pretrained model checkpoint from file.

        Restores model weights and sets the trained flag. Supports custom or default path.

        Args:
            path (str, optional): Path to model checkpoint. Defaults to `self.model_path`.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
            ValueError: If `self.model` is not initialized.

        Example:
            >>> model.load_model("checkpoints/unet_model.pt")
        """
        
        # we first need to check if we create correctly the model
        if self.model is None:
            raise ValueError("Model instance is not initialized. Set self.model before loading weights.")

        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            self._log(f"Model file not found: {path}", level="error")
            raise FileNotFoundError(f"Model file not found: {path}")

        
        self.accelerator.load_state(path)
        self.model.to(self.device)
        self._log(f"Model weights loaded from {path}")

        #change training state 
        self.trained = True