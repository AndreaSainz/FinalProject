from accelerate import Accelerator
import os
import logging

class EarlyStopping:
    """
    Implements early stopping to halt training when the validation loss fails to improve.

    This mechanism helps prevent overfitting by monitoring the validation loss. If the loss 
    does not decrease after a defined number of consecutive epochs (`patience`), training is stopped early. 
    The model with the best (lowest) validation loss is automatically saved to disk using the accelerator.

    Parameters:
        patience (int): Optional. Number of epochs to wait after the last improvement before stopping (default: 10).
        debug (bool): Optional. If True, prints log messages to the console in addition to writing to the logger (default: False).
        delta (float): Optional. Minimum change in validation loss to qualify as an improvement (default: 0.0).
        path (str): Optional. Directory path to save the best model checkpoint (default: 'checkpoint_dir/').
        logger (logging.Logger): Optional. Logger instance to use for logging messages. If None, a default logger is created (default: None).
        accelerator (accelerate.Accelerator): Optional. Accelerator for managing distributed training and saving state (default: `Accelerator()`).

    Attributes:
        early_stop (bool): Flag indicating whether early stopping has been triggered.
        best_score (float or None): Best (lowest) negative validation loss observed so far.
        val_loss_min (float): Lowest validation loss encountered during training.
        counter (int): Number of consecutive epochs without improvement.

    Example:
        >>> from ct_reconstruction.callbacks import EarlyStopping
        >>> from accelerate import Accelerator
        >>> model = MyModel()
        >>> accelerator = Accelerator()

        >>> # Initialize early stopping
        >>> early_stopping = EarlyStopping(
        ...     patience=20,
        ...     debug=True,
        ...     delta=1e-5,
        ...     accelerator=accelerator
        ... )

        >>> for epoch in range(epochs):
        ...     train(model)  # Replace with actual training logic
        ...     val_loss = evaluate(model)  # Replace with actual evaluation logic
        ...     early_stopping(val_loss, model)

        ...     if early_stopping.early_stop:
        ...         print(f"Early stopping triggered at epoch {epoch}")
        ...         break
    """


    def __init__(self, patience=10, debug=False, delta=0, path='checkpoint_dir/', logger=None, accelerator = Accelerator()):
        self.patience = patience
        self.debug = debug
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.logger = (logger.getChild("EarlyStopping") if logger else logging.getLogger(__name__ + ".earlystopping")) 
        self.logger.propagate = False  # prevents logs from being sent to the parent logger
        self.accelerator = accelerator

        # create directory if it does not exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)


    def _log(self, msg):
        """
        Logs a message to the logger and optionally prints it to the console.

        Parameters:
            msg (str): The message to be logged.

        Notes:
            - Logs are always written to the logger.
            - Messages are printed to the console only if `debug` is True.
        """
        # Always log to file
        self.logger.info(msg)

        # Only print to console if debug mode is active
        if self.debug:
            print(msg)

    def __call__(self, val_loss, model):
        """
        Evaluates the current validation loss and decides whether to stop training.

        Parameters:
            val_loss (float): Current validation loss to evaluate.
            model (torch.nn.Module): The model being trained. Used for saving the state.

        Notes:
            - If validation loss has improved beyond `delta`, the model is saved.
            - If no improvement is observed for `patience` consecutive epochs, `early_stop` is set to True.
        """
        # convert loss to score (lower is better) : whatt we really want is to maximize 
        score = -val_loss

        # First epoch: save model and initialize best score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # No significant improvement
        elif score < self.best_score + self.delta:
            self.counter += 1
            self._log(f"[EarlyStopping] EarlyStopping counter: {self.counter} out of {self.patience}")
            # If patience exceeded, early stop will stop the training
            if self.counter >= self.patience:
                self._log(f"[EarlyStopping] Stopping early.")
                self.early_stop = True
        # when improvement found
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the model checkpoint if the validation loss has improved.

        Parameters:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model to save.

        Notes:
            - This method updates the internal best validation loss.
            - Only the main process in a distributed environment performs the actual saving.
            - This method is safe to call in multi-GPU environments. Only the main process will write to disk.
        """
        self._log(f"[EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        # Save model parameters
        self.accelerator.wait_for_everyone()
        
        #This prevents multiple processes from trying to save the same file at the same time
        if self.accelerator.is_main_process:
            self.accelerator.save_state(self.path)
        
        # Update the lowest loss observed
        self.val_loss_min = val_loss