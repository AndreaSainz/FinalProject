import torch 

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    This mechanism helps prevent overfitting by stopping training if the validation loss
    does not improve after a specified number of consecutive epochs (patience).
    The best model (with lowest validation loss) is saved automatically.

    Args:
        patience (int): Number of epochs to wait after the last improvement before stopping.
        debug (bool): If True, logs detailed status messages.
        delta (float): Minimum change in validation loss to qualify as an improvement.
        path (str): File path to save the model with the best validation performance.
        logger (logging.Logger, optional): Custom logger for output. Defaults to module-level logger.

    Attributes:
        early_stop (bool): Flag indicating whether training should be stopped early.
    """

    def __init__(self, patience=10, debug=False, delta=0, path='checkpoint.pth', logger=None):
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

    def _log(self, msg):
        """
        Logs a message to the logger's file and optionally prints it to the console.

        This method ensures that:
        - All messages are always logged to the file through the logger.
        - Messages are printed to the console only if debug mode is enabled.

        Args:
            msg (str): The message to be logged.
        """
        # Always log to file
        self.logger.info(msg)

        # Only print to console if debug mode is active
        if self.debug:
            print(msg)

    def __call__(self, val_loss, model):
        """
        Call this method at the end of each validation phase.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        # Convert loss to score (lower is better)
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
        # whe improvement found
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the current model if the validation loss has improved.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save.
        """
        self._log(f"[EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        # Save model parameters
        torch.save(model.state_dict(), self.path)
        
        # Update the lowest loss observed
        self.val_loss_min = val_loss