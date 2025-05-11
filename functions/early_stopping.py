import torch 

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How many epochs to wait after last improvement.
        debug (bool): If True, logs detailed messages.
        delta (float): Minimum change to qualify as an improvement.
        path (str): Path to save the best model.
        logger (logging.Logger, optional): Logger for output messages.
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
        self.logger = logger or logging.getLogger(__name__) 


    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.debug:
                self.logger.info(f"[EarlyStopping] EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                if self.debug:
                    self.logger.info("[EarlyStopping] Stopping early.")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.debug:
            self.logger.info(f"[EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss