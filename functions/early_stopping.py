import torch 

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How many epochs to wait after last improvement.
        verbose (bool): If True, prints messages when improvement happens.
        delta (float): Minimum change to count as improvement.
        path (str): Where to save the best model.
    """
    def __init__(self, patience=10, debug=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.debug = debug
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.debug:
                print(f"[INFO] EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.debug:
            print(f"[INFO] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss