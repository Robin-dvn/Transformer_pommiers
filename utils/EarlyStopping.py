"""
This module defines an early stopping handler for training neural networks.
"""

import numpy as np
import torch
import os

class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.

    This class monitors the validation loss and stops training when the loss
    hasn't improved for a specified number of epochs (patience).
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Initialize early stopping parameters.

        Args:
            patience (int): Number of epochs to wait after validation loss stops improving.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_loss, model):
        """
        Check if early stopping should be triggered.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save the model checkpoint.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save.
        """
        if self.verbose:
            print(f'Saving model checkpoint with val_loss: {val_loss:.6f}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
