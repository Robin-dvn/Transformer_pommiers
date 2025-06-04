import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Initialise l'early stopping.

        Args:
            patience (int): Nombre d'époques à attendre après que la perte ait cessé de s'améliorer.
            verbose (bool): Si True, imprime un message à chaque amélioration.
            delta (float): Changement minimal à considérer comme une amélioration.
            path (str): Chemin pour sauvegarder le checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """
        Vérifie si l'early stopping doit être déclenché.

        Args:
            val_loss (float): La perte de validation actuelle.
            model (torch.nn.Module): Le modèle à sauvegarder.
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
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Sauvegarde le modèle lorsque la perte de validation s'améliore.

        Args:
            val_loss (float): La perte de validation actuelle.
            model (torch.nn.Module): Le modèle à sauvegarder.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
