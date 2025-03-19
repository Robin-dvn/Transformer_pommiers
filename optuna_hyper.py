import optuna
import numpy as np
import pandas as pd
import os
import sys
import json
import subprocess

def create_trial_folder(trial):
    """ Crée un dossier unique pour chaque essai. """
    trial_id = f"trial_{trial.number:03d}"  # Ex: trial_001, trial_002
    trial_dir = os.path.join("experiments", trial_id)

    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def objective(trial):
    '''
    Objective function for Optuna to optimize the hyperparameters
    '''

    D_MODEL = trial.suggest_categorical("D_MODEL", [32, 64, 128, 256, 512, 1024, 2048])


    valid_nb_heads = [x for x in range(1, D_MODEL + 1) if D_MODEL % x == 0]
    NB_HEAD = trial.suggest_categorical("NB_HEAD", valid_nb_heads)
    
    valid_ff_dims = [x for x in [128, 256, 512, 1024, 2048, 4096, 8192] if x >= 2 * D_MODEL]
    FF_DIM = trial.suggest_categorical("FF_DIM", valid_ff_dims)

    NB_LAYERS = trial.suggest_int("NB_LAYERS", 1, 50)  # Exploration entre 1 et 50
    NB_EPOCHS = trial.suggest_int("NB_EPOCHS", 100, 600, step=50)  # Entre 10 et 100 avec un pas de 10
    BATCHSIZE = trial.suggest_int("BATCHSIZE", 64, 2048, log=True)  # Échelle logarithmique (utile pour batch size)

    BASE_LR = trial.suggest_float("BASE_LR", 1e-5, 1e-1, log=True)
    MAX_LR = trial.suggest_float("MAX_LR", BASE_LR, 10 * BASE_LR, log=True)
    STEP_SIZE = trial.suggest_int("STEP_SIZE", 5, 50)
    MODE = trial.suggest_categorical("MODE", ["TRIANGULAR", "TRIANGULAR2", "EXP_RANGE"])

    print(trial.params)

    return 2





if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
