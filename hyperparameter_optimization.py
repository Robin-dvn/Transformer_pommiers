"""
Hyperparameter Optimization Script for Transformer Models.

This script uses Optuna to optimize hyperparameters for training a Transformer model.
It defines the search space, executes trials, and saves the best results.


Functions:
    - objective: Defines the objective function for Optuna.

Execution:
    Run this script directly to start the optimization process.

"""
import optuna
import numpy as np
import pandas as pd
import os
import sys
import json
import subprocess
from pipelines import train_generate_validate_pipeline


def objective(trial):
    """
    Objective function for Optuna to optimize the hyperparameters.

    This function defines the search space for hyperparameters and evaluates
    the performance of the model using the `train_generate_validate_pipeline` function.

    Returns:
        float: The final validation loss or infinity if the generation fails.
    """
    # Base configuration
    base_config = {
        'dataset_path': "out/markov_python_generated_dataset10000.csv",
        'seed': 42,
        'batch_size': 512,
        'val_split': 0.8,
        'vocab_size': 17,
        'padding_idx': 0,
        'nb_epoch': 500,
        'lr': 5e-5,
        'dynamic': True,
        'scheduler': {
            'name': 'None',
            'params': {}
        },
        'early_stopping': {
            'name': 'None',
            'params': {}
        },
        'continue_training': False,
        'checkpoint_path': None,
        'auto_precision': False
    }

    # Hyperparameters to optimize
    base_config['d_model'] = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    base_config['n_head'] = trial.suggest_categorical("n_head", [1, 2, 4, 8, 16])
    base_config['nb_layers'] = trial.suggest_int("nb_layers", 1, 27)
    base_config['dim_feedforward'] = trial.suggest_categorical("dim_feedforward", [64,128,256,512,1024])

    # Execute the pipeline with the trial
    final_val = train_generate_validate_pipeline(base_config, trial, sync_wandb=True)

    if final_val is None:
        return float('inf')  # Returns infinity if the generation fails

    # Calculate the normalized sum of metrics
    return final_val

if __name__ == "__main__":
    # Create the Optuna study with pruning
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,      # Number of trials before starting pruning
        n_warmup_steps=300,       # Number of steps before starting pruning
        interval_steps=1,         # Interval between pruning evaluations
        n_min_trials=10,          # Minimum number of trials for pruning

    )

    study = optuna.create_study(
        direction="minimize",
        study_name="transformer_optimization",
        pruner=pruner
    )

    # Optimization
    study.optimize(
        objective,
        n_trials=50,  # Number of trials to perform
        n_jobs=1,     # Number of parallel jobs
        show_progress_bar=True,
        gc_after_trial=True  # Cleans memory after each trial
    )

    # Display results
    print("\nBest parameters found:")
    print(study.best_params)
    print("\nBest value found:")
    print(study.best_value)

    # Save results
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "n_pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
    }
    }

    with open("experiments/optuna_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
