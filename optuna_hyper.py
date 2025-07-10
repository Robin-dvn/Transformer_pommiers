import optuna
import numpy as np
import pandas as pd
import os
import sys
import json
import subprocess
from pipelines import train_generate_validate_pipeline


def objective(trial):
    '''
    Objective function for Optuna to optimize the hyperparameters
    '''
    # Configuration de base
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

    # Hyperparamètres à optimiser
    base_config['d_model'] = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    base_config['n_head'] = trial.suggest_categorical("n_head", [1, 2, 4, 8, 16])
    base_config['nb_layers'] = trial.suggest_int("nb_layers", 1, 27)
    base_config['dim_feedforward'] = trial.suggest_categorical("dim_feedforward", [64,128,256,512,1024])

    # Exécution de la pipeline avec le trial
    final_val = train_generate_validate_pipeline(base_config, trial, sync_wandb=True)
    
    if final_val is None:
        return float('inf')  # Retourne une valeur infinie si la génération échoue

    # Calcul de la somme normalisée des métriques
    return final_val

if __name__ == "__main__":
    # Création de l'étude Optuna avec pruning
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,      # Nombre de trials avant de commencer le pruning
        n_warmup_steps=300,       # Nombre d'étapes avant de commencer le pruning
        interval_steps=1,        # Intervalle entre les évaluations de pruning
        n_min_trials=10,          # Nombre minimum de trials pour le pruning
 
    )
    study = optuna.create_study(
        direction="minimize",
        study_name="transformer_optimization",
        pruner=pruner
    )
    
    # Optimisation
    study.optimize(
        objective,
        n_trials=28,  # Nombre d'essais à effectuer
        n_jobs=1,     # Nombre de jobs parallèles
        show_progress_bar=True,
        gc_after_trial=True  # Nettoie la mémoire après chaque essai
    )
    
    # Affichage des résultats
    print("\nMeilleurs paramètres trouvés :")
    print(study.best_params)
    print("\nMeilleure valeur trouvée :")
    print(study.best_value)
    
    # Sauvegarde des résultats
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "n_pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
    }
    
    with open("experiments/optuna_results.json", "w") as f:
        json.dump(results, f, indent=4)
