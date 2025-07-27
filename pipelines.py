"""
Pipeline module for training, generating, and validating Transformer models.
This module contains the main functions to manage the complete learning workflow.
"""

# Standard library imports
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import time

# Third-party imports
import numpy as np
import optuna
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

# Local imports
from utils.EarlyStopping import EarlyStopping
from PommierDataset import (
    PommierDatasetDecoderOnly,
    collate_fn_decoder_only,
    DecoderOnlyDynamicPommierDataset
)
from transformer import TransformerDecoderOnly
from Validator import Validator, GPUOutOfMemoryError,ValidationError


def model_size_mb(model):
    """
    Calculates the size of the model in megabytes.

    Args:
        model: The PyTorch model whose size is to be calculated.

    Returns:
        float: Size of the model in megabytes.
    """
    total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return total_size / (1024 ** 2)



def create_config_file(file_path, config_dict):
    """
    Creates a JSON configuration file from a dictionary.

    Args:
        file_path (str or Path): Path to the file where the configuration will be saved.
        config_dict (dict): Dictionary containing the configuration parameters.

    Raises:
        IOError: If the file cannot be created or written.
        TypeError: If config_dict is not a dictionary.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("config_dict must be a dictionary")

    # Convert Path objects to strings for JSON serialization
    config_dict_serializable = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in config_dict.items()
    }

    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(config_dict_serializable, json_file, indent=4)
    except IOError as e:
        raise IOError(f"Unable to create the configuration file: {e}") from e

def train_decoder_only(config_dict, trial=None):
    """
    Trains a transformer model in decoder-only mode based on the given configuration.

    Args:
        config_dict (dict): Dictionary containing the configuration parameters.
        trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization.

    Returns:
        tuple: (trained model, path to the experiment folder, final validation loss, wandb run)
    """
    try:
        # Extract parameters from config_dict
        dataset_path = config_dict['dataset_path']
        seed = config_dict['seed']
        batch_size = config_dict['batch_size']
        val_split = config_dict['val_split']
        vocab_size = config_dict['vocab_size']
        padding_idx = config_dict['padding_idx']
        n_head = config_dict['n_head']
        d_model = config_dict['d_model']
        nb_layers = config_dict['nb_layers']
        lr = config_dict['lr']
        nb_epoch = config_dict['nb_epoch']
        dim_feedforward = config_dict['dim_feedforward']
        dynamic = config_dict['dynamic']
        scheduler_config = config_dict['scheduler']
        early_stopping_config = config_dict['early_stopping']
        continue_training = config_dict['continue_training']
        checkpoint_path = config_dict['checkpoint_path']
        auto_precision = config_dict['auto_precision']

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        # Device configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generate a timestamp for the experiment name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = (
            f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_TS-{timestamp}_optuna"
        )
        experiment_path = Path("experiments") / exp_name
        experiment_path.mkdir(parents=True, exist_ok=True)
        print(str(experiment_path / "config.json"))
        create_config_file(experiment_path / "config.json", config_dict)

        # Dataset creation
        static_dataset = PommierDatasetDecoderOnly(dataset_path)
        train_size = int(val_split * len(static_dataset))
        val_size = len(static_dataset) - train_size
        train_split, val_split = random_split(static_dataset, [train_size, val_size])

        if dynamic:
            vocab_to_id = {
                '<PAD>': 0, '<SOS>': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
                'DORMANT': 7, 'FLORAL': 8, 'LARGE': 9, 'MEDIUM': 10, 'SMALL': 11,
                'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16
            }
            dynamic_dataset = DecoderOnlyDynamicPommierDataset(
                vocab_to_id, 200000, 4, 70
            )
            train_loader = DataLoader(
                dynamic_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn_decoder_only
            )
        else:
            train_loader = DataLoader(
                train_split,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn_decoder_only
            )

        val_loader = DataLoader(
            val_split,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_decoder_only
        )

        # Model creation
        model = TransformerDecoderOnly(
            vocab_size=vocab_size,
            d_model=d_model,
            n_head=n_head,
            num_decoder_layers=nb_layers,
            padding_idx=padding_idx,
            dim_feedforward=dim_feedforward
        )
        model.to(device)

        # Load saved weights if continuing training
        if continue_training:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters(), lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr)

        # Scheduler creation (if applicable)
        scheduler = None
        if scheduler_config['name'] == "cyclical":
            scheduler = CyclicLR(optimizer, **scheduler_config['params'])
        elif scheduler_config['name'] == "ReduceOnPlatau":
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_config['params'])

        # Early stopping setup (if applicable)
        early_stopping = None
        if early_stopping_config['name'] == "patience":
            early_stopping = EarlyStopping(**early_stopping_config['params'])

        # Calculate the number of trainable parameters and model size
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = model_size_mb(model)

        # Initialize wandb for experiment tracking
        wandb_config = {
            "learning_rate": lr,
            "val_split": config_dict['val_split'],
            "architecture": exp_name,
            "dataset": "100 sample de chaque type",
            "batch_size": batch_size,
            "dimension_model": d_model,
            "number_of_heads": n_head,
            "epochs": nb_epoch,
            "dynamic": dynamic,
            "num_layers": nb_layers,
            "num_params": num_params,
            "dim_feedforward": dim_feedforward,
            "scheduler": scheduler_config['name'],
            "scheduler_params": (
                scheduler_config['params'] if scheduler_config['name'] != "None"
                else None
            ),
            "early_stopping": early_stopping_config['name'],
            "early_stopping_params": (
                early_stopping_config['params'] if early_stopping_config['name'] != "None"
                else None
            ),
            "auto_precision": auto_precision
        }
        wandb.init(
            name=exp_name,
            project="Topologie-Pommiers",
            config=wandb_config,
            mode="offline"
        )

        print(f"Nombre de paramètres : {num_params:,}")
        print(f"Le modèle occupe environ {size_mb:.2f} Mo en mémoire.")
        criterion_unweighted = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

        # Setup GradScaler if auto_precision is enabled
        scaler = torch.amp.GradScaler(device=device) if auto_precision else None

        # Training loop
        global_batch = 0
        best_val_loss = float('inf')
        val_losses = []  # Liste pour stocker les pertes de validation
        for epoch in tqdm(range(nb_epoch), colour="green"):
            model.train()
            total_train_loss_unweighted = 0
            for input_seq, target_seq, _ in tqdm(
                train_loader,
                desc=f"Epoch {epoch} - Train",
                colour="red"
            ):
                try:
                    if device == "cuda":
                        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2  # en Mo
                        # wandb.log({"gpu_memory_allocated_MB": mem_alloc}, step=global_batch)
                    input_seq = input_seq.to(device)
                    target_seq = target_seq.to(device)
                    padding_mask = (input_seq == 0).to(torch.bool).to(model.device)

                    if auto_precision:
                        with torch.amp.autocast(device_type=device):
                            logits = model(input_seq, padding_mask)
                            logits_trim = logits[:, 2:, :]
                            targets_trim = target_seq[:, 2:]
                            logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                            target_flat = targets_trim.reshape(-1)
                            loss_unweighted = criterion_unweighted(logits_flat, target_flat)
                        optimizer.zero_grad()
                        scaler.scale(loss_unweighted).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(input_seq, padding_mask)
                        logits_trim = logits[:, 2:, :]
                        targets_trim = target_seq[:, 2:]
                        logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                        target_flat = targets_trim.reshape(-1)
                        loss_unweighted = criterion_unweighted(logits_flat, target_flat)
                        optimizer.zero_grad()
                        loss_unweighted.backward()
                        optimizer.step()

                    if scheduler is not None and scheduler_config['name'] == "cyclical":
                        scheduler.step()

                    total_train_loss_unweighted += loss_unweighted.item()

                    # Log the learning rate at each batch
                    current_lr = optimizer.param_groups[0]['lr']
                    # wandb.log({"batch_learning_rate": current_lr}, step=global_batch)
                    global_batch += 1

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        raise GPUOutOfMemoryError(f"Erreur de mémoire GPU lors de l'entraînement. Configuration: {config_dict}")
                    raise e

            model.eval()
            total_eval_loss_unweighted = 0
            with torch.no_grad():
                for input_seq, target_seq, loss_mask in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch} - Val",
                    colour="yellow"
                ):
                    try:
                        input_seq = input_seq.to(device)
                        target_seq = target_seq.to(device)
                        loss_mask = loss_mask.to(device)
                        padding_mask = (input_seq == 0).to(torch.bool).to(model.device)
                        logits = model(input_seq, padding_mask)
                        logits_trim = logits[:, 2:, :]
                        targets_trim = target_seq[:, 2:]
                        logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                        target_flat = targets_trim.reshape(-1)
                        loss_unweighted = criterion_unweighted(logits_flat, target_flat)
                        total_eval_loss_unweighted += loss_unweighted.item()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            raise GPUOutOfMemoryError(f"Erreur de mémoire GPU lors de la validation. Configuration: {config_dict}")
                        raise e

            avg_train_loss_unweighted = total_train_loss_unweighted / len(train_loader)
            avg_val_loss_unweighted = total_eval_loss_unweighted / len(val_loader)

            val_losses.append(avg_val_loss_unweighted)  # Stockage de la perte de validation

            # If in an Optuna trial, record the validation loss at each epoch
            if trial is not None:
                trial.report(avg_val_loss_unweighted, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Update the best validation loss
                best_val_loss = min(best_val_loss, avg_val_loss_unweighted)

            wandb.log({
                "train_loss_epochs": avg_train_loss_unweighted,
                "val_loss_epochs": avg_val_loss_unweighted
            })
            tqdm.write(
                f"[INFO] Epoch {epoch} : train loss unweighted = {avg_train_loss_unweighted:.4f}, "
                f"val loss unweighted = {avg_val_loss_unweighted:.4f}"
            )
            if scheduler is not None and scheduler_config['name'] == "ReduceOnPlatau":
                scheduler.step(avg_val_loss_unweighted)

            if early_stopping:
                early_stopping(avg_val_loss_unweighted, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, experiment_path / "model_state.pth")

        # Calculate the average over the last 20 epochs
        last_20_epochs_val_loss = sum(val_losses[-20:]) / min(20, len(val_losses))

        # If in an Optuna trial, use the average of the last 20 epochs
        final_val_loss = last_20_epochs_val_loss

        return model, experiment_path, final_val_loss, wandb.run

    except GPUOutOfMemoryError as e:
        print(f"[ERROR] {e}")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"[ERROR] Erreur inattendue lors de l'entraînement: {e}")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        return None

def find_wandb_run_path(run_id):
    """
    Finds the path of the wandb folder containing the specified run ID.

    Args:
        run_id: The ID of the wandb run.

    Returns:
        str: The full path of the run folder.
    """
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        raise FileNotFoundError("Le dossier wandb n'existe pas")

    for run_dir in wandb_dir.iterdir():
        if run_dir.is_dir() and str(run_id) in run_dir.name:
            return str(run_dir)

    raise FileNotFoundError(f"Aucun dossier trouvé contenant l'ID {run_id}")

def train_generate_validate_pipeline(config_dict, trial=None, sync_wandb=False):
    """
    Pipeline to train, generate, and validate a model using a configuration provided as a dictionary.

    Args:
        config_dict (dict): Dictionary containing the configuration parameters.
        trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization.
        sync_wandb (bool, optional): If True, synchronizes data with wandb online at the end of the run.

    Returns:
        Validator: Instance of the Validator class used to generate and validate data.
    """
    # Train the model
    st = time()
    model, experiment_path, final_val_loss, wandb_run = train_decoder_only(config_dict, trial)
    et = time()
    print(f"[INFO] le temps en heures pour l'entraînement est de : {(et-st)/3600}")

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device=device)
    vocab_to_id = {
        '<PAD>': 0, '<SOS>': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
        'DORMANT': 7, 'FLORAL': 8, 'LARGE': 9, 'MEDIUM': 10, 'SMALL': 11,
        'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16
    }

    # Initialize the validator
    validator = Validator(model, device, token_to_id=vocab_to_id, validation_folder_path=experiment_path)
    st = time()
    try:
        validator.generate_data(10000, experiment_path / "generated_dataset.csv", end_toks_list=[7, 8, 9, 10, 11])
    except ValidationError as e:
        print(f"[ERROR] {e}")
        return None

    print("Génération terminée")
    return None
    et = time()
    print(f"[INFO] le temps en secondes pour la génération est de : {et-st}")
    validator.load_data("dataset/markov_python_generated_dataset10000.csv")
    st = time()
    validator.validation_pipeline("generated_dataset.csv", "generated_dataset_validation_stats.json", windows=False)
    et = time()
    print(f"[INFO] le temps en minutes pour la validation est de : {(et-st)/60}")

    # Read validation statistics
    with open(experiment_path / "generated_dataset_validation_stats.json", "r", encoding='utf-8') as f:
        stats = json.load(f)

    # Calculate the number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Store the validation loss and number of parameters in the validator for return
    stats["final_val"] = final_val_loss
    stats["num_params"] = num_params

    with open(experiment_path / "generated_dataset_validation_stats.json", "w", encoding='utf-8') as f:
        json.dump(stats, f)
    # Calculate global metrics
    metrics = validator.compute_metrics(stats)

    # If in an Optuna trial, record all metrics
    if trial is not None:
        # Record the final validation loss (average over the last 20 epochs)
        trial.set_user_attr('final_val', final_val_loss)
        trial.set_user_attr('num_params', num_params)

        # Record validation metrics
        for metric_name, (mean, std) in metrics.items():
            if mean is not None:  # On n'enregistre que les métriques qui ont des valeurs
                trial.set_user_attr(f'{metric_name}_mean', mean)
                trial.set_user_attr(f'{metric_name}_std', std)

    # Close the wandb run with or without online synchronization
    if sync_wandb:
        # First, finish the run in offline mode
        wandb_run.finish(quiet=True)
        # Then synchronize online using the unique run ID
        print("[INFO] Synchronisation des données wandb en ligne...")
        try:
            run_path = find_wandb_run_path(wandb_run.id)
            subprocess.run(
                ["wandb", "sync", run_path],
                check=True,
                capture_output=True,
                text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[WARNING] Erreur lors de la synchronisation wandb: {e}")
    else:
        wandb_run.finish(quiet=True)

    return stats["final_val"]
