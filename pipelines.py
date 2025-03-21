from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from collections import Counter
import numpy as np
from datetime import datetime
from time import time
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from EarlyStopping import EarlyStopping

# Importer les modules nécessaires (assure-toi que ces fichiers existent)
from PommierDataset import PommierDatasetDecoderOnly, DynamicPommierDataset, collate_fn_decoder_only, DecoderOnlyDynamicPommierDataset
from transformer import TransformerDecoderOnly  # Notre modèle décodeur-only
from Validator import Validator
from ValidationError import ValidationError

def model_size_mb(model):
    total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return total_size / (1024 ** 2)

def calculate_class_weights(dataset, vocab_size):
    # Compter les occurrences de chaque jeton
    token_counts = Counter()
    for data in dataset:
        input_seq, target_seq, _ = data
        seq = input_seq.tolist() + [target_seq.tolist()[-1]]
        token_counts.update(seq)

    # Calculer les poids en utilisant l'inverse de la fréquence
    total_tokens = sum(token_counts.values())
    print(f"Total tokens: {total_tokens}")
    tokens_counts = {token_id: count for token_id, count in token_counts.items()}
    frequencies = {token_id: count / total_tokens for token_id, count in token_counts.items()}
    print(f"Frequencies: {frequencies}")
    print(f"Token counts: {token_counts}")
    class_weights = {token_id: total_tokens / count for token_id, count in token_counts.items()}
    print(f"Class weights: {class_weights}")

    # Convertir en tenseur PyTorch
    weights_tensor = torch.zeros(vocab_size)
    for token_id, weight in class_weights.items():
        weights_tensor[token_id] = weight

    # Normaliser les poids
    weights_tensor = weights_tensor / weights_tensor.sum()

    return weights_tensor

def create_config_file(file_path, config_dict):
    """
    Create a JSON configuration file from a dictionary.

    Args:
        file_path (str or Path): Path to the file where the configuration will be saved.
        config_dict (dict): Dictionary containing configuration parameters.
    """
    # Convertir les objets Path en chaînes de caractères
    config_dict_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items()}

    with open(file_path, 'w') as json_file:
        json.dump(config_dict_serializable, json_file, indent=4)


def train_decoder_only(config_dict):
    """
    Entraîne un modèle transformer en mode decoder-only selon la configuration passée.

    Args:
        config_dict (dict): Dictionnaire contenant les paramètres de configuration. Clés attendues :
            - dataset_path (str ou Path) : Chemin vers le dataset.
            - seed (int) : Seed pour la reproductibilité.
            - batch_size (int) : Taille des batches pour l'entraînement et la validation.
            - val_split (float) : Fraction du dataset utilisée pour la validation.
            - vocab_size (int) : Taille du vocabulaire.
            - padding_idx (int) : Indice de padding.
            - n_head (int) : Nombre de têtes d'attention.
            - d_model (int) : Dimension du modèle.
            - nb_layers (int) : Nombre de couches du décodeur.
            - lr (float) : Taux d'apprentissage.
            - nb_epoch (int) : Nombre d'époques d'entraînement.
            - dim_feedforward (int) : Dimension du réseau feedforward.
            - dynamic (bool) : Indique si un dataset dynamique est utilisé.
            - scheduler (dict) : Dictionnaire contenant :
                - name (str) : Type de scheduler parmi ["cyclical", "ReduceOnPlatau", "None"].
                - params (dict) : Paramètres spécifiques au scheduler.
            - early_stopping (dict) : Dictionnaire contenant :
                - name (str) : Type d'early stopping parmi ["patience", "None"].
                - params (dict) : Paramètres spécifiques à l'early stopping.
            - continue_training (bool) : Indique si l'entraînement doit reprendre depuis un checkpoint.
            - checkpoint_path (str ou Path) : Chemin vers le checkpoint si l'entraînement est repris.
            - auto_precision (bool): Si True, active la précision automatique (torch.cuda.amp).
    Retour:
        tuple: (modèle entraîné, chemin vers le dossier de l'expérience)
    """
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
    auto_precision = config_dict['auto_precision'] # nouveau paramètre

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate a timestamp for the experiment name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_TS-{timestamp}"
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
        dynamic_dataset = DecoderOnlyDynamicPommierDataset(vocab_to_id, 200000, 4, 70)
        train_loader = DataLoader(dynamic_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_decoder_only)
    else:
        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_decoder_only)

    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_decoder_only)

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

    # Initialize wandb for experiment tracking, en ajoutant le paramètre auto_precision
    wandb.init(
        name=exp_name,
        project="Topologie-Pommiers",
        config={
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
            "scheduler_params": scheduler_config['params'] if scheduler_config['name'] != "None" else None,
            "early_stopping": early_stopping_config['name'],
            "early_stopping_params": early_stopping_config['params'] if early_stopping_config['name'] != "None" else None,
            "auto_precision": auto_precision
        },
        mode="offline"
    )

    print(f"Nombre de paramètres : {num_params:,}")
    print(f"Le modèle occupe environ {size_mb:.2f} Mo en mémoire.")

    # Calculate class weights
    class_weights = calculate_class_weights(static_dataset, vocab_size)
    class_weights = class_weights.to(device)
    criterion_weighted = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=padding_idx)
    criterion_unweighted = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    # Setup GradScaler si auto_precision est activé
    scaler = torch.amp.GradScaler(device=device) if auto_precision else None

    # Training loop
    global_batch = 0
    for epoch in tqdm(range(nb_epoch), colour="green"):
        model.train()
        total_train_loss_weighted = 0
        total_train_loss_unweighted = 0
        for input_seq, target_seq, _ in tqdm(train_loader, desc=f"Epoch {epoch} - Train", colour="red"):

            if device == "cuda":
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**2  # en Mo
                wandb.log({"gpu_memory_allocated_MB": mem_alloc}, step=global_batch)
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
                # with torch.no_grad():
                    # loss_weighted = criterion_weighted(logits_flat, target_flat)
            else:
                logits = model(input_seq, padding_mask)
                logits_trim = logits[:, 2:, :]
                targets_trim = target_seq[:, 2:]
                logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                target_flat = targets_trim.reshape(-1)
                loss_unweighted = criterion_unweighted(logits_flat, target_flat)
                # with torch.no_grad():
                    # loss_weighted = criterion_weighted(logits_flat, target_flat)
                optimizer.zero_grad()
                loss_unweighted.backward()
                optimizer.step()

            if scheduler is not None and scheduler_config['name'] == "cyclical":
                scheduler.step()

            total_train_loss_unweighted += loss_unweighted.item()
            # total_train_loss_weighted += loss_weighted.item()

            # Log le learning rate à chaque batch
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"batch_learning_rate": current_lr}, step=global_batch)
            global_batch += 1

        model.eval()
        total_eval_loss_weighted = 0
        total_eval_loss_unweighted = 0
        with torch.no_grad():
            for input_seq, target_seq, loss_mask in tqdm(val_loader, desc=f"Epoch {epoch} - Val", colour="yellow"):
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
                # loss_weighted = criterion_weighted(logits_flat, target_flat)
                total_eval_loss_unweighted += loss_unweighted.item()
                # total_eval_loss_weighted += loss_weighted.item()

        avg_train_loss_unweighted = total_train_loss_unweighted / len(train_loader)
        # avg_train_loss_weighted = total_train_loss_weighted / len(train_loader)
        avg_val_loss_unweighted = total_eval_loss_unweighted / len(val_loader)
        # avg_val_loss_weighted = total_eval_loss_weighted / len(val_loader)

        wandb.log({
            "train_loss_epochs": avg_train_loss_unweighted,
            # "train_loss_weighted_epochs": avg_train_loss_weighted,
            "val_loss_epochs": avg_val_loss_unweighted,
            # "val_loss_weighted_epochs": avg_val_loss_weighted
        })
        tqdm.write(f"[INFO] Epoch {epoch} : train loss unweighted = {avg_train_loss_unweighted:.4f}, val loss unweighted = {avg_val_loss_unweighted:.4f}")
        # tqdm.write(f"[INFO] Epoch {epoch} : train loss unweighted = {avg_train_loss_unweighted:.4f}, train loss weighted = {avg_train_loss_weighted:.4f}, val loss unweighted = {avg_val_loss_unweighted:.4f}, val loss weighted = {avg_val_loss_weighted:.4f}")
        if scheduler is not None and scheduler_config['name'] == "ReduceOnPlatau":
            scheduler.step(avg_val_loss_unweighted)

        if early_stopping:
            early_stopping(avg_val_loss_unweighted,model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, experiment_path / "model_state.pth")

    wandb.finish()

    return model, experiment_path


def train_generate_validate_pipeline(config_dict):
    """
    Pipeline pour entraîner, générer et valider un modèle en utilisant une configuration passée en dictionnaire.
    
    Args:
        config_dict (dict): Dictionnaire contenant les paramètres de configuration. Clés attendues :
            - dataset_path (str ou Path) : Chemin vers le dataset.
            - seed (int) : Seed pour la reproductibilité.
            - batch_size (int) : Taille des batches pour l'entraînement et la validation.
            - val_split (float) : Fraction du dataset utilisée pour la validation.
            - vocab_size (int) : Taille du vocabulaire.
            - padding_idx (int) : Indice de padding.
            - n_head (int) : Nombre de têtes d'attention.
            - d_model (int) : Dimension du modèle.
            - nb_layers (int) : Nombre de couches du décodeur.
            - lr (float) : Taux d'apprentissage.
            - nb_epoch (int) : Nombre d'époques d'entraînement.
            - dim_feedforward (int) : Dimension du réseau feedforward.
            - dynamic (bool) : Indique si un dataset dynamique est utilisé.
            - scheduler (dict) : Dictionnaire contenant :
                - name (str) : Type de scheduler parmi ["cyclical", "ReduceOnPlatau", "None"].
                - params (dict) : Paramètres spécifiques au scheduler.
            - early_stopping (dict) : Dictionnaire contenant :
                - name (str) : Type d'early stopping parmi ["patience", "None"].
                - params (dict) : Paramètres spécifiques à l'early stopping.
            - continue_training (bool) : Indique si l'entraînement doit reprendre depuis un checkpoint.
            - checkpoint_path (str ou Path) : Chemin vers le checkpoint si l'entraînement est repris.
            - auto_precision (bool): Si True, active la précision automatique (torch.cuda.amp).
    
    Returns:
        Validator : Instance de la classe Validator utilisée pour générer et valider les données.
    """
    # Train the model
    st = time()
    model, experiment_path = train_decoder_only(config_dict)
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
    et = time()
    print(f"[INFO] le temps en secondes pour la génération est de : {et-st}")
    validator.load_data("out/markov_python_generated_dataset10000.csv")
    st = time()
    validator.validation_pipeline("generated_dataset.csv", "generated_dataset_validation_stats.json",windows = False)
    et = time()
    print(f"[INFO] le temps en minutes pour la validation est de : {(et-st)/60}")

    # validator.plot_stats_graph([experiment_path / "generated_dataset_validation_stats.json"])

    return validator
