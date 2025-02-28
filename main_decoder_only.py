from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from collections import Counter

# Importer les modules nécessaires (assure-toi que ces fichiers existent)
from PommierDataset import PommierDatasetDecoderOnly, DynamicPommierDataset, collate_fn_decoder_only
from transformer import TransformerDecoderOnly  # Notre modèle décodeur-only

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

if __name__ == "__main__":

    dataset_name = "100 sample de chaque type"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 512
    VAL_SPLIT = 0.8
    VOCAB_SIZE = 17  # Assure-toi que ça correspond à ton mapping
    PADDING_IDX = 0
    N_HEAD = 4
    D_MODEL = 32
    NB_LAYERS = 15
    LR = 5e-5
    NB_EPOCH = 40
    DYNAMIC = False  # Change à True si tu utilises DynamicPommierDataset
    exp_name = f"DecoderOnly_{D_MODEL}_layers_{NB_LAYERS}_epochs_{NB_EPOCH}"

    dataset_path = "out/markov_python_generated_dataset10000.csv"
    dataset = PommierDatasetDecoderOnly(dataset_path)

    # Calculer les poids des classes
    class_weights = calculate_class_weights(dataset, VOCAB_SIZE)

    train_size = int(VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_decoder_only)
    val_loader   = DataLoader(val_split,   batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_decoder_only)

    model = TransformerDecoderOnly(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_head=N_HEAD,
                                   num_decoder_layers=NB_LAYERS, padding_idx=PADDING_IDX)
    model.to(device)

    # Charger les poids sauvegardés
    continue_training = False
    if continue_training:
        checkpoint_path = "path/to/your/saved/model.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), LR)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = optim.Adam(model.parameters(), LR)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = model_size_mb(model)
    wandb.init(
        name=exp_name,
        project="Topologie-Pommiers",
        config={
            "learning_rate": LR,
            "Val split": VAL_SPLIT,
            "architecture": exp_name,
            "dataset": dataset_name,
            "batch size": BATCH_SIZE,
            "Dimension model": D_MODEL,
            "Number of heads": N_HEAD,
            "epochs": NB_EPOCH,
            "dynamic": DYNAMIC,
            "num_layers": NB_LAYERS,
            "num_params": num_params
        },
        mode="offline"
    )

    print(f"Nombre de paramètres : {num_params:,}")
    print(f"Le modèle occupe environ {size_mb:.2f} Mo en mémoire.")
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=PADDING_IDX)

    for epoch in tqdm(range(NB_EPOCH), colour="green"):
        model.train()
        total_train_loss = 0
        for input_seq, target_seq, loss_mask in tqdm(train_loader, desc=f"Epoch {epoch} - Train", colour="red"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            loss_mask = loss_mask.to(device)
            padding_mask = (input_seq == 0).to(torch.bool).to(model.device)

            logits = model(input_seq, padding_mask)  # (batch, seq_len, vocab_sizie)
            logits_trim = logits[:, 2:, :]    # on ignore les 2 premiers tokens
            targets_trim = target_seq[:, 2:]
            logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
            target_flat = targets_trim.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})

        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for input_seq, target_seq, loss_mask in tqdm(val_loader, desc=f"Epoch {epoch} - Val", colour="yellow"):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                loss_mask = loss_mask.to(device)
                padding_mask = (input_seq == 0).to(torch.bool).to(model.device)
                logits = model(input_seq, padding_mask)  # (batch, seq_len, vocab_size)

                logits_trim = logits[:, 2:, :]    # on ignore les 2 premiers tokens
                targets_trim = target_seq[:, 2:]
                logits_flat = logits_trim.reshape(-1, logits_trim.size(-1))
                target_flat = targets_trim.reshape(-1)

                loss = criterion(logits_flat, target_flat)
                total_eval_loss += loss.item()
                wandb.log({"val_loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_eval_loss / len(val_loader)
        print(f"[INFO] Epoch {epoch} : train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")

    # Sauvegarder le modèle et l'optimiseur
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"DecoderOnly_{D_MODEL}_layers_{NB_LAYERS}_epochs_{NB_EPOCH}.pth")

    wandb.finish()
