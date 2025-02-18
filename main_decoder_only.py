from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# Importer les modules nécessaires (assure-toi que ces fichiers existent)
from DatasetCreator import DatasetCreator
from PommierDataset import PommierDatasetDecoderOnly, DynamicPommierDataset, collate_fn_decoder_only
from transformer import TransformerDecoderOnly  # Notre modèle décodeur-only

def model_size_mb(model):
    total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return total_size / (1024 ** 2)

if __name__ == "__main__":
    exp_name = "Transformer Decoder Only"
    dataset_name = "100 sample de chaque type"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 512
    VAL_SPLIT = 0.8
    VOCAB_SIZE = 17  # Assure-toi que ça correspond à ton mapping
    PADDING_IDX = 0
    N_HEAD = 1
    D_MODEL = 16
    LR = 5e-5
    NB_EPOCH = 12
    DYNAMIC = False  # Change à True si tu utilises DynamicPommierDataset

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
        }
    )


    dataset_path = "out/sequence_analysis_generated_dataset10000.csv"
    dataset = PommierDatasetDecoderOnly(dataset_path)

    train_size = int(VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_decoder_only)
    val_loader   = DataLoader(val_split,   batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_decoder_only)

    model = TransformerDecoderOnly(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_head=N_HEAD,
                                   num_decoder_layers=3, padding_idx=PADDING_IDX)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = model_size_mb(model)
    wandb.config.update({"num_params": num_params, "size_mb": size_mb})
    print(f"Nombre de paramètres : {num_params:,}")
    print(f"Le modèle occupe environ {size_mb:.2f} Mo en mémoire.")

    optimizer = optim.Adam(model.parameters(), LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)

    for epoch in range(NB_EPOCH):
        model.train()
        total_train_loss = 0
        for input_seq, target_seq, loss_mask in tqdm(train_loader, desc=f"Epoch {epoch} - Train"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(input_seq)  # (batch, seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_seq.view(-1)
            mask_flat = loss_mask.view(-1)

            loss = criterion(logits_flat[mask_flat], target_flat[mask_flat])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})

        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for input_seq, target_seq, loss_mask in tqdm(val_loader, desc=f"Epoch {epoch} - Val"):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                loss_mask = loss_mask.to(device)

                logits = model(input_seq)
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target_seq.view(-1)
                mask_flat = loss_mask.view(-1)

                loss = criterion(logits_flat[mask_flat], target_flat[mask_flat])
                total_eval_loss += loss.item()
                wandb.log({"val_loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_eval_loss / len(val_loader)
        print(f"[INFO] Epoch {epoch} : train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "decoderonly_test.pth")
