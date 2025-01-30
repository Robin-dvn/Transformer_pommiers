from DatasetCreator import DatasetCreator
from pathlib import Path
from transformer import Transformer
from PommierDataset import PommierDataset,collate_fn
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb

def model_size_mb(model):
    total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return total_size / (1024 ** 2)  # Convertit en Mo

if __name__ == "__main__":

    exp_name = "Simple Transformer"
    dataset_name = "100 sample de chaque type"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ########## CONSTANTS ##################

    BATCH_SIZE = 32
    VAL_SPLIT = 0.8
    IN_VOCAB_SIZE = 16
    OUT_VOCAB_SIZE = 11
    PADDING_IDX = 0
    N_HEAD = 4
    D_MODEL = 128
    LR = 0.001
    NB_EPOCH = 1

    ########## WANDB Project ##############
    wandb.init(
        # set the wandb project where this run will be logged
        project="Topologie-Pommiers",

        # track hyperparameters and run metadata
        config={
        "learning_rate": LR,
        "Val split": VAL_SPLIT,
        "architecture": exp_name,
        "dataset": dataset_name,
        "batch size": BATCH_SIZE,
        "Dimension model": D_MODEL,
        "Numer of heads": N_HEAD,
        "epochs": NB_EPOCH,
        }
    )
    ########## DATASET GENERATION #########
    
    outpath = "out"
    data_creator = DatasetCreator(outpath,1234,4,70,1)
    if Path(outpath+"/dataset.csv").exists():
        data_creator.load_data(outpath+"/dataset.csv")
    else:
        data_creator.create_data(True,True)

   ######### PYTORCH DATASET CREATION ##### 

    dataset = PommierDataset(outpath+ "/dataset.csv")
    train_size = int(VAL_SPLIT*len(dataset))
    val_size = len(dataset) - train_size
    train_split,val_split = random_split(dataset,[train_size, val_size])

    train_loader = DataLoader(train_split,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_split,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)

    ######### MODEL-OPTIMIZER-LOSS#########

    model = Transformer(IN_VOCAB_SIZE,OUT_VOCAB_SIZE,D_MODEL,N_HEAD,PADDING_IDX)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = model_size_mb(model)
    wandb.config.update({"num_params": num_params,"size_mb":size_mb})
    print(f"Nombre de paramètres ajouté à wandb : {num_params:,}")
    print(f"Le modèle occupe environ {size_mb:.2f} Mo en mémoire.")

    optimizer = optim.Adam(model.parameters(),LR)

    criterion = nn.CrossEntropyLoss(ignore_index = PADDING_IDX) 

    ############### TRAINING ##############

    for epoch in range(NB_EPOCH):

        ############### train split ##############
        model.train()
        total_train_loss = 0
        for enc_inp,dec_inp,dec_target in tqdm(train_loader):

            enc_inp =enc_inp.to(device)
            dec_inp = dec_inp.to(device)
            dec_target = dec_target.to(device)
            padding_mask = (dec_inp == PADDING_IDX).to(torch.float32)

            logits = model(enc_inp,dec_inp,padding_mask)

            logits = logits.view(-1,logits.size(-1))
            dec_target = dec_target.view(-1)
            loss = criterion(logits,dec_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            wandb.log({"train_loss":loss.item()})

        ############### val split ##############
        model.eval()
        total_eval_loss = 0
        with torch.no_grad(): 

            for enc_inp,dec_inp,dec_target in val_loader:

                enc_inp = enc_inp.to(device)
                dec_inp = dec_inp.to(device)
                dec_target = dec_target.to(device)
                padding_mask = (dec_inp == PADDING_IDX).to(torch.float32).to(device)

                logits = model(enc_inp,dec_inp,padding_mask)

                logits = logits.view(-1,logits.size(-1))
                dec_target = dec_target.view(-1)
                loss = criterion(logits,dec_target)
                total_eval_loss += loss.item()
                wandb.log({"val loss":loss.item()})
        if epoch % 10 == 0:
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_eval_loss / len(val_loader)
            print(f"[INFO] Epoch {epoch} : train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")








    
        