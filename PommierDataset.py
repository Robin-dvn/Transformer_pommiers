from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import itertools

class PommierDataset(Dataset):
    def __init__(self, dataset_path, token_to_id=None):
        """
        Dataset PyTorch pour un modèle Seq2Seq, qui tokenise et encode à la volée.
        
        Args:
            dataset_path (str): Chemin du fichier CSV contenant les séquences brutes.
            token_to_id (dict, optional): Dictionnaire de mapping token -> ID. Si None, il sera construit.
        """
        self.dataset = pd.read_csv(dataset_path)
        self.vocab = {
            "LARGE": "L", "MEDIUM": "M", "SMALL": "S", "FLORAL": "F","DORMANT":"D",
            "Y1": "Y1", "Y2": "Y2", "Y3": "Y3", "Y4": "Y4", "Y5": "Y5"
        }
        
        # Tokenisation des données brutes (à la volée)
        self.dataset["tokens"] = self.dataset.apply(lambda row: self.tokenize_row(row), axis=1)
        
        # Création du vocabulaire (si non fourni)
        if token_to_id is None:
            self.token_to_id = self.build_vocab(self.dataset["tokens"])
        else:
            self.token_to_id = token_to_id
        
        # Conversion des tokens en IDs
        self.dataset["token_ids"] = self.dataset["tokens"].apply(lambda tokens: [self.token_to_id[token] for token in tokens])

    def tokenize_row(self, row):
        """Tokenise une ligne du dataset."""
        tokens = []
        for item in row:
            item = str(item).strip()
            if item in self.vocab:
                tokens.append(self.vocab[item])
            elif item.isdigit():
                tokens.extend(list(item))  # Chaque chiffre devient un token
        return tokens
    
    def build_vocab(self, token_lists):
        """Construit le dictionnaire token -> ID en incluant les tokens spéciaux."""
        unique_tokens = sorted(set(itertools.chain.from_iterable(token_lists)))
        
        # Ajouter les tokens spéciaux en priorité
        vocab = {
            "<PAD>": 0,
            "<SOS>": 1,
        }
        
        # Ajouter les tokens du dataset en continuant l'indexation
        vocab.update({token: idx + len(vocab) for idx, token in enumerate(unique_tokens)})
        
        return vocab
    
    def __len__(self):
        """Retourne le nombre d'exemples dans le dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Récupère un exemple du dataset sous forme de tenseur PyTorch.
        
        Retourne :
            - Les 2 premiers tokens comme entrée (encodeur)
            - Le reste comme cible (décodeur)
        """
        token_ids = self.dataset.iloc[idx]["token_ids"]
        
        # Séparation encodeur / décodeur
        encoder_input = torch.tensor(token_ids[:2]+[], dtype=torch.long)  # 2 premiers tokens
     
        decoder_input = torch.tensor([self.token_to_id["<SOS>"]] + token_ids[2:-1], dtype=torch.long)  # Ajout <SOS>
       
        decoder_target = torch.tensor(token_ids[2:], dtype=torch.long)  
        
        return encoder_input, decoder_input, decoder_target

def collate_fn(batch):
    """Applique du padding pour aligner les séquences dans un batch."""
    enc_inputs, dec_inputs, dec_targets = zip(*batch)

    # Ajouter du padding pour que toutes les séquences aient la même longueur
    dec_inputs = pad_sequence(dec_inputs, batch_first=True, padding_value=0)# <PAD> token is 0
    dec_targets = pad_sequence(dec_targets, batch_first=True, padding_value=0)
    enc_inputs = pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    return enc_inputs, dec_inputs, dec_targets


