from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import itertools
from utils.HSMM import HSMM
from vmapplet_utils.enums import Observation
import numpy as np
from vmapplet_utils.sequences import terminal_fate, _generate_random_draw_sequence
from torch.utils.data import random_split



class PommierDatasetDecoderOnly(Dataset):
    def __init__(self, dataset_path, token_to_id=None):
        """
        Dataset PyTorch pour un modèle Décodeur-only.

        Args:
            dataset_path (str): Chemin du fichier CSV contenant les séquences brutes.
            token_to_id (dict, optional): Mapping token -> ID. S'il est None, il sera construit.
        """
        self.dataset = pd.read_csv(dataset_path)
        self.vocab = {
            "LARGE": "L", "MEDIUM": "M", "SMALL": "S", "FLORAL": "F", "DORMANT": "D",
            "Y1": "Y1", "Y2": "Y2", "Y3": "Y3", "Y4": "Y4", "Y5": "Y5"
        }

        # Tokenisation à la volée
        self.dataset["tokens"] = self.dataset.apply(lambda row: self.tokenize_row(row), axis=1)

        # Construction du vocabulaire (si non fourni)
        if token_to_id is None:
            self.token_to_id = self.build_vocab(self.dataset["tokens"])
        else:
            self.token_to_id = token_to_id

        # Conversion des tokens en IDs
        self.dataset["token_ids"] = self.dataset["tokens"].apply(
            lambda tokens: [self.token_to_id[token] for token in tokens]
        )

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
        """Construit le mapping token -> ID en incluant les tokens spéciaux."""
        unique_tokens = sorted(set(itertools.chain.from_iterable(token_lists)))
        # Ajout des tokens spéciaux
        vocab = {"<PAD>": 0, "<SOS>": 1}
        vocab.update({token: idx + len(vocab) for idx, token in enumerate(unique_tokens)})
        return vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Pour chaque exemple, on construit :
          - full_seq = [token1, token2, <SOS>, token3, token4, ...]
          - input_seq  = full_seq[:-1]
          - target_seq = full_seq[1:]
        La perte sera calculée uniquement à partir du token situé après <SOS>.
        Ici, on ignore les positions 0, 1 et 2.
        """
        token_ids = self.dataset.iloc[idx]["token_ids"]
        full_seq = token_ids[:2] + [self.token_to_id["<SOS>"]] + token_ids[2:]
        input_seq = torch.tensor(full_seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(full_seq[1:], dtype=torch.long)
        # print(target_seq)

        loss_mask = torch.zeros(len(input_seq), dtype=torch.bool)
        # On calcule la perte seulement à partir du token après <SOS> (index 3 et plus)
        if len(loss_mask) > 3:
            loss_mask[2:] = True
        # print(loss_mask)
        return input_seq, target_seq, loss_mask



def collate_fn_decoder_only(batch):
    inputs, targets, masks = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)   # <PAD> = 0
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=False)
    return inputs, targets, masks


class DecoderOnlyDynamicPommierDataset(Dataset):
    """
    Dataset dynamique pour un modèle Décodeur-only, qui génère des données à la volée.

    Args:
        token_to_id (dict): Dictionnaire de mapping token -> ID.
        num_samples (int): Nombre total d'échantillons à générer.
        min_length (int): Longueur minimale des séquences générées.
        max_length (int): Longueur maximale des séquences générées.
    """

    def __init__(self, token_to_id, num_samples, min_length, max_length):
        self.token_to_id = token_to_id
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.mappings = {
            1: {Observation.LARGE: "data/markov/fuji_long_year_1.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_3.toml"},
            2: {Observation.LARGE: "data/markov/fuji_long_year_2.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_3.toml"},
            3: {Observation.LARGE: "data/markov/fuji_long_year_3.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_3.toml"},
            4: {Observation.LARGE: "data/markov/fuji_long_year_4.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_4.toml"},
            5: {Observation.LARGE: "data/markov/fuji_long_year_5.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_5.toml"}
        }
        self.starting_states = [
            Observation.SMALL,
            Observation.FLORAL,
            Observation.LARGE,
            Observation.MEDIUM,
        ]

        # Initialiser tous les modèles HSMM nécessaires
        self.hsmm_models = {}
        for year, state_dict in self.mappings.items():
            for state, toml_file in state_dict.items():
                self.hsmm_models[(year, state)] = HSMM(toml_file)

    def __len__(self):
        """Retourne le nombre total d'échantillons."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Génère un exemple de données à la volée.

        Retourne :
            - full_seq = [token1, token2, <SOS>, token3, token4, ...]
            - input_seq  = full_seq[:-1]
            - target_seq = full_seq[1:]
        """
        # Sélectionner un état de départ et une année aléatoirement
        starting_state = self.starting_states[np.random.randint(0, len(self.starting_states))]
        year = np.random.randint(1, 6)

        # Générer une séquence
        seq = [starting_state.value, f"Y{year}"]
        hsmm_model = self.hsmm_models.get((year, starting_state))
        terminal = terminal_fate(year, starting_state) if starting_state != Observation.FLORAL else Observation.DORMANT
        seq = seq + self.generate_seq(starting_state, year, hsmm_model)
        seq.append(terminal.value)

        # Convertir les observations en tokens
        tokens = [str(obs) for obs in seq]

        # Convertir les tokens en IDs
        token_ids = [self.token_to_id[token] for token in tokens if token in self.token_to_id]

        # Construire la séquence complète
        full_seq = token_ids[:2] + [self.token_to_id["<SOS>"]] + token_ids[2:]
        input_seq = torch.tensor(full_seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(full_seq[1:], dtype=torch.long)

        # Créer un masque de perte
        loss_mask = torch.zeros(len(input_seq), dtype=torch.bool)
        if len(loss_mask) > 3:
            loss_mask[2:] = True

        return input_seq, target_seq, loss_mask

    def generate_seq(self, starting_state, year, hsmm=None):
        """Génère une séquence en fonction de l'état de départ et de l'année."""
        if starting_state in [Observation.FLORAL, Observation.SMALL]:
            return [0, 0, 0, 0]
        elif year == 2 and starting_state == Observation.LARGE:
            seq = _generate_random_draw_sequence()
            seq = [str(el[1]) for el in seq]
            # print(seq)
            return seq
        seq = hsmm.generate_bounded_sequence(self.min_length, self.max_length)[1]
        # print(seq )
        return seq


if __name__ == "__main__":
    VAL_SPLIT = 0.8
    vocab_to_id ={'<PAD>': 0, '<SOS>': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, 'DORMANT': 7, 'FLORAL': 8, 'LARGE': 9, 'MEDIUM': 10, 'SMALL': 11, 'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16}
    static_dataset = PommierDatasetDecoderOnly("out/markov_python_generated_dataset10000.csv")
    dataset = DecoderOnlyDynamicPommierDataset(vocab_to_id, 10000, 4, 70)
    train_size = int(VAL_SPLIT * len(static_dataset))
    val_size = len(static_dataset) - train_size
    _, val_split = random_split(static_dataset, [train_size, val_size])
    val_loader = DataLoader(val_split, batch_size=2, shuffle=True, collate_fn=collate_fn_decoder_only)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_decoder_only)

    for batch in train_loader:
        inputs, targets, masks = batch
        # Ici, vous pouvez passer 'batch' à votre modèle pour l'entraînement
        print(inputs, targets)
        break
