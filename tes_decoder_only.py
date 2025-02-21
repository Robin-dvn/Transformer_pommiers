import torch
from transformer import TransformerDecoderOnly

import torch.nn.functional as F
import torch
import random
import numpy as np

# Fixer tous les seeds pour obtenir des résultats reproductibles
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Assurer un mode déterministe pour les calculs en GPU (si utilisé)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Charger le modèle

#import torch

# Séquence cible avec tokens ignorés (`10`)
target_sequence = torch.tensor([[13, 1,2,2,2,2,7]], dtype=torch.long)

# Calculer la cross entropy avec `ignore_index=10`
loss_fn_mean = torch.nn.CrossEntropyLoss(ignore_index=10, reduction='mean')
loss_fn_sum = torch.nn.CrossEntropyLoss(ignore_index=10, reduction='sum')

# Nombre de tokens valides (exclure `ignore_index=10`)
valid_tokens = (target_sequence != 10).sum().item()

# Charger le modèle
model_path = "decoderonly_128_layers_3_12.pth"
model = TransformerDecoderOnly(vocab_size=17, d_model=128, n_head=4, num_decoder_layers=3, padding_idx=0)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to('cpu')
model.eval()
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        print("Dropout still active:", module.training)

# Séquence d'entrée avec padding
input_sequence = torch.tensor([[8, 13,1,2,2,2,2 ]], dtype=torch.long)

# Générer les probabilités de sortie
with torch.no_grad():
    # Créer un masque de padding pour l'entrée
    padding_mask = (input_sequence == 0).to(torch.bool)

    # Passer le masque au modèle
    logits = model(input_sequence, tgt_key_padding_mask=padding_mask)
    
    # Ignorer les deux premiers tokens
    logits = logits[:, 2:, :]
    probs = F.softmax(logits, dim=-1)
    target_sequence = target_sequence[:, 2:]

    # Calcul de la perte avec `reduction='mean'`
    loss_mean = loss_fn_mean(logits.view(-1, logits.size(-1)), target_sequence.view(-1))

    # Calcul de la perte avec `reduction='sum'`, puis division par le nombre de tokens valides
    loss_sum = loss_fn_sum(logits.view(-1, logits.size(-1)), target_sequence.view(-1))
    loss_correct = loss_sum / valid_tokens if valid_tokens > 0 else 0.0  # Évite la division par zéro

print(f"Loss (mean) = {loss_mean.item():.4f}")
print(probs[:,:,:])