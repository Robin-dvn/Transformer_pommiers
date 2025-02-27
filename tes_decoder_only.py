# %%

import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
from torch.profiler import profile as torch_profile, record_function, ProfilerActivity
from transformer import TransformerDecoderOnly
from pathlib import Path

# %%

# Paramètres du modèle
BATCH_SIZE = 10
SEQ_LEN = 80
D_MODEL = 32 # Taille d'embedding
NUM_HEADS = 4
NUM_LAYERS = 200
VOCAB_SIZE = 17
PADDING_IDX = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%

# Initialisation du modèle sur GPU
model = TransformerDecoderOnly(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    n_head=NUM_HEADS, 
    num_decoder_layers=NUM_LAYERS, 
    padding_idx=PADDING_IDX
).to(device)

# %%

# Génération des entrées (tokens sous forme d'entiers)
decoder_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(device)
target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(device)

# %%

# Calcul des FLOPs du Forward avec `thop`
flops_forward, params = profile(model, inputs=(decoder_input, None, True))

# Estimation des FLOPs du backward (≈2.5× forward)
flops_backward = 2.5 * flops_forward
flops_total_estimated = flops_forward + flops_backward
numparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"THOP - Forward FLOPs: {flops_forward:.2e}")
print(f"Estimation - Backward FLOPs: {flops_backward:.2e}")
print(f"Total Estimated FLOPs (Forward + Backward): {flops_total_estimated:.2e}")
print(f"Nombre de paramètres : {numparams:,}")
import math

def temps_epoch(nb_samples, batch_size, flops_forward, gpu_flops,
                backward_multiplier=2, flops_optimizer=6.47e10, efficiency=0.5):
    nb_batches = math.ceil(nb_samples / batch_size)
    # Coût total en FLOPS par batch
    flops_batch = flops_forward + backward_multiplier * flops_forward + flops_optimizer
    # Performance effective de la carte (on n'atteint souvent qu'une fraction de la perf théorique)
    effective_gpu_flops = gpu_flops * efficiency
    temps_batch = flops_batch / effective_gpu_flops
    return nb_batches * temps_batch

# Exemple :
# - 50 000 échantillons, batch_size de 64
# - 1e9 FLOPS pour le forward
# - GPU théorique à 10 TFLOPS (10e12 FLOPS)
temps = temps_epoch(nb_samples=200000, batch_size=21, flops_forward=1e9, gpu_flops=10e12)
print(f"Temps estimé par époque : {temps:.3f} sec")

        # %%

# Initialisation de l'optimizer et de la loss
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)

# %%

# Profiling des FLOPs réels avec `torch.profiler`
with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("forward"):
        logits = model(decoder_input, None, True)
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

    with record_function("backward"):
        loss.backward()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# %%
