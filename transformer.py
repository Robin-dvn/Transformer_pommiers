

"""
Transformer Model Implementation for Decoder-Only Architecture.

This module provides the implementation of a Transformer model tailored for decoder-only tasks.
It includes positional encoding, a custom decoder layer, and the main Transformer model class.
The module also supports sequence generation with temperature scaling and token masking.

Classes:
    - PositionalEncoding: Adds positional information to token embeddings.
    - DecoderOnlyTransformerLayer: Custom Transformer decoder layer without cross-attention.
    - TransformerDecoderOnly: Main decoder-only Transformer model.

Functions:
    - generate_batch: Generates sequences using the model.

Exceptions:
    - ValidationError: Raised during sequence generation for invalid tokens.
    - GPUOutOfMemoryError: Raised when GPU memory is insufficient.

"""

import math

from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValidationError(Exception):
    """Exception levée lors d'une erreur de validation."""
    pass

class GPUOutOfMemoryError(Exception):
    """Exception levée lors d'une erreur de mémoire GPU."""
    pass

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # [B, T, D]


        return self.dropout(x)

class BadPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Applies positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            Tensor: Tensor with positional encoding applied.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DecoderOnlyTransformerLayer(nn.TransformerDecoderLayer):
    """
    Custom Transformer decoder layer for decoder-only architecture.

    This layer removes cross-attention and retains only causal self-attention
    and feed-forward network (FFN).

    Args:
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feed-forward network.
        dropout (float): Dropout rate.
        activation (callable): Activation function for FFN.
        layer_norm_eps (float): Epsilon value for layer normalization.
        batch_first (bool): Whether batch dimension comes first.
        norm_first (bool): Whether to apply normalization before other operations.
        bias (bool): Whether to include bias in linear layers.
        device (torch.device, optional): Device for the layer.
        dtype (torch.dtype, optional): Data type for the layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, bias, device, dtype)

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        """
        Forward pass for the decoder-only layer.

        Args:
            tgt (Tensor): Target sequence tensor.
            memory (Tensor, optional): Memory tensor (not used in decoder-only).
            tgt_mask (Tensor, optional): Mask for target sequence.
            memory_mask (Tensor, optional): Mask for memory (not used in decoder-only).
            tgt_key_padding_mask (Tensor, optional): Padding mask for target sequence.
            memory_key_padding_mask (Tensor, optional): Padding mask for memory (not used in decoder-only).
            tgt_is_causal (bool, optional): Whether target attention is causal.
            memory_is_causal (bool, optional): Whether memory attention is causal (not used in decoder-only).

        Returns:
            Tensor: Output tensor after applying self-attention and FFN.


        """

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # 🚀 Cross-Attention supprimé 🚀
            x = x + self._ff_block(self.norm2(x))  # FFN
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # 🚀 Cross-Attention supprimé 🚀
            x = self.norm2(x + self._ff_block(x))  # FFN


        return x



class TransformerDecoderOnly(nn.Module):
    """
    Decoder-only Transformer model.

    This model includes token embedding, positional encoding, a stack of decoder layers,
    and a final linear layer for output logits.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        num_decoder_layers (int): Number of decoder layers.
        padding_idx (int): Index for padding token.
        dim_feedforward (int): Dimension of the feed-forward network.
    """
    def __init__(self, vocab_size, d_model, n_head, num_decoder_layers, padding_idx,dim_feedforward=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.posEmbed = PositionalEncoding(d_model)
        decoder_layer = DecoderOnlyTransformerLayer(d_model, n_head, batch_first=True, dropout=0.1,dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, tgt: torch.Tensor, tgt_key_padding_mask: torch.Tensor = None, generating=False):
        """
        Forward pass for the decoder-only Transformer model.

        Args:
            tgt (Tensor): Target sequence tensor.
            tgt_key_padding_mask (Tensor, optional): Padding mask for target sequence.
            generating (bool, optional): Whether the model is in generation mode.

        Returns:
            Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """
        t_emb = self.embed(tgt)
        t_p_emb = self.posEmbed(t_emb)
        # print(t_p_emb)

        # print("Embeddings avant d'entrer dans le décodeur :", t_p_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)
        # print(tgt_mask)
        # memory = torch.zeros(tgt.shape[0], tgt.shape[1], t_p_emb.shape[-1], device=self.device)

        out_trans = self.decoder(
            t_p_emb,memory=None, tgt_mask=tgt_mask, tgt_is_causal=True, tgt_key_padding_mask=tgt_key_padding_mask
        )
        # print(out_trans[:,:3,:])


        if not generating : out_trans = out_trans * (~tgt_key_padding_mask.unsqueeze(-1))  # Masque les positions padding

        out = self.fc_out(out_trans)
        return out

    def generate_batch(self, input_tokens, sos_idx, device, end_toks_list, max_length=100, temperature=1, batch_size=None):
        """
        Generates sequences using the decoder-only Transformer model.

        Args:
            input_tokens (Tensor): Initial input tokens of shape (batch_size, seq_len).
            sos_idx (int): Index of the start-of-sequence token.
            device (torch.device): Device to run the generation on.
            end_toks_list (list): List of end token indices.
            max_length (int, optional): Maximum length of generated sequences. Defaults to 100.
            temperature (float, optional): Temperature for sampling. Defaults to 1.
            batch_size (int, optional): Batch size for generation. Defaults to None.

        Returns:
            Tensor: Generated sequences of shape (batch_size, generated_seq_len).
        """
        self.eval()
        if batch_size is None:
            batch_size = input_tokens.size(0)
        # Séquence de départ : tokens initiaux + token SOS
        sequence = input_tokens.clone()
        sequence = torch.cat([sequence, torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)], dim=1)
        stop_mask = torch.tensor([False] * batch_size, device=device)

        for i in tqdm(range(max_length), colour="green"):
            with torch.no_grad():
                logits = self(sequence, generating=True)
                logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            cutoff = 0.0008
            probs = torch.where(probs < cutoff, torch.tensor(0.0, device=probs.device), probs)
            probs = probs / probs.sum()  # Renormalisation

            next_tokens = torch.multinomial(probs, 1)

            if i == 0:
                nb_max = 0
                while torch.any(torch.isin(next_tokens, torch.tensor([0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], device=self.device))):
                    next_tokens = torch.multinomial(probs, 1)
                    nb_max += 1
                    if nb_max > 10:
                        raise ValidationError("Trop de tokens non valides générés en début de séquence.")
                        # next_tokens = torch.tensor([[2]] * batch_size, device=device)
                        # break
            nb_max = 0
            while torch.any(torch.isin(next_tokens, torch.tensor([0, 1, 12, 13, 14, 15, 16], device=self.device))):


                next_tokens = torch.multinomial(probs, 1)
                nb_max += 1
                if nb_max > 10:
                    raise ValidationError("Trop de tokens non valides générés durant la génération.")


            sequence = torch.cat([sequence, next_tokens], dim=1)
            has_end_tok = torch.isin(next_tokens, torch.tensor(end_toks_list, device=self.device))
            stop_mask = stop_mask | has_end_tok.flatten()
            if stop_mask.all():
                break

        # Si max_length est atteint, ajoute une colonne contenant le token 7
        initial_length = input_tokens.size(1) + 1  # tokens initiaux + token SOS
        if sequence.shape[1] == initial_length + max_length:
            col7 = torch.full((batch_size, 1), 7, device=device, dtype=torch.long)
            sequence = torch.cat([sequence, col7], dim=1)
            # raise ValidationError("Max_length atteint, ajout d'un token 7.")

        return sequence
