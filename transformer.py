
from PommierDataset import PommierDataset,collate_fn
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOnlyTransformerLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, 
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, 
                         layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        
    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                tgt_is_causal=False, memory_is_causal=False):
        """
        Version Decoder-Only :
        - Supprime le Cross-Attention avec `memory`
        - Garde uniquement le Self-Attention causale et le Feed-Forward Network (FFN)
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


class Transformer(nn.Module):


    def __init__(self,in_vocab_size,out_vocab_size,d_model,n_head,padding_idx,nb_layer= 3) -> None:
        super(Transformer,self).__init__()
        self.embed = nn.Embedding(in_vocab_size,d_model,padding_idx=padding_idx)
        self.posEmbed = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model,n_head,batch_first=True,dropout=0.1,num_decoder_layers=nb_layer,num_encoder_layers=nb_layer)
        self.fc_l = nn.Linear(d_model,out_vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu' 
    def forward(self, src: Tensor, tgt: Tensor, tgt_key_padding_mask: Tensor = None) -> Tensor:

        s_emb = self.embed(src)
        t_emb = self.embed(tgt)
        s_p_emb = self.posEmbed(s_emb)
        t_p_emb = self.posEmbed(t_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)
        
        out_trans = self.transformer(
            s_p_emb, t_p_emb, tgt_mask=tgt_mask, tgt_is_causal=True, tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = self.fc_l(out_trans)
        return out



    def generate_batch(self,enc_input,sos_idx,device,end_toks_list,temperature = 1,max_length = 200,batch_size = 1):


        self.eval()
        sequence = torch.tensor([[sos_idx]]*batch_size,device = device)
        stop_mask = torch.tensor([False]*batch_size,device = device)

        for i in tqdm(range(max_length),colour="green"):
            with torch.no_grad():
                logits = self(enc_input,sequence)
                logits = logits[:,-1,:] /temperature

            probs = F.softmax(logits, dim=-1)
            cutoff = 0.0008
            probs = torch.where(probs < cutoff, torch.tensor(0.0, device=probs.device), probs)

            # Renormaliser les probabilités pour qu'elles forment toujours une distribution valide
            probs = probs / probs.sum()
            next_tokens = torch.multinomial(probs, 1)
            while torch.any(torch.isin(next_tokens, torch.tensor([0, 1,12,13,14,15,16], device=self.device))) :
                next_tokens = torch.multinomial(probs, 1) 
            sequence = torch.cat([sequence,next_tokens],dim = 1)

            has_end_tok =torch.isin(next_tokens,torch.tensor(end_toks_list,device=self.device)) 

            stop_mask = stop_mask | has_end_tok.flatten()
            if stop_mask.all():
                break

        return sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_decoder_layers, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.posEmbed = PositionalEncoding(d_model)
        decoder_layer = DecoderOnlyTransformerLayer(d_model, n_head, batch_first=True, dropout=0.1,dim_feedforward=4*d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, tgt: torch.Tensor, tgt_key_padding_mask: torch.Tensor = None,generating= False):
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

    def generate_batch(self, input_tokens, sos_idx, device, end_toks_list, max_length=200, temperature=1, batch_size=None):
        self.eval()
        if batch_size is None:
            batch_size = input_tokens.size(0)
        sequence = input_tokens.clone()  # Séquence de départ, par ex. [[type, year], ...]
        sequence = torch.cat([sequence, torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)], dim=1)
        stop_mask = torch.tensor([False]*batch_size,device = device)

        for i in tqdm(range(max_length),colour="green"):
            with torch.no_grad():
                logits = self(sequence,generating=True)
                logits = logits[:,-1,:] /temperature

            probs = F.softmax(logits, dim=-1)
            cutoff = 0.0008
            probs = torch.where(probs < cutoff, torch.tensor(0.0, device=probs.device), probs)

            # Renormaliser les probabilités pour qu'elles forment toujours une distribution valide
            probs = probs / probs.sum()
            next_tokens = torch.multinomial(probs, 1)
            while torch.any(torch.isin(next_tokens, torch.tensor([0, 1,12,13,14,15,16], device=self.device))) :
                next_tokens = torch.multinomial(probs, 1) 
            sequence = torch.cat([sequence,next_tokens],dim = 1)

            has_end_tok =torch.isin(next_tokens,torch.tensor(end_toks_list,device=self.device)) 

            stop_mask = stop_mask | has_end_tok.flatten()
            if stop_mask.all():
                break

        return sequence






if __name__ == "__main__":
    dataset = PommierDataset("out/datasetcustom10000.csv")
    dataloader = DataLoader(dataset,3,False,collate_fn=collate_fn)

    batch = next(iter(dataloader))



    pad_token_id = 0
    enc_inputs, dec_inputs, dec_targets = batch
    padding_mask = (dec_inputs == pad_token_id)

    model = Transformer(16,11,4,2,0)
    out = model(enc_inputs,dec_inputs,padding_mask)

    pad_token_id = 0

    # Transformer logits et targets
    print(out.shape)
    logits = out.view(-1, out.size(-1))  # (batch_size * seq_len, vocab_size)
    print(logits.shape)
    print(dec_targets.shape)
    targets = dec_targets.view(-1)  # (batch_size * seq_len,)
    print(targets.shape)

    # Calculer la perte en ignorant le padding
    loss = F.cross_entropy(logits, targets, ignore_index=pad_token_id)
    print("Loss:\n",loss)
    print("Encoder Inputs:\n", enc_inputs)
    print("Decoder Inputs:\n", dec_inputs)
    print("Decoder Targets:\n", dec_targets)
    print("outdu transformer:\n",out)
    print("targets:\n",targets)
    print("Logits:\n",logits)