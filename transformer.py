
from PommierDataset import PommierDataset,collate_fn
from torch.utils.data import DataLoader
from torch import Tensor

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

class Transformer(nn.Module):


    def __init__(self,in_vocab_size,out_vocab_size,d_model,n_head,padding_idx) -> None:
        super(Transformer,self).__init__()
        self.embed = nn.Embedding(in_vocab_size,d_model,padding_idx=padding_idx)
        self.posEmbed = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model,n_head,batch_first=True,dropout=0.1,num_decoder_layers=3,num_encoder_layers=3)
        self.fc_l = nn.Linear(d_model,out_vocab_size)
    
    def forward(self, src: Tensor, tgt: Tensor, tgt_key_padding_mask: Tensor = None) -> Tensor:

        s_emb = self.embed(src)
        t_emb = self.embed(tgt)
        s_p_emb = self.posEmbed(s_emb)
        t_p_emb = self.posEmbed(t_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to("cuda")
        
        out_trans = self.transformer(
            s_p_emb, t_p_emb, tgt_mask=tgt_mask, tgt_is_causal=True, tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = self.fc_l(out_trans)
        return out



    def generate_batch(self,enc_input,sos_idx,device,end_toks_list,temperature = 1,max_length = 200):
        ################# A Recoder C'est pas bon ################
        self.eval()
        sequence = torch.tensor([[sos_idx]],device = device)
        for i in range(max_length):
            with torch.no_grad():
                logits = self(enc_input,sequence)
                # print(logits)
                logits = logits[:,-1,:] /temperature
            probs = F.softmax(logits, dim=-1)
            # print(probs)
            next_token = torch.multinomial(probs, 1)
            if next_token.item() in end_toks_list:
                sequence = torch.cat([sequence,next_token],dim = 1)
                break
            sequence = torch.cat([sequence,next_token],dim = 1)
            # print(sequence)
        return sequence[:,0:].to('cpu').tolist()[0]






if __name__ == "__main__":
    dataset = PommierDataset("out/dataset.csv")
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