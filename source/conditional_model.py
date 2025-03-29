import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and multinomial diffusion (https://github.com/ehoogeboom/multinomial_diffusion/tree/main)
'''

class ConditionalModel(nn.Module):
    def __init__(self,
        tokeniser,
        max_seq_len,
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        activation,
        dropout=0.1,
        ):
        super(ConditionalModel, self).__init__()

        self.tokeniser = tokeniser
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.activation = activation
        self.dropout = dropout

        self.vocab_size = vocab_size = len(tokeniser)
        self.pad_token_idx = pad_token_idx = self.tokeniser.vocab[self.tokeniser.pad_token]

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.dropout = nn.Dropout(dropout)

        self.length_rep = nn.Embedding(1, self.d_model)
        self.length_map = nn.Linear(self.d_model, self.max_seq_len)
        
        enc_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation, norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation, norm_first=True, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)

        self._init_params()
        self.register_buffer("pos_emb", self.positional_embs())

        RDLogger.DisableLog("rdApp.*")

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def forward(self, batch):
        memory, memory_pad_mask, predicted_lengths = self.encode(batch['y_0'], batch['y_mask'])
        logits = self.decode(batch['x_t'], batch['x_mask'], memory, memory_pad_mask, batch['t'])
        return logits, predicted_lengths

    def embed_onehot(self, onehot_input, t=None):
        _, seq_len, _ = tuple(onehot_input.size())

        onehot_embs = torch.matmul(onehot_input, self.emb.weight)
        onehot_embs = onehot_embs * np.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0)
        onehot_embs = onehot_embs + positional_embs
        if t is not None:
            time_embs = self.time_emb(t)
            onehot_embs += time_embs.unsqueeze(1)
        onehot_embs = self.dropout(onehot_embs)
        return onehot_embs

    def encode(self, encoder_input, encoder_pad_mask):
        encoder_embs = self.embed_onehot(encoder_input)
        batch, _, _ = tuple(encoder_embs.size())
        
        len_tokens = self.length_rep(torch.zeros(batch, 1, dtype=torch.int32, device=encoder_embs.device))
        encoder_embs = torch.cat([len_tokens, encoder_embs], dim=1)
        encoder_pad_mask = torch.cat([encoder_pad_mask[:, :1], encoder_pad_mask], dim=-1)

        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)

        predicted_lengths_logits = self.length_map(model_output[:, 0, :])

        return model_output, encoder_pad_mask, predicted_lengths_logits

    def decode(self, decoder_input, decoder_pad_mask, memory, memory_pad_mask, t):
        decoder_embs = self.embed_onehot(decoder_input, t)

        _, seq_len, _ = tuple(decoder_embs.size())
        tgt_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=decoder_embs.device)

        model_output = self.decoder(decoder_embs, memory,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        logits = self.token_fc(model_output)
        return logits

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
