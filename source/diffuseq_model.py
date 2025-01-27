import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_util import *
from rdkit import Chem, RDLogger

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and multinomial diffusion (https://github.com/ehoogeboom/multinomial_diffusion/tree/main)
adjusted to match DiffuSeq's approach (https://arxiv.org/abs/2210.08933)
'''

class DiffuseqModel(nn.Module):
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
        super(DiffuseqModel, self).__init__()

        self.tokeniser = tokeniser
        self.max_seq_len = max_seq_len * 2 + 1
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.activation = activation
        self.dropout = dropout

        self.register_buffer('log_alpha', self.collate_fn.log_alpha)
        self.register_buffer('log_1_min_alpha', self.collate_fn.log_1_min_alpha)
        self.register_buffer('log_cumprod_alpha', self.collate_fn.log_cumprod_alpha)
        self.register_buffer('log_1_min_cumprod_alpha', self.collate_fn.log_1_min_cumprod_alpha)

        self.vocab_size = vocab_size = len(tokeniser)
        self.pad_token_idx = pad_token_idx = self.tokeniser.vocab[self.tokeniser.pad_token]
                
        self.sep_token_idx = self.tokeniser.vocab[self.tokeniser.sep_token]
        self._sep_token = index_to_log_onehot(torch.tensor([[self.sep_token_idx]]), len(self.tokeniser)).permute(2, 0, 1)

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.dropout = nn.Dropout(dropout)

        layer_norm = nn.LayerNorm(d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation, norm_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers, norm=layer_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()
        self.register_buffer("pos_emb", self.positional_embs())
        self.register_buffer("sep_token", self._sep_token)

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
        in_input = batch["encoder_input"]
        in_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        
        out_input = batch["decoder_input"]
        out_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        
        batch_size, in_seq_len = tuple(in_pad_mask.size())

        t = batch["decoder_t"]

        # padding is 1 where pad is, 0 else
        combined_input = torch.concat([in_input, self.sep_token.repeat(1, batch_size, 1), out_input], axis=0)
        combined_mask = torch.concat([in_pad_mask, torch.zeros((batch_size, 1), device=batch['device']), out_pad_mask], axis=-1)
        
        embs = self.embed_log_onehot(combined_input, t)
        model_output = self.transformer(embs, src_key_padding_mask=combined_mask)
        token_output = self.token_fc(model_output)

        output_mask = torch.ones(combined_input.size()[0], device=batch['device'], dtype=bool)
        output_mask[:in_seq_len + 1] = 0
        return token_output[output_mask]

    def embed_log_onehot(self, log_onehot_input, t=None):
        seq_len, _, _ = tuple(log_onehot_input.size())

        onehot_input = torch.exp(log_onehot_input)
        onehot_embs = torch.matmul(onehot_input, self.emb.weight)
        onehot_embs = onehot_embs * np.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        onehot_embs = onehot_embs + positional_embs
        if t is not None:
            time_embs = self.time_emb(t)
            onehot_embs += time_embs
        onehot_embs = self.dropout(onehot_embs)
        return onehot_embs
