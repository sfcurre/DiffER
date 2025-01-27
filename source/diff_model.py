import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_util import SinusoidalPosEmb
from rdkit import Chem, RDLogger

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and multinomial diffusion (https://github.com/ehoogeboom/multinomial_diffusion/tree/main)
'''

class DiffusionModel(nn.Module):
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
        super(DiffusionModel, self).__init__()

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
        self.embed_lengths = nn.Embedding(self.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(dropout)

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

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
        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        
        memory, memory_pad_mask, predicted_lengths = self.encode(encoder_input, encoder_pad_mask)
        
        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        t = batch["decoder_t"]
        
        token_output = self.decode(decoder_input, decoder_pad_mask, memory, memory_pad_mask, t)
        return token_output, predicted_lengths

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

    def encode(self, encoder_input, encoder_pad_mask):
        encoder_embs = self.embed_log_onehot(encoder_input)
        
        len_tokens = self.embed_lengths(torch.zeros(1, encoder_embs.size(1), dtype=torch.int32).cuda())
        # len_tokens = self.embed_lengths(encoder_pad_mask.sum(-1).unsqueeze(-1)) # input to embedding is source length
        encoder_embs = torch.cat([len_tokens, encoder_embs], dim=0)
        encoder_pad_mask = torch.cat([encoder_pad_mask[:, :1], encoder_pad_mask], dim=-1)

        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)

        predicted_lengths_logits = torch.matmul(model_output[0, :, :], self.embed_lengths.weight.transpose(0, 1)).float()
        predicted_lengths_logits[:, 0] += float('-inf')   # Cannot predict the len_token
        predicted_lengths = F.log_softmax(predicted_lengths_logits, dim=-1)

        return model_output, encoder_pad_mask, predicted_lengths

    def decode(self, decoder_input, decoder_pad_mask, memory, memory_pad_mask, t):
        decoder_embs = self.embed_log_onehot(decoder_input, t)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool).cuda()

        model_output = self.decoder(decoder_embs, memory,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.token_fc(model_output)
        return token_output

    