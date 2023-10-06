import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_util import log_sample_categorical

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
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = self.positional_embs()

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
        memory = self.encode(encoder_input, encoder_pad_mask)
        
        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        token_output = self.decode(decoder_input, decoder_pad_mask, memory, encoder_pad_mask)
        return token_output

    def embed_onehot(self, onehot_input):
        seq_len, _, _ = tuple(onehot_input.size())

        onehot_embs = torch.matmul(onehot_input, self.emb.weight)
        onehot_embs = onehot_embs * np.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        onehot_embs = onehot_embs + positional_embs
        onehot_embs = self.dropout(onehot_embs)
        return onehot_embs

    def encode(self, encoder_input, encoder_pad_mask):
        encoder_embs = self.embed_onehot(encoder_input)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, decoder_input, decoder_pad_mask, memory, memory_pad_mask):
        decoder_embs = self.embed_onehot(decoder_input)
        model_output = self.decoder(decoder_embs, memory,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )
        token_output = self.token_fc(model_output)
        return token_output

    def get_target_lengths(self, pad_mask):
        lengths = len(pad_mask) - pad_mask.sum(0).unsqueeze(-1)
        return lengths

    def get_length_mask(self, lengths):
        max_len = lengths.max().item()
        length_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.int32), 1)
        length_mask = torch.stack([length_mask[lengths[batch] - 1] for batch in range(batch_size)], dim=0)
        return length_mask

    def init_noise(self, target_lengths):
        length_mask = self.get_length_mask(target_lengths)
        
        uniform_logits = torch.zeros((length_mask.shape[0], len(self.tokeniser), length_mask.shape[1]))
        tgt_tokens = torch.exp(log_sample_categorical(uniform_logits, len(self.tokeniser))).permute(2, 0, 1)
        
        pad_token =  torch.exp(index_to_log_onehot(torch.tensor([[self.pad_token_id]]), len(self.tokeniser))).permute(2, 0, 1)
        tgt_tokens = (1 - length_mask.transpose(0, 1).unsqueeze(-1)) * tgt_tokens + length_mask.transpose(0, 1).unsqueeze(-1) * pad_token 
        return tgt_tokens, length_mask

    def sample(self, batch, verbose=True):
        self.freeze()

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        memory, memory_pad_mask = self.encode(encoder_input, encoder_pad_mask)

        lengths = self.get_lengths_from_padding(batch['target_pad'])
        tgt_tokens, length_mask = self.init_noise(lengths)
        
        for t in range(self.num_timesteps):
            token_output = self.decode(tgt_tokens, length_mask, memory, memory_pad_mask)
            tgt_tokens = torch.softmax(token_output, dim=-1)

            if self.verbose:
                ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).detach().numpy()
                tokens = self.tokeniser.convert_ids_to_tokens(ids)
                sampled_mols = self.tokeniser.detokenise(tokens)

                print(f'{t + 1}: {sampled_mols[0]}')

        if self.verbose:
            print('-' * 20)

        ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).detach().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)

        sampled_mols = [m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols]
        sampled_mols = [m.replace('?', '') for m in sampled_mols]

        self.unfreeze()
        return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0])
