import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_util import extract, log_sample_categorical, index_to_log_onehot, log_add_exp, SinusoidalPosEmb
from rdkit import Chem, RDLogger

class DiffusionModel(nn.Module):
    def __init__(self,
        tokeniser,
        collate_fn,
        max_seq_len,
        num_timesteps,
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        activation,
        dropout=0.1,
        ):
        super(DiffusionModel, self).__init__()

        self.tokeniser = tokeniser
        self.collate_fn = collate_fn
        self.max_seq_len = max_seq_len
        self.num_timesteps = num_timesteps 
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.activation = activation
        self.dropout = dropout

        self.register_buffer("log_alpha", self.collate_fn.log_alpha)
        self.register_buffer("log_1_min_alpha", self.collate_fn.log_1_min_alpha)
        self.register_buffer("log_cumprod_alpha", self.collate_fn.log_cumprod_alpha)
        self.register_buffer("log_1_min_cumprod_alpha", self.collate_fn.log_1_min_cumprod_alpha)

        self.vocab_size = vocab_size = len(tokeniser)
        self.pad_token_idx = pad_token_idx = self.tokeniser.vocab[self.tokeniser.pad_token]

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.time_emb = SinusoidalPosEmb(d_model, num_timesteps)
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

    def get_lengths_from_padding(self, pad_mask):
        lengths = len(pad_mask) - pad_mask.sum(0).unsqueeze(-1)
        return lengths.squeeze()

    def get_length_mask(self, lengths):
        max_len = lengths.max().item()
        length_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1)
        length_mask = torch.stack([length_mask[lengths[batch] - 1] for batch in range(len(lengths))], dim=0)
        return length_mask.squeeze()

    def init_noise(self, target_lengths):
        length_mask = self.get_length_mask(target_lengths)
        uniform_logits = torch.zeros((length_mask.shape[0], len(self.tokeniser), length_mask.shape[1]))
        tgt_tokens = log_sample_categorical(uniform_logits, len(self.tokeniser)).permute(2, 0, 1)
        
        pad_token =  index_to_log_onehot(torch.tensor([[self.pad_token_idx]]), len(self.tokeniser)).permute(2, 0, 1)
        tgt_tokens = (~length_mask.transpose(0, 1).unsqueeze(-1)) * tgt_tokens + length_mask.transpose(0, 1).unsqueeze(-1) * pad_token 
        return tgt_tokens, length_mask

    def sample(self, batch, verbose=True, use_gpu=True, return_chain=False, pred_lengths=True,
               return_lengths=False, diffuse_timesteps=False, clean=True):
        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        memory, memory_pad_mask, predicted_lengths = self.encode(encoder_input, encoder_pad_mask)
            
        true_lengths = self.get_lengths_from_padding(batch['target_mask'])
        if pred_lengths:
            lengths = predicted_lengths.max(dim=-1)[1] - 10
            lengths = self.get_lengths_from_padding(batch['encoder_pad_mask']) + lengths
            #length_dist = torch.exp(predicted_lengths)
            #length_indices = torch.arange(0, predicted_lengths.shape[-1], device='cuda').repeat(len(length_dist), 1)
            #lengths = torch.ceil((length_dist * length_indices).sum(1)).int()
        else:
            lengths = true_lengths
        
        tgt_tokens, length_mask = self.init_noise(lengths.cpu())

        if use_gpu:
            tgt_tokens = tgt_tokens.cuda()
            length_mask = length_mask.cuda()
    
        if verbose:
            print(f'target: {batch["target_smiles"][0]}')

        if return_chain:
            chain = [batch["target_smiles"]]

        for t in reversed(range(1, self.num_timesteps)):
            # My code likes (time, batch, tokens)
            # MultiDiffusion code likes (batch, tokens, time)
            t_tensor = torch.full((length_mask.shape[0],), t)
            if use_gpu:
                t_tensor = t_tensor.cuda()
            
            token_output = self.decode(tgt_tokens, length_mask, memory, memory_pad_mask, t_tensor)
            if diffuse_timesteps:
                #time, batch, tokens
                add_index = self.tokeniser.vocab['<ADD>']
                max_tokens = token_output.max(dim=-1)[1]
                add_indicator = max_tokens == add_index
                lengths = self.get_lengths_from_padding(length_mask)
                added_lengths = lengths + add_indicator
                batch_idx = torch.range(0, len(length_mask))
                token_output[added_lengths, batch_idx] = token_output[lengths, batch_idx]
                length_mask[batch_idx, added_lengths] = False

            log_token_output = torch.log_softmax(token_output, dim=-1).permute((1, 2, 0))
            log_model_pred = self.q_posterior(log_x_start=log_token_output, log_x_t=tgt_tokens.permute((1, 2, 0)), t=t_tensor)
            tgt_tokens = log_sample_categorical(log_model_pred, len(self.tokeniser)).permute((2, 0, 1))

            if verbose and (t <= 10 or t == 50 or (t) % 100 == 0):
                ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
                tokens = self.tokeniser.convert_ids_to_tokens(ids)
                sampled_mols = self.tokeniser.detokenise(tokens)

                m = sampled_mols[0]
                
                sampled_mol = m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m
                sampled_mol = sampled_mol.replace('?', '')
                sampled_mol = Chem.MolFromSmiles(sampled_mol)

                if sampled_mol is not None:
                    m = Chem.MolToSmiles(sampled_mol)

                if verbose:
                    print(f'{t}: {m}')

            if return_chain:
                ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
                tokens = self.tokeniser.convert_ids_to_tokens(ids)
                sampled_mols = self.tokeniser.detokenise(tokens)
                chain.append(sampled_mols)

        if verbose:
            print('-' * 20)

        ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)

        sampled_mols = [m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols]
        if clean:
            sampled_mols = [m.replace('?', '') for m in sampled_mols]

        if return_chain:
            return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0]), chain

        if return_lengths:
            return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0]), predicted_lengths.detach().cpu().numpy(), true_lengths.cpu().numpy()

        return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0])

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(len(self.tokeniser))
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(len(self.tokeniser))
        )

        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart
