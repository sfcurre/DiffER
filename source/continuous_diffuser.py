import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_util import *
from .rate_models import *
from rdkit import Chem, RDLogger

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and adapted from continuous discrete diffusion (https://github.com/andrew-cr/tauLDR)
'''

class ContinuousDiffuser(nn.Module):
    def __init__(self, tokeniser, forward_pred, num_timesteps, max_seq_len, rate_model=UniformRate, min_time=0.01, pad_limit=20):
        super(ContinuousDiffuser, self).__init__()
        self.tokeniser = tokeniser
        self.forward_pred = forward_pred
        self.num_timesteps = num_timesteps
        self.max_seq_len = max_seq_len
        self.rate_model = rate_model
        self.min_time = min_time
        self.pad_limit = pad_limit
        self.pad_token_idx  = self.tokeniser.vocab[self.tokeniser.pad_token]

        self.ratio_eps = 1e-9
        self.update_Lt = False

    def __call__(self, batch):
        return self._collate(batch)

    def _collate(self, batch):
        reacts_smiles, prods_smiles = tuple(zip(*batch))

        if self.forward_pred:
            encoder_smiles = reacts_smiles
            decoder_smiles = prods_smiles
        else:
            encoder_smiles = prods_smiles
            decoder_smiles = reacts_smiles

        # add some number of length-padding tokens
        lpad_token = self.tokeniser.unk_token
        
        if self.pad_limit is None:
            decoder_smiles_padded = decoder_smiles

        elif isinstance(self.pad_limit, int):
            if self.pad_limit > 0: 
                decoder_smiles_padded = tuple(smi + lpad_token * np.random.randint(1, self.pad_limit) for smi in decoder_smiles)
            elif self.pad_limit == 0:
                decoder_smiles_padded = decoder_smiles
            elif self.pad_limit == -1:
                decoder_smiles_padded = tuple(smi + lpad_token * self.max_seq_len for smi in decoder_smiles)

        encoder_input = self.tokeniser.tokenise(encoder_smiles, mask=False, pad=True)
        decoder_input = self.tokeniser.tokenise(decoder_smiles_padded, mask=False, pad=True)
        
        encoder_token_ids, encoder_pad_mask = self._partial_collate(encoder_input)
        # m_encoder_token_ids, m_encoder_pad_mask, m_encoder_t = self._partial_collate(encoder_input, noised=True)
        decoder_token_ids, decoder_pad_mask = self._partial_collate(decoder_input)
        m_decoder_token_ids, m_decoder_pad_mask, m_decoder_t = self._partial_collate(decoder_input, noised=True)
        
        pad_index = self.tokeniser.vocab[lpad_token]
        collate_output = {
            "encoder_input": encoder_token_ids,
            "encoder_pad_mask": encoder_pad_mask,
            "decoder_input": m_decoder_token_ids,
            "decoder_pad_mask": m_decoder_pad_mask,
            "target": decoder_token_ids.max(dim=-1)[1],
            "target_onehots": decoder_token_ids,
            "target_mask": decoder_pad_mask,
            "target_smiles": decoder_smiles,
            "target_padding": (decoder_token_ids == pad_index).sum(dim=0),
            # "masked_encoder_input": m_encoder_token_ids,
            # "masked_encoder_pad_mask": m_encoder_pad_mask,
            "encoder_smiles": encoder_smiles,
            "decoder_t": m_decoder_t,
        }
        
        return collate_output

    def _partial_collate(self, inputs, noised=False, t=None):
        input_tokens = inputs["original_tokens"]
        input_mask = inputs["original_pad_masks"]
        
        input_token_ids = self.tokeniser.convert_tokens_to_ids(input_tokens)
        input_token_ids = torch.tensor(input_token_ids)

        if noised:
            #importance sample t
            if t is None:
                t = self.sample_time(size=len(input_tokens))
            input_token_ids = self.q_sample(input_token_ids, t)

        input_token_ids = F.one_hot(input_token_ids, len(self.tokeniser))
        input_pad_mask = torch.tensor(input_mask, dtype=torch.bool).transpose(0, 1)

        input_token_ids = input_token_ids[..., :self.max_seq_len]
        input_pad_mask = input_pad_mask[:self.max_seq_len]

        if noised:
            return torch.permute(input_token_ids, (1, 0, 2)), input_pad_mask, t
        
        return torch.permute(input_token_ids, (1, 0, 2)), input_pad_mask

    def q_sample(self, x_start, t):
        B, D = x_start.shape
        S = len(self.tokeniser)
        device = x_start.device
        
        qt0 = self.rate_model.transition(t).to(device)
        rate = self.rate_model.rate(t).to(device)

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            x_start.flatten().long(),
            :
        ] # (B*D, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            x_t.long().flatten(),
            :
        ] # (B*D, S)
        rate_vals_square[
            torch.arange(B*D, device=device),
            x_t.long().flatten()
        ] = 0.0 # 0 the diagonals

        rate_vals_square = rate_vals_square.view(B, D, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, D)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )

        square_dims = square_dimcat.sample() # (B,) taking values in [0, D)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device),
            square_dims,
            :
        ] # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample() # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples
        # x_tilde (B, D)
        return x_tilde
        
    def sample_time(self, size):
        return torch.rand((size,)) * (1.0 - self.min_time) + self.min_time
        
    def get_lengths_from_padding(self, pad_mask):
        lengths = len(pad_mask) - pad_mask.sum(0).unsqueeze(-1)
        return lengths.squeeze()

    def get_length_mask(self, lengths):
        max_len = lengths.max().item()
        length_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool, device=lengths.device), 1)
        length_mask = torch.stack([length_mask[lengths[batch] - 1] for batch in range(len(lengths))], dim=0)
        return length_mask.squeeze()

    def init_noise(self, target_lengths):
        length_mask = self.get_length_mask(target_lengths)
        uniform_logits = torch.zeros((length_mask.shape[0], len(self.tokeniser), length_mask.shape[1]), device=length_mask.device)
        tgt_tokens = log_sample_categorical(uniform_logits, len(self.tokeniser)).permute(2, 0, 1)
        
        pad_token =  index_to_log_onehot(torch.tensor([[self.pad_token_idx]], device=length_mask.device), len(self.tokeniser)).permute(2, 0, 1)
        tgt_tokens = (~length_mask.transpose(0, 1).unsqueeze(-1)) * tgt_tokens + length_mask.transpose(0, 1).unsqueeze(-1) * pad_token 
        return tgt_tokens, length_mask

    def sample(self, batch, model, verbose=True, pred_lengths=True, clean=True):
        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        memory, memory_pad_mask, predicted_lengths = model.encode(encoder_input, encoder_pad_mask)
            
        true_lengths = self.get_lengths_from_padding(batch['target_mask'])
        if pred_lengths:
            if self.pad_limit == -1:
                lengths[:] = self.max_seq_len
            else:
                # lengths = torch.multinomial(torch.exp(predicted_lengths), num_samples=1).squeeze()
                lengths = predicted_lengths.max(dim=-1)[1]
                # leverage that change in length will be less than half the size of the product, use large indices for negative change
                lengths[lengths > self.max_seq_len / 2] = lengths[lengths > self.max_seq_len / 2] - self.max_seq_len
                lengths = self.get_lengths_from_padding(batch['encoder_pad_mask']) + lengths
        else:
            lengths = true_lengths

        tgt_tokens, length_mask = self.init_noise(lengths)
    
        if verbose:
            print(f'target: {batch["target_smiles"][0]}')

        ts = np.concatenate((np.linspace(1.0, self.min_time, self.num_timesteps), np.array([0])))
        device = tgt_tokens.device
        D, B, S = tgt_tokens.shape

        for idx, t in enumerate(ts[0:-1]):
            h = ts[idx] - ts[idx+1]
            t_tensor = torch.full((length_mask.shape[0],), t, device=self.rate_model.device)

            qt0 = self.rate_model.transition(t_tensor).to(device)
            rate = self.rate_model.rate(t_tensor).to(device)

            token_output = model.decode(tgt_tokens, length_mask, memory, memory_pad_mask, t_tensor.to(device))            
            p0t = F.softmax(token_output, dim=2).transpose(0, 1) # (B, D, S)
            tgt_tokens = tgt_tokens.max(dim=-1)[1].transpose(0, 1)
            
            qt0_denom = qt0[
                torch.arange(B, device=device).repeat_interleave(D*S),
                torch.arange(S, device=device).repeat(B*D),
                tgt_tokens.long().flatten().repeat_interleave(S)
            ].view(B,D,S) + self.ratio_eps

            # First S is x0 second S is x tilde

            qt0_numer = qt0 # (B, S, S)

            forward_rates = rate[
                torch.arange(B, device=device).repeat_interleave(D*S),
                torch.arange(S, device=device).repeat(B*D),
                tgt_tokens.long().flatten().repeat_interleave(S)
            ].view(B, D, S)

            inner_sum = (p0t / qt0_denom) @ qt0_numer # (B, D, S)

            reverse_rates = forward_rates * inner_sum # (B, D, S)

            reverse_rates[
                torch.arange(B, device=device).repeat_interleave(D),
                torch.arange(D, device=device).repeat(B),
                tgt_tokens.long().flatten()
            ] = 0.0

            diffs = torch.arange(S, device=device).view(1,1,S) - tgt_tokens.view(B,D,1)
            poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
            jump_nums = poisson_dist.sample()
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = tgt_tokens + overall_jump
            x_new = torch.clamp(xp, min=0, max=S-1)

            tgt_tokens = F.one_hot(x_new.long(), num_classes=len(self.tokeniser)).transpose(0, 1)

            if verbose and (idx <= 10 or idx == 50 or (idx) % 100 == 0):
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

        if verbose:
            print('-' * 20)

        tgt_tokens = model.decode(tgt_tokens, length_mask, memory, memory_pad_mask, t_tensor.to(device))
        
        ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)

        sampled_mols = [m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols]
        if clean:
            sampled_mols = [m.replace('?', '') for m in sampled_mols]

        return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0])

    def calc_elbo(self, ts, x_logits, x_tilde, x_start):
        x_logits = x_logits.permute(1, 0, 2)
        x_tilde = x_tilde.permute(1, 0, 2)
        x_start = x_start.permute(1, 0, 2)
        device = x_logits.device

        qt0 = self.rate_model.transition(ts.to(self.rate_model.device)).to(device)
        rate = self.rate_model.rate(ts.to(self.rate_model.device)).to(device)

        B, D, S = x_start.shape
        x_start = x_start.max(dim=-1)[1]
        x_tilde = x_tilde.max(dim=-1)[1]

        # ---------- First term of ELBO (regularization) ---------------
        p0t_sig = F.softmax(x_logits, dim=2) # (B, D, S)
        p0t_reg = p0t_sig
        reg_x = x_tilde


        # For (B, D, S, S) first S is x_0 second S is x'
        mask_reg = torch.ones((B,D,S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            reg_x.long().flatten()
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)
        
        qt0_denom_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1,2) # (B, D, S)

        reg_term = torch.sum(
            (p0t_reg / qt0_denom_reg) * reg_tmp,
            dim=(1,2)
        )

        # ----- second term of continuous ELBO (signal term) ------------

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            x_start.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        outer_qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            x_start.long().flatten(),
            x_tilde.long().flatten()
        ] + self.ratio_eps # (B, D)


        qt0_numer_sig = qt0.view(B, S, S) # first S is x_0, second S is x


        qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_tilde.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        ) # (B, D, S)


        x_tilde_mask = torch.ones((B,D,S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            x_tilde.long().flatten()
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B,D,S)

        outer_sum_sig = torch.sum(
            x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B,D,1)) * inner_log_sig,
            dim=(1,2)
        )

        # now getting the 2nd term normalization

        rate_row_sums = - rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B)
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(D),
            x_tilde.long().flatten()
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp # (B,D)
        Z_addition = rate_row_sums

        Z_sig_norm = base_Z.view(B, 1, 1) - \
            Z_subtraction.view(B, D, 1) + \
            Z_addition.view(B, 1, S)

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, D, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            x_start.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        qt0_sig_norm_denom = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            x_start.long().flatten(),
            x_tilde.long().flatten()
        ].view(B, D) + self.ratio_eps


        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B,D,1)),
            dim=(1,2)
        )

        sig_mean = torch.mean(- outer_sum_sig/sig_norm)

        reg_mean = torch.mean(reg_term)

        neg_elbo = sig_mean + reg_mean

        return neg_elbo