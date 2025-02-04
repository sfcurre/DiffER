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
'''

class DiscreteDiffuser(nn.Module):
    def __init__(self, tokeniser, forward_pred, num_timesteps, max_seq_len, beta_schedule='cosine', pad_limit=20):
        super(DiscreteDiffuser, self).__init__()
        self.tokeniser = tokeniser
        self.forward_pred = forward_pred
        self.num_timesteps = num_timesteps
        self.max_seq_len = max_seq_len
        self.pad_limit = pad_limit
        self.pad_token_idx  = self.tokeniser.vocab[self.tokeniser.pad_token]
        
        alphas = 1 - get_named_beta_schedule(beta_schedule, num_diffusion_timesteps=num_timesteps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        self.log_alpha = log_alpha.float()
        self.log_1_min_alpha = log_1_min_alpha.float()
        self.log_cumprod_alpha = log_cumprod_alpha.float()
        self.log_1_min_cumprod_alpha = log_1_min_cumprod_alpha.float()

        self.Lt_history = torch.zeros(num_timesteps)
        self.Lt_count = torch.zeros(num_timesteps)

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
        input_token_ids = index_to_log_onehot(input_token_ids, len(self.tokeniser))
        input_pad_mask = torch.tensor(input_mask, dtype=torch.bool).transpose(0, 1)

        input_token_ids = input_token_ids[..., :self.max_seq_len]
        input_pad_mask = input_pad_mask[:self.max_seq_len]

        if noised:
            #importance sample t
            if t is None:
                t, _ = self.sample_time(size=len(input_tokens), method='importance')
            input_token_ids = self.q_sample(input_token_ids, t)
            return torch.permute(input_token_ids, (2, 0, 1)), input_pad_mask, t
        
        return torch.permute(input_token_ids, (2, 0, 1)), input_pad_mask

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = log_sample_categorical(log_EV_qxt_x0, len(self.tokeniser))
        return log_sample

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha.to(t.device), t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha.to(t.device), t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(len(self.tokeniser))
        )

        return log_probs
    
    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha.to(t.device), t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha.to(t.device), t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(len(self.tokeniser))
        )

        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1, device=t_minus_1.device), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start, device=log_x_start.device)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def sample_time(self, size, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(size, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=size, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (size,)).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError
        
    def update_Lt(self, t, kl):
        t = t.cpu()
        Lt2 = kl.pow(2).cpu()
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

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

        for t in reversed(range(1, self.num_timesteps)):
            # My code likes (time, batch, tokens)
            # MultiDiffusion code likes (batch, tokens, time)
            t_tensor = torch.full((length_mask.shape[0],), t, device=tgt_tokens.device)
            token_output = model.decode(tgt_tokens, length_mask, memory, memory_pad_mask, t_tensor)

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

        if verbose:
            print('-' * 20)

        ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)

        sampled_mols = [m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols]
        if clean:
            sampled_mols = [m.replace('?', '') for m in sampled_mols]

        return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0])
