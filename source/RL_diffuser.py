import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_util import *
from .rate_models import *
from .continuous_diffuser import ContinuousDiffuser
from rdkit import Chem, RDLogger

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and adapted from continuous discrete diffusion (https://github.com/andrew-cr/tauLDR)
'''

class RLDiffuser(ContinuousDiffuser):
    def __init__(self, *args, **kwargs):
        super(RLDiffuser, self).__init__(
            *args, **kwargs
        )
        
    def sample(self, batch, model, guidance_model, optimizer, gamma=0.5, verbose=True, pred_lengths=True, clean=True, record_attns=False):
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

        if record_attns:
            in_attns = model.get_attn('encoder')
            out_attns = {}

        ts = np.concatenate((np.linspace(1.0, self.min_time, self.num_timesteps), np.array([0])))
        device = tgt_tokens.device
        D, B, S = tgt_tokens.shape

        for idx, t in enumerate(ts[0:-1]):
            if record_attns and (idx + 1) in [1, 10, 50, 100, 150, 200]:
                ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
                tokens = self.tokeniser.convert_ids_to_tokens(ids)
                sampled_mols = self.tokeniser.detokenise(tokens)
                m = sampled_mols[0]
                out_attns[idx + 1] = [m]

            h = ts[idx] - ts[idx+1]
            t_tensor = torch.full((length_mask.shape[0],), t, device=self.rate_model.device)

            qt0 = self.rate_model.transition(t_tensor).to(device)
            rate = self.rate_model.rate(t_tensor).to(device)

            token_output = model.decode(tgt_tokens, length_mask, memory, memory_pad_mask, t_tensor.to(device))            
            token_log_prob = F.log_softmax(token_output, dim=2)

            # Run guidance model on token_output
            ##############################
            classifier_output = guidance_model(tgt_tokens, length_mask, t_tensor.to(device))
            classifier_log_prob = F.log_softmax(classifier_output, dim=2)
            classifier_log_prob.sum().backward(retain_graph=True)
            classifier_grad = classifier_log_prob.grad

            classifier_log_prob_ratio = (
                classifier_grad - (tgt_tokens * classifier_grad).sum(dim=-1, keepdim=True)
            ).detach().requires_grad_(False)
            classifier_log_prob = (
                classifier_log_prob_ratio +
                classifier_log_prob[..., None, None]
            ).detach().requires_grad_(False)

            valid_scores, memory_scores = guidance_model.get_scores(tgt_tokens)

            optimizer.zero_grad()
            classifier_loss = F.binary_cross_entropy_with_logits(classifier_output, memory_scores)
            classifier_loss.sum().backward()
            optimizer.step()

            # only apply guidance to valid molecules
            guided_output = valid_scores * (gamma * classifier_log_prob) + token_log_prob
            p0t = F.exp(guided_output, dim=2).transpose(0, 1) # (B, D, S)
            ##############################

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
            
            if record_attns and (idx + 1) in [1, 10, 50, 100, 150, 200]:
                out_attns[idx + 1].append(model.get_attns('decoder'))

        if verbose:
            print('-' * 20)

        tgt_tokens = model.decode(tgt_tokens, length_mask, memory, memory_pad_mask, t_tensor.to(device))
        
        ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)

        sampled_mols = [m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols]
        if clean:
            sampled_mols = [m.replace('?', '') for m in sampled_mols]

        if record_attns:
            return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0]), in_attns, out_attns

        return sampled_mols, torch.log(tgt_tokens.max(dim=-1)[0])
