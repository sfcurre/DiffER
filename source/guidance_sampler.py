import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampler import UnifiedSampler
from .discrete_diffusion import *

from rdkit import Chem, RDLogger

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and adapted from continuous discrete diffusion (https://github.com/andrew-cr/tauLDR)
'''

class GuidanceSampler(UnifiedSampler):
    def __init__(self, *args, **kwargs):
        super(GuidanceSampler, self).__init__(
            *args, **kwargs
        )
        
    def sample(self, batch, model, guidance_model, optimizer, gamma=0.5, verbose=True, pred_lengths=True, clean=True, record_attns=False, num_samples=1):
        
        memory, memory_pad_mask, predicted_lengths = model.encode(batch['y_0'], batch['y_mask'])
            
        if pred_lengths:
            lengths = predicted_lengths.max(dim=-1)[1]
            if self.pad_limit == -1:
                lengths[:] = self.max_seq_len
            else:
                # leverage that change in length will be less than half the size of the product, use large indices for negative change
                lengths[lengths > self.max_seq_len / 2] = lengths[lengths > self.max_seq_len / 2] - self.max_seq_len
                lengths = self.get_lengths_from_padding(batch['y_mask']) + lengths
        else:
            lengths = self.get_lengths_from_padding(batch['x_mask'])

        if num_samples > 1:
            lengths = lengths.repeat(num_samples)
            memory = memory.repeat(num_samples, 1, 1)
            memory_pad_mask = memory_pad_mask.repeat(num_samples, 1)

        x_t, length_mask = self.init_noise(lengths)
    
        if verbose:
            print(f'target: {batch["target_smiles"][0]}')

        if record_attns:
            in_attns = model.get_attn('encoder')
            out_attns = {}

        ts = np.concatenate((np.linspace(1.0, self.min_time, self.num_timesteps), np.array([0])))
        device = x_t.device

        for idx, t in enumerate(ts[0:-1]):
            if record_attns and (idx + 1) in [1, 10, 50, 100, 150, 200]:
                ids = x_t.max(dim=-1)[1].cpu().numpy()
                tokens = self.tokeniser.convert_ids_to_tokens(ids)
                sampled_mols = self.tokeniser.detokenise(tokens)
                m = sampled_mols[0]
                out_attns[idx + 1] = [m]

            s = ts[idx+1]
            t_tensor = torch.full((length_mask.shape[0],), t, device=device)
            s_tensor = torch.full((length_mask.shape[0],), s, device=device)            

            logits = model.decode(x_t, length_mask, memory, memory_pad_mask, t_tensor)

            # Run guidance model on token_output
            ##############################
            log_x_t = torch.log(x_t)
            log_x_t.requires_grad = True
            classifier_output = guidance_model.forward(log_x_t, length_mask)
            classifier_log_prob = F.log_softmax(classifier_output, dim=-1)
            classifier_log_prob.sum().backward(retain_graph=True)
            classifier_log_prob = log_x_t.grad

            # classifier_log_prob_ratio = (
            #     classifier_grad - (x_t * classifier_grad).sum(dim=-1, keepdim=True)
            # ).detach().requires_grad_(False)
            # classifier_log_prob = (
            #     classifier_log_prob_ratio +
            #     classifier_log_prob[..., None]
            # ).detach().requires_grad_(False)

            valid_scores, scores = guidance_model.get_scores(x_t)

            optimizer.zero_grad()
            classifier_loss = guidance_model.get_loss(classifier_output, scores.to(classifier_output.device))
            classifier_loss.sum().backward()
            optimizer.step()

            ##############################
            fprob_t = F.softmax(logits, dim=-1)
            x_t = x_t.max(dim=-1)[1]
            
            prob_s = self.diffuser.ps_t_prob(fprob_t, x_t, t_tensor, s_tensor).type(torch.float)
            prob_s[s==0] = fprob_t[s==0]

            # apply guidance
            diffusion_log_probs = torch.log(prob_s)
            guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs

            x_s = sample_categorical(torch.exp(guided_log_probs))
            x_s[batch['x_mask']] = x_t[batch['x_mask']]
            x_t = F.one_hot(x_s, len(self.tokeniser)).to(torch.float)
            
            if verbose and (idx <= 10 or idx == 50 or (idx) % 100 == 0):
                ids = x_t.max(dim=-1)[1].cpu().numpy()
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

        x_t = model.decode(x_t, length_mask, memory, memory_pad_mask, t_tensor)
        
        ids = x_t.max(dim=-1)[1].cpu().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)

        sampled_mols = [m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols]
        if clean:
            sampled_mols = [m.replace('?', '') for m in sampled_mols]

        if record_attns:
            return sampled_mols, torch.log(x_t.max(dim=-1)[0]), in_attns, out_attns

        return sampled_mols, torch.log(x_t.max(dim=-1)[0])
