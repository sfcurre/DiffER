import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discrete_diffusion import *
from .utils import noise_schedule
from rdkit import Chem, RDLogger

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and adapted from unified discrete diffusion (https://github.com/lingxiaoshawn/usd3?tab=readme-ov-file)
'''

class UnifiedSampler(nn.Module):
    def __init__(self, model, diffuser, tokeniser, num_timesteps, max_seq_len, min_time=0.01):
        super(UnifiedSampler, self).__init__()
        self.diffuser = diffuser
        self.tokeniser = tokeniser
        self.num_timesteps = num_timesteps
        self.max_seq_len = max_seq_len
        self.min_time = min_time
        self.pad_token_idx  = self.tokeniser.vocab[self.tokeniser.pad_token]

        self.ratio_eps = 1e-9
        self.update_Lt = False
        
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
        x_t = sample_uniform_categorical(length_mask.shape, self.num_classes, device=length_mask.device)
        x_t[length_mask] = self.pad_token_idx 
        x_t = F.one_hot(x_t)
        return x_t, length_mask

    def sample(self, batch, model, verbose=True, pred_lengths=True, clean=True, record_attns=False, num_samples=1):

        memory, memory_pad_mask, predicted_lengths = model.encode(batch['y_0'], batch['y_mask'])
            
        if pred_lengths:
            lengths = predicted_lengths.max(dim=-1)[1]
            if self.pad_limit == -1:
                lengths[:] = self.max_seq_len
            else:
                # leverage that change in length will be less than half the size of the product, use large indices for negative change
                lengths[lengths > self.max_seq_len / 2] = lengths[lengths > self.max_seq_len / 2] - self.max_seq_len
                lengths = self.get_lengths_from_padding(batch['encoder_pad_mask']) + lengths
        else:
            lengths = self.get_lengths_from_padding(batch['target_mask'])

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
                ids = x_t.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
                tokens = self.tokeniser.convert_ids_to_tokens(ids)
                sampled_mols = self.tokeniser.detokenise(tokens)
                m = sampled_mols[0]
                out_attns[idx + 1] = [m]

            s = ts[idx+1]
            t_tensor = torch.full((length_mask.shape[0],), t, device=self.rate_model.device)
            s_tensor = torch.full((length_mask.shape[0],), s, device=self.rate_model.device)            

            logits = model.decode(x_t, length_mask, memory, memory_pad_mask, t_tensor.to(device))
            fprob_t = F.softmax(logits, dim=2)
            x_t = x_t.max(dim=-1)
            
            prob_s = self.diffuser.ps_t_prob(fprob_t, x_t, t_tensor, s_tensor).type(torch.float)
            prob_s[s==0] = fprob_t[s==0]

            x_t = torch.multinomial(prob_s)
            x_t = F.one_hot(x_t, len(self.tokeniser))

            if verbose and (idx <= 10 or idx == 50 or (idx) % 100 == 0):
                ids = x_t.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
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

        x_t = model.decode(x_t, length_mask, memory, memory_pad_mask, t_tensor.to(device))
        
        ids = x_t.max(dim=-1)[1].cpu().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)

        sampled_mols = [m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols]
        if clean:
            sampled_mols = [m.replace('?', '') for m in sampled_mols]

        if record_attns:
            return sampled_mols, torch.log(x_t.max(dim=-1)[0]), in_attns, out_attns

        return sampled_mols, torch.log(x_t.max(dim=-1)[0])
