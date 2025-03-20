import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger

from utils import canonicalize

'''
This code is heavily inspired by Chemformer (https://github.com/MolecularAI/Chemformer)
and multinomial diffusion (https://github.com/ehoogeboom/multinomial_diffusion/tree/main)
'''

class ParticleGuidanceModel(nn.Module):
    def __init__(self,
        conditional_model,
        ):
        super(ParticleGuidanceModel, self).__init__()

        self.tokeniser = conditional_model.tokeniser
        self.max_seq_len = conditional_model.max_seq_len

        self.vocab_size = conditional_model.vocab_size
        self.pad_token_idx = conditional_model.pad_token_idx

        self.emb = conditional_model.emb.clone()
        self.time_emb = conditional_model.time_emb
        self.dropout = conditional_model.dropout

        self.length_rep = conditional_model.length_rep
        self.encoder = conditional_model.encoder.clone()

        self.output_fc = nn.Linear(conditional_model.d_model, 1)

        self._init_params()
        self.register_buffer("pos_emb", self.positional_embs())

        RDLogger.DisableLog("rdApp.*")

    def forward(self, encoder_input, encoder_pad_mask):
        encoder_embs = self.embed_log_probs(encoder_input)
        
        len_tokens = self.length_rep(torch.zeros(1, encoder_embs.size(1), dtype=torch.int32, device=encoder_embs.device))
        encoder_embs = torch.cat([len_tokens, encoder_embs], dim=0)
        encoder_pad_mask = torch.cat([encoder_pad_mask[:, :1], encoder_pad_mask], dim=-1)

        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output = model_output.mean(dim=1)

        distance = model_output.unsqueeze(1) - model_output.unsqueeze(0)
        output = self.output_fc(distance)

        return output

    def embed_log_probs(self, log_probs, t=None):
        seq_len, _, _ = tuple(log_probs.size())

        onehot_input = torch.exp(log_probs)
        onehot_embs = torch.matmul(onehot_input, self.emb.weight)
        onehot_embs = onehot_embs * np.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        onehot_embs = onehot_embs + positional_embs
        if t is not None:
            time_embs = self.time_emb(t)
            onehot_embs += time_embs
        onehot_embs = self.dropout(onehot_embs)
        return onehot_embs
    
    def get_canonical(self, tgt_tokens):
        ids = tgt_tokens.max(dim=-1)[1].transpose(0, 1).cpu().numpy()
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        sampled_mols = self.tokeniser.detokenise(tokens)
        sampled_mols = list(map(canonicalize, (m[:m.find('<PAD>')] if m.find('<PAD>') > 0 else m for m in sampled_mols)))
        return sampled_mols
        
    def get_distance_scores(self, sampled_mols):
        scores = []
        for m1 in sampled_mols:
            scores.append([])
            for m2 in sampled_mols:
                if m1 == m2:
                    scores[-1].append(0)
                else:
                    scores[-1].append(1)
        return torch.tensor(scores)

    def get_valid_scores(self, sampled_mols):
        scores = []
        for m in sampled_mols:
            if m is None:
                scores.append(0)
            else:
                scores.append(1)
        return torch.tensor(scores)
    
    def get_scores(self, tgt_tokens):
        sampled_mols = self.get_canonical(tgt_tokens)
        valid_scores = self.get_valid_scores(sampled_mols)
        distance_scores = self.get_distance_scores(sampled_mols)
        return valid_scores, distance_scores
    
    def get_loss(self, output, scores):
        classifier_loss = F.binary_cross_entropy_with_logits(output, scores)
        return classifier_loss