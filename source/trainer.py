import time
import numpy as np
import glob, os, json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger

from .utils import move_batch_to_gpu

'''
This code is inspired by https://github.com/ehoogeboom/multinomial_diffusion/tree/main
'''

class UnifiedTrainer:
    def __init__(self, model, optimizer, diffuser, sampler, name='Default', length_loss = 'cross_entropy', coeff_ce=1., coeff_vlb=1., use_gpu=True, min_time=0.01, moe_loss=0):
        self.model = model
        self.optimizer = optimizer
        self.diffuser = diffuser
        self.sampler = sampler
        self.name = name
        self.length_loss = length_loss
        self.coeff_ce = coeff_ce
        self.coeff_vlb = coeff_vlb
        self.use_gpu = use_gpu
        self.min_time = min_time # TODO
        self.moe_loss = moe_loss

        RDLogger.DisableLog("rdApp.*")

    def train(self, dataloaders, epochs, val_limit=100, pred_lengths=True):
        # Train model
        t_total = time.time()
        loss_values = []
        for epoch in range(epochs):
            print(f'Epoch {epoch} - {time.time() - t_total}')
            epoch_losses = []
            for i, batch in enumerate(dataloaders['train']):
                epoch_losses.append(self.train_step(batch))
                    
            with torch.no_grad():
                self.print_metrics(dataloaders['val'], str(epoch) + f'.{i+1} - {time.time() - t_total}',
                                   val_limit, pred_lengths=pred_lengths)

            torch.save(self.model.state_dict(), 'out/models/{}.pkl'.format(self.name))
            loss_values.append(np.mean(epoch_losses))
            np.save(f'out/losses/{self.name}_losses.npy', np.array(loss_values))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        return loss_values

    def print_metrics(self, val_loader, epoch, val_limit, pred_lengths=True):
        self.model.eval()
        metrics = defaultdict(list)
        for i, batch in enumerate(val_loader):
            if i == val_limit:
                break
            batch_metrics, _ = self.val_step(batch, pred_lengths=pred_lengths)

            for key, score in batch_metrics.items():
                metrics[key].append(score)

        log = f'Epoch - {epoch} | ' + ' | '.join(f'{key} - {sum(l) / len(l)}' for key, l in metrics.items())
        with open(f'out/metrics/{self.name}_metrics_log.txt', 'a') as fp:
            print(log + '\n', file=fp)

    def sample_time(self, size, device):
        return torch.rand((size,), device=device) * (1.0 - self.min_time) + self.min_time
        
    def train_step(self, batch):
        if self.use_gpu:
            move_batch_to_gpu(batch)

        self.model.train()
        self.optimizer.zero_grad()
        
        batch['t'] = self.sample_time(size=len(batch['x_0']), device=batch['x_0'].device)
        x_t = self.diffuser.qt_0_sample(batch['x_0'].max(dim=-1)[1], batch['t'], conditional_mask=batch['x_mask'])
        batch['x_t'] = F.one_hot(x_t, len(self.sampler.tokeniser)).to(torch.float)
        
        output, lengths = self.model.forward(batch)

        total_loss = self._calc_loss(batch, output)['loss']
        if self.sampler.pad_limit > -1:
            total_loss += self._calc_length_loss(batch, lengths)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.cpu().item()

    def val_step(self, batch, pred_lengths=True):
        if self.use_gpu:
            move_batch_to_gpu(batch)

        batch['t'] = self.sample_time(size=len(batch['x_0']), device=batch['x_0'].device)
        x_t = self.diffuser.qt_0_sample(batch['x_0'].max(dim=-1)[1], batch['t'], conditional_mask=batch['x_mask'])
        batch['x_t'] = F.one_hot(x_t, len(self.sampler.tokeniser)).to(torch.float)
        
        self.model.eval()
        output, lengths = self.model.forward(batch)
        loss = self._calc_loss(batch, output)['loss']
        length_loss = self._calc_length_loss(batch, lengths)
        token_acc = self._calc_token_acc(batch, output)
        perplexity = self._calc_perplexity(batch, output)

        sampled_smiles, _ = self.sampler.sample(batch, self.model, verbose=True, pred_lengths=pred_lengths)
        sampling_metrics = self._calc_sampling_metrics(batch, sampled_smiles)

        metrics = dict(val_loss=loss.cpu(),
                       length_loss=length_loss.cpu(),
                       token_accuracy=token_acc,
                       perplexity=perplexity,
                       mol_accuracy=sampling_metrics['accuracy'],
                       mol_invalid=sampling_metrics['invalid'])

        return metrics, sampled_smiles

    def _calc_loss(self, batch, x_logits):
        loss = self.diffuser.compute_loss( 
                     x_logits,
                     batch['x_t'].max(dim=-1)[1], 
                     batch['x_0'].max(dim=-1)[1],
                     batch['t'], 
                     m=None, 
                     coeff_ce=self.coeff_ce,
                     coeff_vlb=self.coeff_vlb,
                     conditional_mask=batch['x_mask'],
                     simplified_vlb=False)
        if self.moe_loss:
            loss['aux_loss'] = self.model.get_aux_loss()
            loss['loss'] += self.moe_loss * loss['aux_loss']
        return loss
    
    def _calc_length_loss(self, batch_input, pred_lengths):
        pred_lengths = F.log_softmax(pred_lengths)
        input_length = batch_input['y_mask'].shape[1] - batch_input['y_mask'].sum(1).unsqueeze(-1)
        output_length = batch_input['x_mask'].shape[1] - batch_input['x_mask'].sum(1).unsqueeze(-1)
        # leverage the fact that the change in length will be small, so large indices can be used for negative length change
        length_target = ((output_length - input_length) % self.sampler.max_seq_len).to(torch.int64)
        if self.length_loss == 'cross_entropy':
            length_loss = -pred_lengths.gather(dim=-1, index=length_target)
        elif self.length_loss == 'focal':
            gamma = 0.25
            length_loss = -pred_lengths.gather(dim=-1, index=length_target)
            length_dist = torch.exp(-length_loss)
            focal_mod = (1 - length_dist) ** gamma
            length_loss *= focal_mod
        return length_loss.mean()

    def _calc_token_acc(self, batch_input, token_output):
        token_ids = batch_input["x_0"].max(dim=-1)[1]
        target_mask = batch_input["x_mask"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def _calc_perplexity(self, batch_input, vocab_dist_output):
        target_ids = batch_input["x_0"].max(dim=-1)[1]
        target_mask = batch_input["x_mask"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = - (1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_sampling_metrics(self, batch_input, sampled_smiles):
        target_smiles = batch_input['target_smiles']
        mol_targets = [Chem.MolFromSmiles(smi) for smi in target_smiles]
        canon_targets = [Chem.MolToSmiles(mol) for mol in mol_targets]
        sampled_mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
        invalid = [mol is None for mol in sampled_mols]

        canon_smiles = ["Unknown" if mol is None else Chem.MolToSmiles(mol) for mol in sampled_mols]
        correct_smiles = [canon_targets[idx] == smi for idx, smi in enumerate(canon_smiles)]

        num_correct = sum(correct_smiles)
        total = len(correct_smiles)
        num_invalid = sum(invalid)
        perc_invalid = num_invalid / total
        accuracy = num_correct / total

        metrics = {
            "accuracy": accuracy,
            "invalid": perc_invalid
        }

        return metrics
