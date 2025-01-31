import time
import numpy as np
import glob, os, json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger

'''
This code is inspired by https://github.com/ehoogeboom/multinomial_diffusion/tree/main
'''

class DiffusionModelTrainer:
    def __init__(self, model, optimizer, diffuser, name='Default', loss_components=['nll'], length_loss = 'cross_entropy', use_gpu=True):
        self.model = model
        self.optimizer = optimizer
        self.diffuser = diffuser
        self.name = name
        self.loss = loss_components
        self.length_loss = length_loss
        self.use_gpu = use_gpu
        
        RDLogger.DisableLog("rdApp.*")

    def train(self, dataloaders, epochs, patience, val_limit=100, pred_lengths=True):
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        for epoch in range(epochs):
            print(f'Epoch {epoch} - {time.time() - t_total}')
            epoch_losses = []
            for i, batch in enumerate(dataloaders['train']):
                epoch_losses.append(self.train_step(batch))
                    
            with torch.no_grad():
                self.print_metrics(dataloaders['val'], str(epoch) + f'.{i+1} - {time.time() - t_total}',
                                   val_limit, pred_lengths=pred_lengths)
            torch.save(self.model.state_dict(), 'out/models/{}_{}.pkl'.format(self.name, epoch))
            loss_values.append(np.mean(epoch_losses))
            np.save(f'out/losses/{self.name}_losses.npy', np.array(loss_values))

            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience:
                break

            files = glob.glob(f'out/models/{self.name}_*.pkl')
            for file in files:
                epoch_nb = int(file.split('_')[-1].split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        # files = glob.glob(f'out/models/{self.name}_*.pkl')
        # for file in files:
        #     epoch_nb = int(file.split('_')[-1].split('.')[0])
        #     if epoch_nb > best_epoch:
        #         os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        self.model.load_state_dict(torch.load('out/models/{}_{}.pkl'.format(self.name, best_epoch)))

        # Testing
        return loss_values

    def print_metrics(self, val_loader, epoch, val_limit, pred_lengths=True):
        self.model.eval()
        metrics = defaultdict(list)
        mols = []
        for i, batch in enumerate(val_loader):
            if i == val_limit:
                break
            batch_metrics, sampled_mols = self.val_step(batch, pred_lengths=pred_lengths)
            
            for j, sample in enumerate(sampled_mols):
                data = {}
                data['target'] = batch["target_smiles"][j]
                data['sample'] = sample
                data['source'] = batch["encoder_smiles"][j]
                mols.append(data)

            for key, score in batch_metrics.items():
                metrics[key].append(score)

        log = f'Epoch - {epoch} | ' + ' | '.join(f'{key} - {sum(l) / len(l)}' for key, l in metrics.items())
        with open(f'out/metrics/{self.name}_metrics_log.txt', 'a') as fp:
            print(log + '\n', file=fp)
 
        # with open(f'out/samples/{self.name}/sampled_mols_e{epoch.split()[0]}.json', 'w') as fp:
        #     json.dump(mols, fp)

    def move_batch_to_gpu(self, batch):
        for key, value in batch.items():
            if hasattr(value, 'cuda'):
                batch[key] = value.cuda()
        batch['device'] = 'cuda'
        
    def train_step(self, batch):
        if self.use_gpu:
            self.move_batch_to_gpu(batch)

        self.model.train()
        self.optimizer.zero_grad()
        
        output, lengths = self.model.forward(batch)
        loss = self._calc_loss(batch, output)['loss']
        total_loss = loss + self._calc_length_loss(batch, lengths)
        total_loss.backward()

        self.optimizer.step()
        return loss.cpu().item()

    def val_step(self, batch, pred_lengths=True):
        if self.use_gpu:
            self.move_batch_to_gpu(batch)

        self.model.eval()
        output, lengths = self.model.forward(batch)
        loss = self._calc_loss(batch, output)['loss']
        length_loss = self._calc_length_loss(batch, lengths)
        token_acc = self._calc_token_acc(batch, output)
        perplexity = self._calc_perplexity(batch, output)

        sampled_smiles, lprobs = self.diffuser.sample(batch, self.model, verbose=True, pred_lengths=pred_lengths)
        sampling_metrics = self._calc_sampling_metrics(batch, sampled_smiles)

        metrics = dict(val_loss=loss.cpu(),
                       length_loss=length_loss.cpu(),
                       token_accuracy=token_acc,
                       perplexity=perplexity,
                       mol_accuracy=sampling_metrics['accuracy'],
                       mol_invalid=sampling_metrics['invalid'])

        return metrics, sampled_smiles

    def _calc_loss(self, batch_input, token_output, update_Lt=True):
        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        x_start = batch_input["target_onehots"]
        t = batch_input['decoder_t']

        loss_terms = {}    
        
        if 'nll' in self.loss or 'vb' in self.loss:
            lprobs = F.log_softmax(token_output, dim=-1)
            non_pad_mask = tokens.ne(self.diffuser.pad_token_idx)
            nll_loss = -lprobs.gather(dim=-1, index=tokens[..., None])
            nll_loss = nll_loss.squeeze() * non_pad_mask
            nll_loss = nll_loss.sum(dim=0)
            loss_terms['nll'] = nll_loss.mean()

        if 'mse' in self.loss:
            probs = F.softmax(token_output, dim=-1)
            mse_loss = (x_start - probs) ** 2
            mse_loss = mse_loss.sum(dim=(0, 2))
            loss_terms['mse'] = mse_loss.mean()

        if 'kl' in self.loss or 'vb' in self.loss or update_Lt:
            log_x_t = batch_input['decoder_input'].permute((1, 2, 0))
            log_true_prob = self.diffuser.q_posterior(torch.log_softmax(x_start, dim=-1).permute((1, 2, 0)), log_x_t, t)
            log_model_prob = self.diffuser.q_posterior(torch.log_softmax(token_output, dim=-1).permute((1, 2, 0)), log_x_t, t)
            kl = -(log_true_prob.exp() * (log_true_prob - log_model_prob))
            kl = kl.sum(dim=(1, 2))
            if update_Lt:
                self.diffuser.update_Lt(t, kl)
            loss_terms['kl'] = kl.mean()

        if 'vb' in self.loss:
            mask = (t == torch.zeros_like(t)).float()
            vb_loss = mask * nll_loss + (1. - mask) * kl
            loss_terms['vb'] = vb_loss.mean()
            
        loss_terms['loss'] = sum(loss_terms[term] for term in loss_terms)

        return loss_terms
    
    def _calc_length_loss(self, batch_input, pred_lengths):
        pad_mask = batch_input['target_mask']
        input_length = len(batch_input['encoder_pad_mask']) - batch_input['encoder_pad_mask'].sum(0).unsqueeze(-1)
        length_target = len(pad_mask) - pad_mask.sum(0).unsqueeze(-1)
        length_target = length_target - input_length
        if self.length_loss == 'cross_entropy':
            # leverage the fact that the change in length is likely to be small, so large indices can be used for negative length change
            length_loss = -pred_lengths.gather(dim=-1, index=length_target % self.diffuser.max_seq_len)
        elif self.length_loss == 'weighted_sum':
            length_dist = torch.exp(pred_lengths)
            length_indices = torch.arange(0, length_dist.shape[-1], device='cuda').repeat(len(length_target), 1)
            index_errors = (length_indices - length_target) ** 2
            length_loss = (length_dist * index_errors).sum(1)
        elif self.length_loss == 'focal':
            gamma = 0.25
            length_loss = -pred_lengths.gather(dim=-1, index=length_target + 10)
            length_dist = torch.exp(-length_loss)
            focal_mod = (1 - length_dist) ** gamma
            length_loss *= focal_mod
        return length_loss.mean()

    def _calc_token_acc(self, batch_input, token_output):
        token_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def _calc_perplexity(self, batch_input, vocab_dist_output):
        target_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]

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
