import time
import numpy as np
import glob, os, json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger

class DiffusionModelTrainer:
    def __init__(self, model, optimizer, name='Default', loss_components=['nll'], use_gpu=True):
        self.model = model
        self.optimizer = optimizer
        self.name = name
        self.loss = loss_components
        self.use_gpu = use_gpu
        
        RDLogger.DisableLog("rdApp.*")

    def train(self, dataloaders, epochs, patience, report_interval=None, batch_limit=None, val_limit=100):
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        for epoch in range(epochs):
            print(f'Epoch {epoch} - {time.time() - t_total}')
            for i, batch in enumerate(dataloaders['train']):
                loss_values.append(self.train_step(batch))
                
                if batch_limit is not None and i == batch_limit:
                    print("Batch limit reached.")
                    break
                
                if report_interval is not None and (i+1) % report_interval == 0:
                    print('Recording metrics...')
                    with torch.no_grad():
                        self.print_metrics(dataloaders['val'], str(epoch) + f'.{i+1} - {time.time() - t_total}', val_limit)
                    torch.save(self.model.state_dict(), 'out/models/{}_{}.pkl'.format(self.name, epoch))
                    np.save(f'out/losses/{self.name}_losses.npy', np.array(loss_values))
                    
            with torch.no_grad():
                self.print_metrics(dataloaders['val'], str(epoch) + f'.{i+1} - {time.time() - t_total}', val_limit)
            torch.save(self.model.state_dict(), 'out/models/{}_{}.pkl'.format(self.name, epoch))
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
                    os.remove(f'out/models/{self.name}_{hm.name}-HM_{epoch_nb}.pkl')

        files = glob.glob(f'out/models/{self.name}_.pkl')
        for file in files:
            epoch_nb = int(file.split('_')[-1].split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)
                os.remove(f'out/models/{self.name}_{hm.name}-HM_{epoch_nb}.pkl')

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        self.model.load_state_dict(torch.load('out/models/{}_{}.pkl'.format(self.name, best_epoch)))

        # Testing
        return loss_values

    def print_metrics(self, val_loader, epoch, val_limit):
        self.model.eval()
        metrics = defaultdict(list)
        mols = []
        for i, batch in enumerate(val_loader):
            if i == val_limit:
                break
            batch_metrics, sampled_mols = self.val_step(batch)
            
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
 
        with open(f'out/samples/{self.name}/sampled_mols_e{epoch.split()[0]}.json', 'w') as fp:
            json.dump(mols, fp)

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
        
        output = self.model.forward(batch)
        loss = self._calc_loss(batch, output)['loss']
        loss.backward()

        self.optimizer.step()
        return loss.cpu().item()

    def val_step(self, batch):
        if self.use_gpu:
            self.move_batch_to_gpu(batch)

        self.model.eval()
        output = self.model.forward(batch)
        loss = self._calc_loss(batch, output)['loss']
        token_acc = self._calc_token_acc(batch, output)
        perplexity = self._calc_perplexity(batch, output)

        sampled_smiles, lprobs = self.model.sample(batch, verbose=True, use_gpu=self.use_gpu)
        sampling_metrics = self._calc_sampling_metrics(batch, sampled_smiles)

        metrics = dict(val_loss=loss.cpu(),
                       token_accuracy=token_acc,
                       perplexity=perplexity,
                       mol_accuracy=sampling_metrics['accuracy'],
                       mol_invalid=sampling_metrics['invalid'])

        return metrics, sampled_smiles

    def _calc_loss(self, batch_input, token_output):
        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        x_start = batch_input["target_onehots"]
        t = batch_input['decoder_t']

        lprobs = F.log_softmax(token_output, dim=-1)

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = tokens.reshape(-1, 1)
        non_pad_mask = target.ne(pad_mask.reshape(-1, 1))

        loss_terms = {}    
        
        if 'nll' in self.loss or 'vb' in self.loss:
            nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            nll_loss = nll_loss.reshape((token_output.size(0), token_output.size(1))).transpose(0, 1)
            nll_loss = nll_loss.sum(dim=-1)
            loss_terms['nll'] = nll_loss.sum()

        if 'mse' in self.loss:
            probs = F.softmax(token_output, dim=-1)
            probs = probs.view(-1, probs.size(-1))
            mse_loss = (x_start.reshape(-1, probs.size(-1)) - probs) ** 2
            loss_terms['mse'] = mse_loss.sum()

        if 'kl' in self.loss or 'vb' in self.loss:
            log_x_t = batch_input['decoder_input'].permute((1, 2, 0))
            log_true_prob = self.model.q_posterior(torch.log_softmax(x_start, dim=-1).permute((1, 2, 0)), log_x_t, t)
            log_model_prob = self.model.q_posterior(torch.log_softmax(token_output, dim=-1).permute((1, 2, 0)), log_x_t, t)
            kl = (log_true_prob.exp() * (log_true_prob - log_model_prob)).sum(dim=(1, 2))
            loss_terms['kl'] = kl.sum()

        if 'vb' in self.loss:
            mask = (t == torch.zeros_like(t)).float()

            vb_loss = mask * nll_loss + (1. - mask) * kl
            loss_terms['vb'] = vb_loss.sum()

        loss_terms['loss'] = sum(loss_terms[term] for term in loss_terms)

        return loss_terms

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
