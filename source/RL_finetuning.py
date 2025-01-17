import time
import numpy as np
import glob, os, json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger

from .trainer import DiffusionModelTrainer

class RLFineTuner(DiffusionModelTrainer):
    def __init__(self, model, optimizer, name='Default', loss_components=['nll'], use_gpu=True):
        super(DiffusionModelTrainer, self).__init__(model, optimizer, name=name, loss_components=loss_components, use_gpu=use_gpu)

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

    def _calc_loss(self, batch_input, token_output, update_Lt=True):
        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        x_start = batch_input["target_onehots"]
        t = batch_input['decoder_t']

        loss_terms = {}    
        
        if 'nll' in self.loss or 'vb' in self.loss:
            lprobs = F.log_softmax(token_output, dim=-1)
            non_pad_mask = tokens.ne(self.model.pad_token_idx)
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
            log_true_prob = self.model.q_posterior(torch.log_softmax(x_start, dim=-1).permute((1, 2, 0)), log_x_t, t)
            log_model_prob = self.model.q_posterior(torch.log_softmax(token_output, dim=-1).permute((1, 2, 0)), log_x_t, t)
            kl = -(log_true_prob.exp() * (log_true_prob - log_model_prob))
            kl = kl.sum(dim=(1, 2))
            if update_Lt:
                self.model.collate_fn.update_Lt(t, kl)
            loss_terms['kl'] = kl.mean()

        if 'vb' in self.loss:
            mask = (t == torch.zeros_like(t)).float()
            vb_loss = mask * nll_loss + (1. - mask) * kl
            loss_terms['vb'] = vb_loss.mean()

        if 'parentheses' in self.loss:
            open_idx = self.model.tokeniser.vocab['(']
            close_idx = self.model.tokeniser.vocab[')']
            probs = F.softmax(token_output, dim=-1)
            err = (probs[..., open_idx] - probs[..., close_idx]) ** 2
            parentheses_loss = err.sum(dim=0)
            loss_terms['parentheses'] = parentheses_loss.mean()
            
        loss_terms['loss'] = sum(loss_terms[term] for term in loss_terms)

        return loss_terms
