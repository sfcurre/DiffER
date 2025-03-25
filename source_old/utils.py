import numpy
import torch

from rdkit import Chem

def move_batch_to_gpu(batch):
    for key, value in batch.items():
        if hasattr(value, 'cuda'):
            batch[key] = value.cuda()
    batch['device'] = 'cuda'

def canonicalize(smi):
    smi = smi.replace('?', '')
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    else:
        return Chem.MolToSmiles(m)
