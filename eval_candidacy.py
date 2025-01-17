import argparse, sys, os, copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from source.data import RSmilesUspto50
from source.diff_util import DiffusionCollater
from source.tokeniser import load_tokeniser_from_rsmiles
from source.diff_model import DiffusionModel
from source.trainer import DiffusionModelTrainer

import json

# torch.autograd.set_detect_anomaly(True)

# Example Call
# python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --name ForwardDiffusion

# Default model hyperparams
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_ACTIVATION = "gelu"
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_DROPOUT = 0.1

DEFAULT_CHEM_TOKEN_START = 272

DEFAULT_GPUS = 1
USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()
if use_gpu:
    print("Using CUDA.")

DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 50
DEFAULT_AUG_PROB = 0.0

def parse_args():
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--name", type=str)

    parser.add_argument("--load", type=str, default='')

    # Model and training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--aug_prob", type=float, default=DEFAULT_AUG_PROB)
    parser.add_argument("--gpus", type=int, default=DEFAULT_GPUS)
    
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--diffuseq", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--num_samples", type=int, default=20)

    parser.add_argument("--beta_schedule", type=str, default='cosine')
    parser.add_argument("--loss_terms", type=str, default='nll')
    parser.add_argument("--true_lengths", action='store_true')
    parser.add_argument("--pad_limit", type=int, default=None)
    parser.add_argument("--length_diff", type=int, nargs='+', default=None)

    # For debugging
    parser.add_argument("--batch_limit", type=int, default=-1)

    # Rand init model args
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--d_feedforward", type=int, default=DEFAULT_D_FEEDFORWARD)

    return parser.parse_args()

#========================================================================
def main():
    args = parse_args()

    print("Building tokeniser...")
    tokeniser = load_tokeniser_from_rsmiles(args.data_path)
    print("Finished tokeniser.")
    
    if args.task == "forward_prediction":
        forward_pred = True
    elif args.task == "backward_prediction":
        forward_pred = False
    else:
        raise ValueError(f"Unknown task {args.task}")

    print("Reading datasets...")
    dataloaders = {}
    num_available_cpus = len(os.sched_getaffinity(0))
    num_workers = num_available_cpus // args.gpus
    
    collate_fn = DiffusionCollater(tokeniser, num_timesteps=args.num_timesteps, forward_pred=forward_pred, beta_schedule=args.beta_schedule,
                                   max_seq_len=DEFAULT_MAX_SEQ_LEN, pad_limit=args.pad_limit)
    for split in ['train', 'val', 'test']:
        dataset = RSmilesUspto50(args.data_path, split, args.aug_prob, forward=forward_pred)
        dataloaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    print("Finished datasets.")

    model = DiffusionModel(
        tokeniser=tokeniser,
        collate_fn=collate_fn,    
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        num_timesteps=args.num_timesteps,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_feedforward=args.d_feedforward,
        activation=DEFAULT_ACTIVATION,
        dropout=DEFAULT_DROPOUT,
    )
   
    if args.load:
        model.load_state_dict(torch.load(args.load))

    if use_gpu:
        model = model.cuda()
 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = DiffusionModelTrainer(model, optimizer, args.name, loss_components=args.loss_terms.split(','), use_gpu=use_gpu)

    if os.path.exists(f'out/metrics/{args.name}_metrics_log.txt'):
        os.remove(f'out/metrics/{args.name}_metrics_log.txt')

    if not os.path.exists(f'out/samples/{args.name}/'):
        os.mkdir(f'out/samples/{args.name}/')

    print(f'Evaluating {args.name}...')

    torch.manual_seed(1998) 
    all_targets = {}
    for i, batch in enumerate(dataloaders['val']):
        if args.batch_limit >= 0 and i >= args.batch_limit:
            break

        targets = {}
        for target, source in zip(batch['target_smiles'], batch['encoder_smiles']):
            targets[source] = {'target': target, 'samples':[], 'pred_lengths':[], 'true_lengths':[]}

        range_ = range(args.num_samples)
        if args.length_diff is not None:
            range_ = []
            for diff in range(*args.length_diff):
                for s in range(args.num_samples):
                    range_.append(diff)

        trainer.move_batch_to_gpu(batch)
        for i in range_:
            length_diff = None if args.length_diff is None else i
            sampled_mols, memories = model.sample(batch, verbose=False, use_gpu=True, return_chain=False,
                                                pred_lengths=not args.true_lengths, return_lengths=True,
                                                clean=False, length_diff=length_diff, return_memory=True)
            
            for j, smi in enumerate(sampled_mols):
                targets[batch['encoder_smiles'][j]]['samples'].append(smi)
                #targets[batch['encoder_smiles'][j]]['pred_lengths'].append(pred_lengths[j].tolist())     
                #targets[batch['encoder_smiles'][j]]['true_lengths'].append(true_lengths[j].tolist())
                targets[batch['encoder_smiles'][j]]['memory'] = memories[j].tolist()     

        if args.verbose:   
            for target, source in zip(batch['target_smiles'], batch['encoder_smiles']):
                print(f'Target: {target}')
                print(f'Samples:')
                for smi in targets[source]['samples']:
                    print(f'\t{smi}')

        print(f'Batch {i} complete.')
        
        for source in targets:
            if source in all_targets:
                all_targets[source]['samples'].extend(targets[source]['samples'])
            else:
                all_targets[source] = targets[source]
        
        #all_targets.update(targets)

    with open(f"out/samples/{args.name}/{args.name}_repeated_samples.json", 'w') as fp:
        json.dump(all_targets, fp)

    print('Evaluation complete.')

if __name__ == '__main__':
    main()
