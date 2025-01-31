import argparse, os, yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from source.data import RSmilesUspto50
from source.discrete_diffuser import DiscreteDiffuser
from source.tokeniser import load_tokeniser_from_rsmiles
from source.diff_model import DiffusionModel
from source.diffuseq_model import DiffuseqModel
from source.trainer import DiffusionModelTrainer

import json

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()
if use_gpu:
    print("Using CUDA.")

#========================================================================
def main(name, config, load, num_samples, test, pred_lengths):

    print("Building tokeniser...")
    tokeniser = load_tokeniser_from_rsmiles(config['data']['data_path'])
    print("Finished tokeniser.")

    DATASET = 'test' if test else 'val'
    
    if config['data']['task'] == "forward_prediction":
        forward_pred = True
    elif config['data']['task'] == "backward_prediction":
        forward_pred = False
    else:
        raise ValueError(f"Unknown task {config['data']['task']}")

    print("Reading datasets...")
    dataloaders = {}
    num_available_cpus = len(os.sched_getaffinity(0))
    num_workers = num_available_cpus // config['training']['gpus']
    
    diffuser = DiscreteDiffuser(tokeniser,
                                forward_pred=forward_pred,
                                num_timesteps=config['model']['num_timesteps'],
                                max_seq_len=config['model']['max_seq_len'],
                                beta_schedule=config['model']['beta_schedule'],
                                pad_limit=config['model']['pad_limit'])
    for split in ['train', 'val', 'test']:
        dataset = RSmilesUspto50(config['data']['data_path'], split, forward=forward_pred)
        dataloaders[split] = DataLoader(dataset,
                                        batch_size=config['training']['batch_size'],
                                        shuffle=True,
                                        num_workers=num_workers,
                                        collate_fn=diffuser)
    print("Finished datasets.")

    model_class = DiffusionModel
    if config['model']['diffuseq']:
        model_class = DiffuseqModel
    
    model = model_class(
        tokeniser=tokeniser,
        max_seq_len=config['model']['max_seq_len'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_feedforward=config['model']['d_feedforward'],
        activation=config['model']['activation'],
        dropout=config['model']['dropout'])
   
    state_dict = torch.load(load)
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()
 
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    
    trainer = DiffusionModelTrainer(model, optimizer, diffuser, name, loss_components=config['model']['loss_terms'],
                                    length_loss=config['model']['length_loss'], use_gpu=use_gpu)
    
    if os.path.exists(f'out/metrics/{name}_metrics_log.txt'):
        os.remove(f'out/metrics/{name}_metrics_log.txt')

    if not os.path.exists(f'out/samples/{name}/'):
        os.mkdir(f'out/samples/{name}/')

    print(f'Evaluating {name}...')
  
    torch.manual_seed(1998) 
    with torch.no_grad():
        trainer.print_metrics(dataloaders[DATASET], 'Eval', 10)

    torch.manual_seed(1998) 
    all_targets = {}
    for i, batch in enumerate(dataloaders[DATASET]):

        targets = {}
        for target, source in zip(batch['target_smiles'], batch['encoder_smiles']):
            targets[source] = {'target': target, 'samples':[]}
        
        trainer.move_batch_to_gpu(batch)
        for _ in range(num_samples):
            sampled_mols, lprobs = diffuser.sample(batch,
                                                   model,
                                                   verbose=False,
                                                   pred_lengths=pred_lengths,
                                                   clean=False)
            for j, smi in enumerate(sampled_mols):
                targets[batch['encoder_smiles'][j]]['samples'].append(smi)
                
        print(f'Batch {i} complete.')
        
        for source in targets:
            if source in all_targets:
                all_targets[source]['samples'].extend(targets[source]['samples'])
            else:
                all_targets[source] = targets[source]
        
    with open(f"out/samples/{name}/{name}_samples.json", 'w') as fp:
        json.dump(all_targets, fp)

    print('Evaluation complete.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--load", type=str, default='')
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--use_true_lengths", action='store_true')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    main(args.name, config, args.load, args.num_samples, args.test, not args.use_true_lengths)
