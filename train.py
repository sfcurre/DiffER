import argparse, os, yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from source.data import RSmilesUspto50
from source.tokeniser import load_tokeniser_from_rsmiles
from source.discrete_diffuser import DiscreteDiffuser
from source.conditional_model import ConditionalModel
from source.diffuseq_model import DiffuseqModel
from source.trainer import DiffusionModelTrainer

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()
if use_gpu:
    print("Using CUDA.")
else:
    print("Using CPU.")
    
#========================================================================
def main(name, config, load):

    print("Building tokeniser...")
    tokeniser = load_tokeniser_from_rsmiles(config['data']['data_path'])
    print(f"Finished tokeniser with {len(tokeniser)} tokens.")
    
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

    model_class = ConditionalModel
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

    if load:
        model.load_state_dict(torch.load(load))

    if use_gpu:
        model = model.cuda()
 
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    
    trainer = DiffusionModelTrainer(model, optimizer, diffuser, name, loss_components=config['model']['loss_terms'],
                                    length_loss=config['model']['length_loss'], use_gpu=use_gpu)

    if os.path.exists(f'out/metrics/{name}_metrics_log.txt'):
        os.remove(f'out/metrics/{name}_metrics_log.txt')

    print(f'Training {name} with heuristics...')
    trainer.train(dataloaders,
                  config['training']['epochs'],
                  config['training']['patience'],
                  val_limit=10)
    
#========================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--load", type=str, default='')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    main(args.name, config, args.load)
