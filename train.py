import argparse, os, yaml

import torch
from functools import partial
import torch.optim as optim
from torch.utils.data import DataLoader

from source.data import RSmilesUspto50
from source.tokeniser import load_tokeniser_from_rsmiles
from source.conditional_model import ConditionalModel
from source.conditional_moe_model import ConditionalMoEModel
from source.discrete_diffusion import UnifiedDiscreteDiffusion
from source.trainer import UnifiedTrainer
from source.sampler import UnifiedSampler

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
    
    for split in ['train', 'val', 'test']:
        dataset = RSmilesUspto50(tokeniser, config['data']['data_path'], split, forward=forward_pred, pad_limit=config['data']['pad_limit'], max_seq_len=config['model']['max_seq_len'])
        dataloaders[split] = DataLoader(dataset,
                                        batch_size=config['training']['batch_size'],
                                        shuffle=True,
                                        num_workers=num_workers,
                                        collate_fn=dataset.collate_fn)
    print("Finished datasets.")

    model_class = ConditionalModel
    moe_weight = 0
    if 'moe' in config['model'] and config['model']['moe']:
        model_class = partial(ConditionalMoEModel, num_experts=config['model']['num_experts'])
        moe_weight = config['model']['moe_loss_weight']

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
    
    diffuser = UnifiedDiscreteDiffusion(num_steps=config['model']['num_timesteps'] * (not config['model']['continuous']),
                                        num_classes=len(tokeniser),
                                        noise_schedule_type=config['model']['noise_schedule'],
                                        noise_schedule_args=config['model']['noise_schedule_args'],
                                        )
    sampler = UnifiedSampler(diffuser, tokeniser, config['model']['num_timesteps'], config['model']['max_seq_len'], min_time=0.01, pad_limit=config['data']['pad_limit'])

    trainer = UnifiedTrainer(model, optimizer, diffuser, sampler, name, length_loss=config['model']['length_loss'], coeff_ce=config['model']['coeff_ce'], coeff_vlb=config['model']['coeff_vlb'], use_gpu=use_gpu, moe_loss=moe_weight)

    if os.path.exists(f'out/metrics/{name}_metrics_log.txt'):
        os.remove(f'out/metrics/{name}_metrics_log.txt')

    print(f'Training {name} with heuristics...')
    trainer.train(dataloaders,
                  config['training']['epochs'],
                  val_limit=10)
    
#========================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--load", type=str, default='')
    parser.add_argument("--pad_limit", type=int, default=None)
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if config['data']['pad_limit'] is None and args.pad_limit is not None:
        config['data']['pad_limit'] = args.pad_limit
    elif config['data']['pad_limit'] is None:
        raise ValueError('Pad limit not specified in config or command line arguments.')

    main(args.name, config, args.load)
