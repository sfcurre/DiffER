import argparse, os, yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from source.data import RSmilesUspto50
from source.tokeniser import load_tokeniser_from_rsmiles, load_selfies_tokeniser_from_rsmiles
from source.conditional_model import ConditionalModel
from source.conditional_model_attn_eval import ConditionalModelAttnEval
from source.discrete_diffusion import UnifiedDiscreteDiffusion
from source.trainer import UnifiedTrainer
from source.sampler import UnifiedSampler
from source.utils import move_batch_to_gpu, repeat_batch

import json

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()
if use_gpu:
    print("Using CUDA.")
else:
    print("Using CPU.")

#========================================================================
def main(name, config, load, num_samples, test, pred_lengths):

    print("Building tokeniser...")
    if config['data']['selfies']:
        tokeniser = load_selfies_tokeniser_from_rsmiles(config['data']['data_path'])
    else:
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
    
    print("Reading datasets...")
    dataloaders = {}
    num_available_cpus = len(os.sched_getaffinity(0))
    num_workers = num_available_cpus // config['training']['gpus']
    
    for split in ['train', 'val', 'test']:
        dataset = RSmilesUspto50(tokeniser, config['data']['data_path'], split, forward=forward_pred, pad_limit=config['data']['pad_limit'], max_seq_len=config['model']['max_seq_len'], selfies=config['data']['selfies'])
        dataloaders[split] = DataLoader(dataset,
                                        batch_size=config['training']['batch_size'],
                                        shuffle=True,
                                        num_workers=num_workers,
                                        collate_fn=dataset.collate_fn)
    print("Finished datasets.")

    model_type = ConditionalModel
    if args.record_attns:
        model_type = ConditionalModelAttnEval
    model = model_type(
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
    
    diffuser = UnifiedDiscreteDiffusion(num_steps=config['model']['num_timesteps'] * (not config['model']['continuous']),
                                        num_classes=len(tokeniser),
                                        noise_schedule_type=config['model']['noise_schedule'],
                                        noise_schedule_args=config['model']['noise_schedule_args'],
                                        )

    sampler = UnifiedSampler(diffuser, tokeniser, config['model']['num_timesteps'], config['model']['max_seq_len'], min_time=0.01, pad_limit=config['data']['pad_limit'])
    
    trainer = UnifiedTrainer(model, optimizer, diffuser, sampler, name, length_loss=config['model']['length_loss'], coeff_ce=config['model']['coeff_ce'], coeff_vlb=config['model']['coeff_vlb'], use_gpu=use_gpu)    
    
    # if os.path.exists(f'out/metrics/{name}_metrics_log.txt'):
    #     os.remove(f'out/metrics/{name}_metrics_log.txt')

    # if not os.path.exists(f'out/samples/{name}/'):
    #     os.mkdir(f'out/samples/{name}/')

    print(f'Evaluating {name}...')
    model.eval()
  
    # torch.manual_seed(1998) 
    # with torch.no_grad():
    #     trainer.print_metrics(dataloaders[DATASET], 'Eval', 10)

    torch.manual_seed(1998)
    all_targets = {}
    attns = {}
    for i, batch in enumerate(dataloaders[DATASET]):

        if i == args.batch_limit:
            break
        
        targets = {}
        for target, source in zip(batch['target_smiles'], batch['encoder_smiles']):
            targets[source] = {'target': target, 'samples':[]}
        
        if use_gpu:
            move_batch_to_gpu(batch)

        if num_samples > 1:
            repeat_batch(batch, num_samples)

        sampled_mols, _ = sampler.sample(batch,
                                          model,
                                          verbose=False,
                                          pred_lengths=pred_lengths,
                                          clean=False)
        for j, smi in enumerate(sampled_mols):
            targets[batch['encoder_smiles'][j]]['samples'].append(smi)

        if i < args.record_attns:
            sampled_mols, _, in_attns, out_attns = sampler.sample(batch,
                                                                   model,
                                                                   verbose=False,
                                                                   pred_lengths=pred_lengths,
                                                                   clean=False,
                                                                   record_attns=True)
            for j, smi in enumerate(sampled_mols):
                attns[batch['encoder_smiles'][j]] = smi_data = {}
                smi_data['target'] = batch['decoder_smiles'][j]
                smi_data['sample'] = smi
                smi_data['in_attns'] = {k: v[j] for k, v in in_attns.items()}
                smi_data['x_t'] = {t: k[j] for t, (k, _) in out_attns.items()}
                smi_data['out_attns'] = {t: {k: v[j] for k, v in out_t.items()} for t, (_, out_t) in out_attns.items()}

        print(f'Batch {i} complete.')
        
        for source in targets:
            if source in all_targets:
                all_targets[source]['samples'].extend(targets[source]['samples'])
            else:
                all_targets[source] = targets[source]
        
        with open(f"out/samples/{name}_samples.json", 'w') as fp:
            json.dump(all_targets, fp)

        if i < args.record_attns:
            torch.save(attns, f"out/samples/attns/{name}_attns.json")

    print('Evaluation complete.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--load", type=str, default='')
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--use_true_lengths", action='store_true')
    parser.add_argument("--record_attns", type=int, default=0)
    parser.add_argument("--pad_limit", type=int, default=None)
    parser.add_argument("--batch_limit", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_timesteps", type=int, default=None)
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if args.pad_limit is not None:
        config['data']['pad_limit'] = args.pad_limit

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    if args.num_timesteps is not None:
        config['model']['num_timesteps'] = args.num_timesteps
    
    main(args.name, config, args.load, args.num_samples, args.test, not args.use_true_lengths)
