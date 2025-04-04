import numpy
import torch
import math

from rdkit import Chem

def move_batch_to_gpu(batch):
    for key, value in batch.items():
        if hasattr(value, 'cuda'):
            batch[key] = value.cuda()
    batch['device'] = 'cuda'

def repeat_batch(batch, num_samples):
    batch['y_0'] = batch['y_0'].repeat(num_samples, 1, 1)
    batch['x_0'] = batch['x_0'].repeat(num_samples, 1, 1)
    batch['y_mask'] = batch['y_mask'].repeat(num_samples, 1)
    batch['x_mask'] = batch['x_mask'].repeat(num_samples, 1)
    batch['encoder_smiles'] *= num_samples
    batch['decoder_smiles'] *= num_samples
    batch['target_smiles'] *= num_samples

def canonicalize(smi):
    smi = smi.replace('?', '')
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    else:
        return Chem.MolToSmiles(m)

def noise_schedule(t_step, 
                   s_step=None,
                   schedule_type:str= "cosine",
                   N:int = 1000, # N=0 means continuous 
                   Tmax:float =1,
                   a:float=None, b:float=None, 
                   min_alphabar:float=1e-10, max_beta:float=250, 
                   sigma_min:float = 1e-4, sigma_max:float=1,
                   eps:float =1e-3,
                   **kwargs):
    assert t_step.max() <= Tmax if N == 0 else t_step.max() <= N
    step_to_time = lambda step: step if N == 0 else step/N * Tmax
    t = step_to_time(t_step)
    s = torch.tensor(0.0) if s_step is None else step_to_time(s_step)

    if schedule_type == "cosine":      
        a = a or 0.008            # set default value
        h = lambda t: torch.cos((t/Tmax + a)/ (1+a) * torch.pi * 0.5)
        h_t, h_s = h(t), h(s) 
        alphabar_t = h_t / h_s
        beta_t = torch.pi * torch.tan((t/Tmax + a) / (1 + a) * torch.pi * 0.5)
        beta_t = beta_t / (2*Tmax*(1+a))
    elif schedule_type == "exponential":
        a, b = a or 0.5, b or 10  # set default value
        b_power = lambda t: torch.exp(t/Tmax * math.log(b))
        b_power_t, b_power_s = b_power(t), b_power(s)
        alphabar_t = torch.exp(a * t * (b_power_s - b_power_t))
        beta_t = a * b_power_t * math.log(b)  
    elif schedule_type == "linear":
        alphabar_t = 1 - t/Tmax
        beta_t = 1/(Tmax - t)
    elif schedule_type == "constant":
        a = a or 0.03
        h = lambda t: torch.exp(-a * t)
        h_t, h_s = h(t), h(s) 
        alphabar_t = h_t / h_s
        beta_t = torch.full_like(t, a) 
    elif schedule_type == 'geometric':
        sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        beta_t = sigmas[0] ** (1 - t) * sigmas[1] ** t * (sigmas[1].log() - sigmas[0].log())
        betabar_t_s = sigmas[0] ** (1 - t) * sigmas[1] ** t - sigmas[0] ** (1 - s) * sigmas[1] ** s 
        alphabar_t = torch.exp(-1.0 * betabar_t_s)
    elif schedule_type == 'loglinear':
        beta_t= (1 - eps) / (1 - (1 - eps) * t)
        betabar_t_s = -torch.log1p(-(1 - eps) * t) + torch.log1p(-(1 - eps) * s)
        alphabar_t = torch.exp(-1.0 * betabar_t_s)
    else:
        raise NotImplementedError

    assert alphabar_t.dim() == 1 
    alphabar_t = torch.clip(alphabar_t, min=min_alphabar, max=1-min_alphabar)
    beta_t = torch.clip(beta_t, max=max_beta)  # TODO: revise later

    return alphabar_t, beta_t
