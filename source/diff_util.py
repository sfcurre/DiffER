import torch
import torch.nn.functional as F
import numpy as np
import math

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_sample_categorical(logits, num_classes):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample = (gumbel_noise + logits).argmax(dim=1)
    log_sample = index_to_log_onehot(sample, num_classes)
    #log_sample = gumbel_noise + logits
    return log_sample

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    
    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)    

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionCollater:
    def __init__(self, tokeniser, num_timesteps, forward_pred, max_seq_len, beta_schedule='cosine', pad_limit=20):
        self.tokeniser = tokeniser
        self.num_timesteps = num_timesteps
        self.forward_pred = forward_pred
        self.max_seq_len = max_seq_len
        self.pad_limit = pad_limit
        
        # alphas = cosine_beta_schedule(num_timesteps)
        alphas = 1 - get_named_beta_schedule(beta_schedule, num_diffusion_timesteps=num_timesteps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.log_alpha = log_alpha.float()
        self.log_1_min_alpha = log_1_min_alpha.float()
        self.log_cumprod_alpha = log_cumprod_alpha.float()
        self.log_1_min_cumprod_alpha = log_1_min_cumprod_alpha.float()

        self.Lt_history = torch.zeros(num_timesteps)
        self.Lt_count = torch.zeros(num_timesteps)

    def __call__(self, batch):
        return self._collate(batch)

    def _collate(self, batch):
        reacts_smiles, prods_smiles = tuple(zip(*batch))

        if self.forward_pred:
            encoder_smiles = reacts_smiles
            decoder_smiles = prods_smiles
        else:
            encoder_smiles = prods_smiles
            decoder_smiles = reacts_smiles

        # add some number of length-padding tokens
        
        if self.pad_limit is None:
            decoder_smiles_padded = tuple(smi + '?' * self.max_seq_len for smi in decoder_smiles)
        elif self.pad_limit > 0: 
            decoder_smiles_padded = tuple(smi + '?' * np.random.randint(1, self.pad_limit) for smi in decoder_smiles)
        elif self.pad_limit == 0:
            decoder_smiles_padded = decoder_smiles
        else:
            decoder_smiles_padded = tuple('?' * np.random.randint(1, -self.pad_limit) + smi + '?' * np.random.randint(1, -self.pad_limit) for smi in decoder_smiles)

        encoder_input = self.tokeniser.tokenise(encoder_smiles, mask=False, pad=True)
        decoder_input = self.tokeniser.tokenise(decoder_smiles_padded, mask=False, pad=True)
        
        encoder_token_ids, encoder_pad_mask = self._partial_collate(encoder_input)
        m_encoder_token_ids, m_encoder_pad_mask, m_encoder_t = self._partial_collate(encoder_input, noised=True)
        decoder_token_ids, decoder_pad_mask = self._partial_collate(decoder_input)
        m_decoder_token_ids, m_decoder_pad_mask, m_decoder_t = self._partial_collate(decoder_input, noised=True)

        collate_output = {
            "encoder_input": encoder_token_ids,
            "encoder_pad_mask": encoder_pad_mask,
            "decoder_input": m_decoder_token_ids,
            "decoder_pad_mask": m_decoder_pad_mask,
            "target": decoder_token_ids.max(dim=-1)[1],
            "target_onehots": decoder_token_ids,
            "target_mask": decoder_pad_mask,
            "target_smiles": decoder_smiles,
            # "masked_encoder_input": m_encoder_token_ids,
            # "masked_encoder_pad_mask": m_encoder_pad_mask,
            "encoder_smiles": encoder_smiles,
            "decoder_t": m_decoder_t,
            "device": "cpu"
        }
        
        return collate_output

    def _partial_collate(self, inputs, noised=False):
        input_tokens = inputs["original_tokens"]
        input_mask = inputs["original_pad_masks"]
        
        input_token_ids = self.tokeniser.convert_tokens_to_ids(input_tokens)
        
        input_token_ids = torch.tensor(input_token_ids)
        input_token_ids = index_to_log_onehot(input_token_ids, len(self.tokeniser))
        input_pad_mask = torch.tensor(input_mask, dtype=torch.bool).transpose(0, 1)

        input_token_ids = input_token_ids[..., :self.max_seq_len]
        input_pad_mask = input_pad_mask[:self.max_seq_len]

        if noised:
            #importance sample t
            t, _ = self.sample_time(size=len(input_tokens), method='importance')
            input_token_ids = self.q_sample(input_token_ids, t)
            return torch.permute(input_token_ids, (2, 0, 1)), input_pad_mask, t
        
        return torch.permute(input_token_ids, (2, 0, 1)), input_pad_mask

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = log_sample_categorical(log_EV_qxt_x0, len(self.tokeniser))
        return log_sample

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(len(self.tokeniser))
        )

        return log_probs

    def sample_time(self, size, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(size, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=size, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (size,)).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError
        
    def update_Lt(self, t, kl):
        Lt2 = kl.pow(2).cpu()
        t = t.cpu()
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))
