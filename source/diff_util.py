import torch
import torch.nn.functional as F
import numpy as np

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

class DiffusionCollater:
    def __init__(self, tokeniser, num_timesteps, forward_pred):
        self.tokeniser = tokeniser
        self.num_timesteps = num_timesteps
        self.forward_pred = forward_pred
        
        alphas = cosine_beta_schedule(num_timesteps)

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

        # # add some number of length-padding tokens
        decoder_smiles_padded = tuple(smi + '?' * np.random.randint(1, 10) for smi in decoder_smiles)
        
        encoder_input = self.tokeniser.tokenise(encoder_smiles, mask=False, pad=True)
        decoder_input = self.tokeniser.tokenise(decoder_smiles_padded, mask=False, pad=True)
        
        encoder_token_ids, encoder_pad_mask = self._partial_collate(encoder_input)
        m_encoder_token_ids, m_encoder_pad_mask = self._partial_collate(encoder_input, noised=True)
        decoder_token_ids, decoder_pad_mask = self._partial_collate(decoder_input)
        m_decoder_token_ids, m_decoder_pad_mask = self._partial_collate(decoder_input, noised=True)

        collate_output = {
            "encoder_input": encoder_token_ids,
            "encoder_pad_mask": encoder_pad_mask,
            "decoder_input": m_decoder_token_ids,
            "decoder_pad_mask": m_decoder_pad_mask,
            "target": decoder_token_ids.max(dim=-1)[1],
            "target_mask": decoder_pad_mask,
            "target_smiles": decoder_smiles,
            "masked_encoder_input": m_encoder_token_ids,
            "masked_encoder_pad_mask": m_encoder_pad_mask,
            "encoder_smiles": encoder_smiles
        }
        
        return collate_output

    def _partial_collate(self, inputs, noised=False):
        input_tokens = inputs["original_tokens"]
        input_mask = inputs["original_pad_masks"]
        
        input_token_ids = self.tokeniser.convert_tokens_to_ids(input_tokens)
        
        input_token_ids = torch.tensor(input_token_ids)
        input_token_ids = index_to_log_onehot(input_token_ids, len(self.tokeniser))
        input_pad_mask = torch.tensor(input_mask, dtype=torch.bool).transpose(0, 1)

        if noised:
            t = np.random.randint(0, self.num_timesteps - 1, size=len(input_tokens), dtype=np.int64)
            t = torch.tensor(t)
            input_token_ids = self.q_sample(input_token_ids, t)
        
        return torch.permute((2, 0, 1)), input_pad_mask

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
