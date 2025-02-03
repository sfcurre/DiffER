import numpy as np
import math

import torch

'''
This code is copied and adapted from https://github.com/andrew-cr/tauLDR/blob/main/lib/models/models.py
'''

class BirthDeathForwardBase():
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.sigma_min, self.sigma_max = cfg.model.sigma_min, cfg.model.sigma_max
        self.device = device

        base_rate = np.diag(np.ones((S-1,)), 1)
        base_rate += np.diag(np.ones((S-1,)), -1)
        base_rate -= np.diag(np.sum(base_rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(base_rate)

        self.base_rate = torch.from_numpy(base_rate).float().to(self.device)
        self.base_eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.base_eigvecs = torch.from_numpy(eigvecs).float().to(self.device)

    def _rate_scalar(self, t, #["B"]
    ):
        return self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t) *\
            math.log(self.sigma_max / self.sigma_min)

    def _integral_rate_scalar(self, t, # ["B"]
    ):
        return 0.5 * self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t) -\
            0.5 * self.sigma_min**2

    def rate(self, t, #["B"]
    ):
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.base_eigvals.view(1, S)

        transitions = self.base_eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(adj_eigvals)) @ \
            self.base_eigvecs.T.view(1, S, S)

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(f"[Warning] BirthDeathForwardBase, large negative transition values {torch.min(transitions)}")

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions

class UniformRate():
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_const = cfg.model.rate_const
        self.device = device

        rate = self.rate_const * np.ones((S,S))
        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(rate)

        self.rate_matrix = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)

    def rate(self, t, #["B"]
    ):
        B = t.shape[0]
        S = self.S

        return torch.tile(self.rate_matrix.view(1,S,S), (B, 1, 1))

    def transition(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S
        transitions = self.eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(self.eigvals.view(1, S) * t.view(B,1))) @\
            self.eigvecs.T.view(1, S, S)

        if torch.min(transitions) < -1e-6:
            print(f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}")

        transitions[transitions < 1e-8] = 0.0

        return transitions

class GaussianTargetRate():
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_sigma = cfg.model.rate_sigma
        self.Q_sigma = cfg.model.Q_sigma
        self.time_exponential = cfg.model.time_exponential
        self.time_base = cfg.model.time_base
        self.device = device

        rate = np.zeros((S,S))

        vals = np.exp(-np.arange(0, S)**2/(self.rate_sigma**2))
        for i in range(S):
            for j in range(S):
                if i < S//2:
                    if j > i and j < S-i:
                        rate[i, j] = vals[j-i-1]
                elif i > S//2:
                    if j < i and j > -i+S-1:
                        rate[i, j] = vals[i-j-1]
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * np.exp(- ( (j+1)**2 - (i+1)**2 + S*(i+1) - S*(j+1) ) / (2 * self.Q_sigma**2)  )

        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))

        eigvals, eigvecs = np.linalg.eig(rate)
        inv_eigvecs = np.linalg.inv(eigvecs)

        self.base_rate = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float().to(self.device)

    def _integral_rate_scalar(self, t, # ["B"]
    ):
        return self.time_base * (self.time_exponential ** t) - \
            self.time_base
    
    def _rate_scalar(self, t, # ["B"]
    ):
        return self.time_base * math.log(self.time_exponential) * \
            (self.time_exponential ** t)

    def rate(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t, # ["B"]
    ):
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)

        transitions = self.eigvecs.view(1, S, S) @ \
            torch.diag_embed(torch.exp(adj_eigvals)) @ \
            self.inv_eigvecs.view(1, S, S)

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(f"[Warning] GaussianTargetRate, large negative transition values {torch.min(transitions)}")

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions

