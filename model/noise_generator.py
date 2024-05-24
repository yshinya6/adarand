import pdb

import torch
from torch import nn
import torch.distributions as dists
import numpy as np


class NoiseGenerator(nn.Module):
    def select_dist_func(self, dist_name):
        if dist_name == "normal":
            return dists.Normal
        elif dist_name == "laplace":
            return dists.Laplace
        elif dist_name == "uniform":
            return dists.Uniform
        else:
            raise NotImplementedError

    def __init__(self,
                 feat_dim=512,
                 ema_decay=0.5,
                 dist='normal',
                 mu=0.0,
                 sigma=1.0,
                 precomputed_stats=None,
                 update_mu=True,
                 update_sigma=True) -> None:
        super().__init__()
        self.ema_decay = ema_decay
        self.dist_name = dist
        self.dist_fn = self.select_dist_func(dist)
        self.update_mu = update_mu
        self.update_sigma = update_sigma
        if precomputed_stats:
            stats = np.load(precomputed_stats)
            self.mu = torch.nn.Parameter(torch.from_numpy(stats["mean"]))
            self.sigma = torch.nn.Parameter(torch.from_numpy(stats["std"]))
        else:
            self.mu = torch.nn.Parameter(torch.tensor([mu] * feat_dim).float())
            self.sigma = torch.nn.Parameter(torch.tensor([sigma] * feat_dim).float())

    def update_params(self, batch_mu, batch_sigma) -> None:
        if self.update_mu:
            self.mu.data = self.ema_decay * self.mu.data + (1 - self.ema_decay) * batch_mu
        if self.update_sigma:
            self.sigma.data = self.ema_decay * self.sigma.data + (1 - self.ema_decay) * batch_sigma

    def forward(self, y) -> torch.Tensor:
        return self.dist_fn(self.mu, self.sigma).sample((len(y),))


class MultiClassNoiseGenerator(NoiseGenerator):
    def __init__(self,
                 num_classes,
                 feat_dim=512,
                 ema_decay=0.5,
                 dist='normal',
                 mu=0.0,
                 sigma=1.0,
                 precomputed_stats=None,
                 update_mu=True,
                 update_sigma=True
                 ) -> None:
        super().__init__(feat_dim, ema_decay, dist, mu, sigma, precomputed_stats, update_mu, update_sigma)
        self.num_classes = num_classes
        if precomputed_stats:
            stats = np.load(precomputed_stats)
            if dist in ["normal", "laplace"]:
                self.mu = torch.nn.Parameter(torch.from_numpy(stats["mean"]))
                self.sigma = torch.nn.Parameter(torch.from_numpy(stats["std"]))
            elif dist == "uniform":
                self.mu = torch.nn.Parameter(torch.from_numpy(stats["min"]))
                self.sigma = torch.nn.Parameter(torch.from_numpy(stats["max"]))
        else:
            self.mu = torch.nn.Parameter(torch.tensor([mu] * feat_dim * num_classes).view(-1, feat_dim))
            self.sigma = torch.nn.Parameter(torch.tensor([sigma] * feat_dim * num_classes).view(-1, feat_dim))

    def forward(self, y) -> torch.Tensor:
        batch_mu, batch_sigma = self.mu[y], self.sigma[y]
        return self.dist_fn(batch_mu, batch_sigma).sample()


class MultiClassFixedNoiseGenerator(MultiClassNoiseGenerator):
    def __init__(self,
                 num_classes,
                 feat_dim=512,
                 ema_decay=0.5,
                 dist='normal',
                 mu=0.0,
                 sigma=1.0,
                 precomputed_stats=None,
                 update_mu=True,
                 update_sigma=True
                 ) -> None:
        super().__init__(num_classes, feat_dim, ema_decay, dist, mu, sigma, precomputed_stats, update_mu, update_sigma)

    def update_params(self, batch_mu, batch_sigma) -> None:
        return
