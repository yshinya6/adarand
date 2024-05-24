import functools
import pdb

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from ignite.utils import convert_tensor


class ClassifierUpdater:
    def __init__(
        self,
        arch="resnet18",
        strategy="std_normal",
        lambda_reg=1.0,
        precomputed_stats=None,
        loss_type="l2",
        stop_epoch=200,
        mu=0.0,
        sigma=1.0,
        *args,
        **kwargs
    ):
        self.classifier = kwargs.pop("classifier")
        self.optimizer_c = kwargs.pop("optimizer_c")
        self.device = kwargs.pop("device")
        self.ema_model = kwargs.pop("ema_model")
        self.max_iter = kwargs.pop("max_iter")
        self.lambda_reg = lambda_reg
        self.strategy = strategy
        self.loss_fn = F.cross_entropy
        self.arch = arch
        if self.strategy == "std_sample":
            self.mean, self.std = torch.tensor([mu]).to(self.device), torch.tensor([sigma]).to(self.device)
            self.random_dist = dists.Normal(loc=torch.tensor([mu]), scale=torch.tensor([sigma]))
        if self.strategy in ["uniform_sample", "uniform_fixed"]:
            self.mean, self.std = torch.tensor([mu]).to(self.device), torch.tensor([sigma]).to(self.device)
            self.random_dist = dists.Uniform(low=torch.tensor([mu]), high=torch.tensor([sigma]))
        elif self.strategy in ["precomputed_fixed", "precomputed_sample", "hybrid"]:
            assert precomputed_stats is not None
            stats = np.load(precomputed_stats)
            self.mean, self.std = torch.from_numpy(stats["mean"]).to(self.device), torch.from_numpy(stats["std"]).to(self.device)
            self.random_dist = dists.Normal(loc=self.mean, scale=self.std)

        self.feature_loss_fn = self.select_feature_loss(loss_type)
        self.stop_epoch = stop_epoch

    def feature_entropy(self, feat_real):
        N, D = feat_real.size(0), feat_real.size(1)
        pair_dist = F.pairwise_distance(feat_real.unsqueeze(1), feat_real.unsqueeze(0))
        mask = torch.logical_not(torch.eye(N, N)).long().cuda()
        entropy = torch.sum(mask * torch.log(pair_dist)).mul(D).div(N * (N - 1))
        return entropy

    def kl_div(self, feat_real, feat_noise, eps=1e-7):
        mean_real = feat_real.mean(dim=0)
        std_real = feat_real.std(dim=0)
        loss = torch.log((std_real / self.std) + eps) + (self.std.pow(2.0) + (self.mean - mean_real).pow(2.0)).div(2 * std_real.pow(2.0))
        return loss.mean()

    def nll(self, feat_real, feat_noise):
        return -self.random_dist.log_prob(feat_real).mean()

    def nll_normal(self, feat_real, feat_noise):
        return (feat_real - self.mean).div(self.std).pow(2.0).mean()

    def generate_random_feature(self, size):
        if self.strategy == "std_sample":
            f_rand = self.random_dist.sample(size).to(self.device)
        elif self.strategy == "uniform_sample":
            f_rand = self.random_dist.sample(size).to(self.device)
        elif self.strategy == "precomputed_sample":
            f_rand = self.random_dist.sample([size[0]]).to(self.device)
        elif self.strategy in ["precomputed_fixed", "uniform_fixed"]:
            f_rand = self.mean
        elif self.strategy == "hybrid":
            noise = torch.empty(size, dtype=torch.float32, device=self.device).normal_()
            f_rand = self.std * noise + self.mean
        else:
            raise NotImplementedError
        return f_rand.squeeze()

    def select_feature_loss(self, loss_type):
        if loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "sl1":
            loss_fn = F.smooth_l1_loss
        elif loss_type == "nll_normal":
            loss_fn = self.nll_normal
        elif loss_type == "nll":
            loss_fn = self.nll
        elif loss_type == "kl":
            loss_fn = self.kl_div
        else:
            raise NotImplementedError
        return loss_fn

    def get_batch(self, batch):
        x, y = batch
        return (
            convert_tensor(x, device=self.device, non_blocking=True),
            convert_tensor(y, device=self.device, non_blocking=True),
        )

    def __call__(self, engine, batch):
        report = {}
        self.classifier.train()

        # 1. Train classifier with pseudo semi-supervised learning
        # Generate pseudo features
        x, y = self.get_batch(batch)
        logit_real, feat_real = self.classifier(x)
        feat_real = feat_real.squeeze()
        feat_p = self.generate_random_feature(feat_real.size())

        # Calculate supervised loss
        loss_supervised = self.loss_fn(logit_real, y)
        report.update({"y_pred": logit_real.detach()})
        report.update({"y": y.detach()})
        report.update({"loss": loss_supervised.detach()})

        # Calculate feature regularization loss
        loss_pseudo = self.feature_loss_fn(feat_real, feat_p)
        if engine.state.epoch < self.stop_epoch:
            lambda_reg = self.lambda_reg
        else:
            lambda_reg = 0.0
        loss_log = loss_pseudo.detach().item()
        report.update({"loss_pseudo": loss_log})

        # Calculate all losses and update classifier
        loss_target = loss_supervised + lambda_reg * loss_pseudo
        self.optimizer_c.zero_grad()
        loss_target.backward()
        grad_norm = self.classifier.module.fc.weight.grad.norm()
        self.optimizer_c.step()

        if self.ema_model is not None:
            self.ema_model.update(self.classifier)

        report.update({"feat_norm": feat_real.mean(dim=0).detach().cpu().norm()})
        report.update({"grad_norm": grad_norm.item()})
        feat_entropy = self.feature_entropy(feat_real).detach().item()
        report.update({"feat_entropy": feat_entropy})

        return report
