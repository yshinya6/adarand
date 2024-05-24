import numpy as np
import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor
import pdb


class ClassifierUpdater:
    def __init__(self, *args, **kwargs):
        self.classifier = kwargs.pop("classifier")
        self.optimizer_c = kwargs.pop("optimizer_c")
        self.device = kwargs.pop("device")
        self.ema_model = kwargs.pop("ema_model")
        self.loss = F.cross_entropy

    def feature_entropy(self, feat_real):
        N, D = feat_real.size(0), feat_real.size(1)
        pair_dist = F.pairwise_distance(feat_real.unsqueeze(1), feat_real.unsqueeze(0))
        mask = torch.logical_not(torch.eye(N, N)).long().cuda()
        entropy = torch.sum(mask * torch.log(pair_dist)).mul(D).div(N * (N - 1))
        return entropy

    def get_batch(self, batch, device=None, non_blocking=True):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )

    def __call__(self, engine, batch):
        report = {}
        self.classifier.train()
        x, y = self.get_batch(batch, device=self.device)
        y_pred, feat_real = self.classifier(x)
        loss = self.loss(y_pred, y)
        self.optimizer_c.zero_grad()
        loss.backward()
        grad_norm = self.classifier.module.fc.weight.grad.norm()
        self.optimizer_c.step()
        report.update({"y_pred": y_pred.detach()})
        report.update({"y": y.detach()})
        report.update({"loss": loss.detach().item()})
        report.update({"feat_norm": feat_real.mean(dim=0).detach().cpu().norm()})
        report.update({"grad_norm": grad_norm.item()})
        feat_entropy = self.feature_entropy(feat_real).detach().item()
        report.update({"feat_entropy": feat_entropy})
        return report
