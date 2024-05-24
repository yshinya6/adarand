import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor


class ClassifierUpdater:
    def __init__(
        self,
        lambda_reg=1.0,
        loss_type="l2",
        *args,
        **kwargs
    ):
        self.classifier = kwargs.pop("classifier")
        self.generator = kwargs.pop("generator")
        self.optimizer_c = kwargs.pop("optimizer_c")
        self.device = kwargs.pop("device")
        self.ema_model = kwargs.pop("ema_model")
        self.max_iter = kwargs.pop("max_iter")
        self.lambda_reg = lambda_reg
        self.loss_fn = F.cross_entropy

        self.feature_loss_fn = self.select_feature_loss(loss_type)

    def feature_entropy(self, feat_real):
        N, D = feat_real.size(0), feat_real.size(1)
        pair_dist = F.pairwise_distance(feat_real.unsqueeze(1), feat_real.unsqueeze(0))
        mask = torch.logical_not(torch.eye(N, N)).long().cuda()
        entropy = torch.sum(mask * torch.log(pair_dist)).mul(D).div(N * (N - 1))
        return entropy

    def select_feature_loss(self, loss_type):
        if loss_type == "l2":
            loss_fn = F.mse_loss
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

        x, y = self.get_batch(batch)
        logit_real, feat_real = self.classifier(x)
        feat_real = feat_real.squeeze()
        feat_noise = self.generator(y)

        # Calculate supervised loss
        loss_supervised = self.loss_fn(logit_real, y)
        report.update({"y_pred": logit_real.detach()})
        report.update({"y": y.detach()})
        report.update({"loss": loss_supervised.detach()})

        # Calculate feature regularization loss
        loss_pseudo = self.feature_loss_fn(feat_real, feat_noise)
        lambda_reg = self.lambda_reg
        loss_log = loss_pseudo.detach().item()
        report.update({"loss_pseudo": loss_log})

        # Calculate all losses and update classifier
        loss_target = loss_supervised + lambda_reg * loss_pseudo
        self.optimizer_c.zero_grad()
        loss_target.backward()
        grad_norm = self.classifier.module.fc.weight.grad.norm()
        self.optimizer_c.step()

        self.generator.module.update_params(feat_real.mean(dim=0).detach(), feat_real.std(dim=0).detach())

        report.update({"feat_norm": feat_real.mean(dim=0).detach().cpu().norm()})
        report.update({"grad_norm": grad_norm.detach().item()})

        if self.ema_model is not None:
            self.ema_model.update(self.classifier)

        feat_entropy = self.feature_entropy(feat_real).detach().item()
        report.update({"feat_entropy": feat_entropy})

        return report

class MultiClassUpdater(ClassifierUpdater):
    def __init__(
        self,
        lambda_reg=1.0,
        loss_type="l2",
        feat_dim=512,
        alpha=0.5,
        *args,
        **kwargs
    ):
        self.classifier = kwargs.pop("classifier")
        self.generator = kwargs.pop("generator")
        self.optimizer_c = kwargs.pop("optimizer_c")
        self.optimizer_g = kwargs.pop("optimizer_g")
        self.device = kwargs.pop("device")
        self.ema_model = kwargs.pop("ema_model")
        self.max_iter = kwargs.pop("max_iter")
        self.num_classes = kwargs.pop("num_classes")
        self.feat_dim = feat_dim
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.loss_fn = F.cross_entropy
        self.running_mu = self.generator.module.mu.data.clone().detach()
        self.running_sigma = self.generator.module.sigma.clone().detach()
        self.running_squared_mean = self.running_mu.pow(2)
        self.feature_loss_fn = self.select_feature_loss(loss_type)

    def compute_running_stats(self, batch_x, batch_y):
        for label in range(self.num_classes):
            index = torch.where(batch_y == label)[0]
            batch_x_label = batch_x[index]
            if len(index) == 0:
                continue
            batch_mu_t = batch_x_label.mean(dim=0)
            mu_t = self.alpha * self.running_mu[label] + (1 - self.alpha) * batch_mu_t
            self.running_mu[label] = mu_t
            mu_squared_t = self.alpha * self.running_squared_mean[label] + (1 - self.alpha) * batch_mu_t.pow(2)
            self.running_sigma[label] = mu_squared_t + mu_t.pow(2)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def td_loss(self):
        mu = self.generator.module.mu
        on_diag = F.cosine_similarity(mu, self.running_mu, dim=1).add(-1).pow_(2).mean()
        similarity_matrix_mu = F.cosine_similarity(mu.unsqueeze(1), mu.unsqueeze(0), dim=2)
        off_diag = self.off_diagonal(similarity_matrix_mu).pow_(2).mean()
        return on_diag + off_diag

    def __call__(self, engine, batch):
        report = {}
        self.classifier.train()

        x, y = self.get_batch(batch)
        logit_real, feat_real = self.classifier(x)
        feat_real = feat_real.squeeze()
        feat_noise = self.generator(y)

        # Calculate supervised loss
        loss_supervised = self.loss_fn(logit_real, y)
        report.update({"y_pred": logit_real.detach()})
        report.update({"y": y.detach()})
        report.update({"loss": loss_supervised.detach()})

        # Calculate feature regularization loss
        loss_pseudo = self.feature_loss_fn(feat_real, feat_noise)
        lambda_reg = self.lambda_reg
        loss_log = loss_pseudo.detach().item()
        report.update({"loss_pseudo": loss_log})

        # Calculate all losses and update classifier
        loss_target = loss_supervised + lambda_reg * loss_pseudo
        self.optimizer_c.zero_grad()
        loss_target.backward()
        grad_norm = self.classifier.module.fc.weight.grad.norm()
        self.optimizer_c.step()

        # Update generator parameters
        self.compute_running_stats(feat_real.detach(), y.detach())
        loss_td = self.td_loss()
        self.optimizer_g.zero_grad()
        loss_td.backward()
        self.optimizer_g.step()
        report.update({"loss_gen": loss_td.detach().item()})

        report.update({"feat_norm": feat_real.mean(dim=0).detach().cpu().norm()})
        report.update({"grad_norm": grad_norm.item()})

        if self.ema_model is not None:
            self.ema_model.update(self.classifier)

        feat_entropy = self.feature_entropy(feat_real).detach().item()
        report.update({"feat_entropy": feat_entropy})

        return report
