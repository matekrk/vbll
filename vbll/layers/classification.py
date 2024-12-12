from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
import abc
import warnings

from vbll.utils.distributions import Normal, DenseNormal, LowRankNormal, get_parameterization

def KL(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(mse_term + trace_term + logdet_term) # currently exclude constant

@dataclass
class VBLLReturn():
    predictive: Union[Normal, DenseNormal] # Could return distribution or mean/cov
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: Union[None, Callable[[torch.Tensor], torch.Tensor]] = None
    train_loss_fn_empirical: Union[None, Callable[[torch.Tensor], torch.Tensor]] = None

class DiscClassification(nn.Module):
    """Variational Bayesian Disciminative Classification

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        regularization_weight : float
            Weight on regularization term in ELBO
        parameterization : str
            Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank'}
        softmax_bound : str
            Bound to use for softmax. Currently supports 'jensen' and 'montecarlo'
        return_ood : bool
            Whether to return OOD scores
        prior_scale : float
            Scale of prior covariance matrix
        wishart_scale : float
            Scale of Wishart prior on noise covariance
        dof : float
            Degrees of freedom of Wishart prior on noise covariance
     """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 softmax_bound='jensen',
                 return_empirical=False,
                 softmax_bound_empirical=None,
                 return_ood=False,
                 prior_scale=1.,
                 width_scale=None,
                 wumean_scale=None,
                 wulogdiag_scale=None,
                 wuoffdiag_scale=None,
                 noise_label=True,
                 wishart_scale=1.,
                 cov_rank=None,
                 dof=1.):
        super(DiscClassification, self).__init__()

        self.width_scale = width_scale if width_scale is not None else (2./in_features)
        self.wumean_scale = np.sqrt(self.width_scale) if wumean_scale is None else wumean_scale
        self.wulogdiag_scale = wulogdiag_scale if wulogdiag_scale is not None else -2*np.log(in_features)
        self.wuoffdiag_scale = wuoffdiag_scale if wuoffdiag_scale is not None else 0.01
        self.noise_label = noise_label
        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * self.width_scale

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(out_features) - 1)

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features) * self.wumean_scale)

        self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - self.wulogdiag_scale) # - 0.5 * np.log(in_features))
        if parameterization == 'dense':
            self.W_offdiag = nn.Parameter(self.wuoffdiag_scale * torch.randn(out_features, in_features, in_features)/in_features)
        elif parameterization == 'lowrank':
            self.W_offdiag = nn.Parameter(self.wuoffdiag_scale * torch.randn(out_features, in_features, cov_rank)/in_features)
        
        self.softmax_bound = softmax_bound

        self.return_empirical = return_empirical
        if self.return_empirical and softmax_bound_empirical == 'montecarlo':
            self.softmax_bound_empirical = softmax_bound_empirical
        else:
            self.softmax_bound_empirical = None

        self.return_ood = return_ood

    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif self.W_dist == DenseNormal:
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    # ----- bounds

    def bound(self, x, y, method=None, n_samples=10):
        if method is None:
            method = self.softmax_bound
        if method == 'jensen':
            return self.jensen_bound(x, y)
        elif method == 'montecarlo':
            return self.montecarlo_bound(x, y, n_samples)
        else:
            raise ValueError("Invalid method specified")

    def adaptive_bound(self, x, y):
        # TODO(jamesharrison)
        raise NotImplementedError('Adaptive bound not currently implemented')

    def jensen_bound(self, x, y):
        pred = self.logit_predictive(x)
        linear_term = pred.mean[torch.arange(x.shape[0]), y]
        pre_lse_term = pred.mean + 0.5 * pred.covariance_diagonal
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)
        return linear_term - lse_term

    def montecarlo_bound(self, x, y, n_samples=10):
        sampled_log_sm = F.log_softmax(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        mean_over_samples = torch.mean(sampled_log_sm, dim=0)
        return mean_over_samples[torch.arange(x.shape[0]), y]

    # ----- forward and core ops

    def forward(self, x):
        # TODO(jamesharrison): add assert on shape of x input
        out = VBLLReturn(torch.distributions.Categorical(probs = self.predictive(x)),
                          self._get_train_loss_fn(x, method=self.softmax_bound),
                          self._get_val_loss_fn(x))
        if self.return_empirical: out.train_loss_fn_empirical = self._get_train_loss_fn(x, method=self.softmax_bound_empirical)
        if self.return_ood: out.ood_scores = self.max_predictive(x)
        return out

    def logit_predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise() * int(self.noise_label)

    def predictive(self, x, n_samples = 10):
        softmax_samples = F.softmax(self.logit_predictive(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        return torch.clip(torch.mean(softmax_samples, dim=0),min=0.,max=1.)

    def _get_train_loss_fn(self, x, method):

        def loss_fn(y, n_samples = None):
            noise = self.noise()

            kl_term = KL(self.W(), self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)
            wishart_term = int(self.noise_label) * wishart_term

            total_elbo = torch.mean(self.bound(x, y, method, n_samples))
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x)[torch.arange(x.shape[0]), y]))

        return loss_fn

    # ----- OOD metrics

    def max_predictive(self, x):
        return torch.max(self.predictive(x), dim=-1)[0]
