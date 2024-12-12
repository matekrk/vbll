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
from vbll.layers.classification import VBLLReturn, KL

class GenClassification(nn.Module):
    """Variational Bayesian Generative Classification

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        regularization_weight : float
            Weight on regularization term in ELBO
        parameterization : str
            Parameterization of covariance matrix. Currently supports 'dense' and 'diagonal'
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
                 width_scale=1.0,
                 mumean_scale=0.1,
                 mulogiag_scale=1.,
                 noise_label=True,
                 wishart_scale=1.,
                 dof=1.):
        super(GenClassification, self).__init__()

        self.width_scale = width_scale if width_scale is not None else (2./in_features)
        self.noise_label = noise_label
        self.wishart_scale = wishart_scale
        self.dof = (dof + in_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(in_features), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(in_features))

        # last layer distribution
        self.mu_dist = get_parameterization(parameterization)
        self.mu_mean = nn.Parameter(mumean_scale*torch.randn(out_features, in_features))
        self.mu_logdiag = nn.Parameter(mulogiag_scale*torch.randn(out_features, in_features))
        if parameterization == 'dense':
            raise NotImplementedError('Dense embedding cov not implemented for g-vbll')

        self.softmax_bound = softmax_bound

        self.return_empirical = return_empirical
        if self.return_empirical and softmax_bound_empirical == 'montecarlo':
            self.softmax_bound_empirical = softmax_bound_empirical
        else:
            self.softmax_bound_empirical = None

        self.return_ood = return_ood

    def mu(self):
        # TODO(jamesharrison): add impl for dense/low rank cov
        return self.mu_dist(self.mu_mean, torch.exp(self.mu_logdiag))

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
        raise NotImplementedError('Adaptive bound not implemented for g-vbll')

    def jensen_bound(self, x, y):
        linear_pred = self.noise() + self.mu_mean[y]
        linear_term = linear_pred.log_prob(x)
        if isinstance(linear_pred, Normal):
            # Is there a more elegant way to handle this?
            linear_term = linear_term.sum(-1)

        trace_term = (self.mu().covariance_diagonal[y] / self.noise().covariance_diagonal).sum(-1)

        pre_lse_term = self.logit_predictive(x)
        lse_term = torch.logsumexp(pre_lse_term, dim=-1)
        return linear_term - 0.5 * trace_term - lse_term

    def montecarlo_bound(self, x, y, n_samples=10):
        # TODO(jamesharrison)
        # raise NotImplementedError('Monte carlo bound not implemented for g-vbll')
        sampled_noise = self.noise().rsample(sample_shape=torch.Size[n_samples])
        sampled_pred = sampled_noise + self.mu_mean[y].unsqueeze(0).expand(n_samples, -1)
        sampled_log_softmax = F.log_softmax(sampled_pred.log_prob(x).rsample(sample_shape=torch.Size([n_samples])), dim=-1)
        mean_over_samples = torch.mean(sampled_log_softmax, dim=0)
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
        # likelihood of x under marginalized
        logprob = (self.mu() + self.noise()).log_prob(x.unsqueeze(-2))
        if isinstance(self.mu(), Normal):
            # Is there a more elegant way to handle this?
            logprob = logprob.sum(-1)
        return logprob

    def predictive(self, x):
        return torch.clip(F.softmax(self.logit_predictive(x), dim=-1), min=0., max=1.)
    
    def logit_predictive_likedisc(self, x, n_samples=10):
        return torch.einsum("ijk,lk->ilj", (self.mu() + (self.noise()*int(self.noise_label))  ).rsample(torch.Size([n_samples])), x)
    
    def predictive_likedisc(self, x, n_samples=10): # HERE WE SHOULD TAKE MEAN
        return torch.clip(F.softmax(self.logit_predictive_likedisc(x, n_samples), dim=-1), min=0., max=1.)

    def _get_train_loss_fn(self, x, method):

        def loss_fn(y, n_samples = None):
            noise = self.noise()
            kl_term = KL(self.mu(), self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)
            wishart_term = int(self.noise_label) * wishart_term

            total_elbo = torch.mean(self.bound(x, y, method, n_samples))
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -torch.mean(torch.log(self.predictive(x)[np.arange(x.shape[0]), y]))

        return loss_fn

    # ----- OOD metrics

    def max_predictive(self, x):
        return torch.max(self.predictive(x), dim=-1)[0]
