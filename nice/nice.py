"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np

"""Additive coupling layer.
"""
def split_channels_even_odd(x):
    even = []
    odd = []
    for ind in range(x.size(1)):
        if ind % 2 == 0:
            even.append(x[:, ind])
        else:
            odd.append(x[:, ind])

    return torch.stack(even, dim=1), torch.stack(odd, dim=1)

def stack_data(kept, transformed, mask_config):
    columns_list = []

    if mask_config:
        for ind in range(kept.shape[1]):
            columns_list.append(transformed[:, ind])
            columns_list.append(kept[:, ind])

        if transformed.shape[1] > kept.shape[1]:
            columns_list.append(transformed[:, -1])
    else:
        for ind in range(transformed.shape[1]):
            columns_list.append(kept[:, ind])
            columns_list.append(transformed[:, ind])

        if kept.shape[1] > transformed.shape[1]:
            columns_list.append(kept[:, -1])

    return torch.stack(columns_list, dim=1)

class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        # TODO fill in
        self.mask_config = mask_config
        self.add_module('nn_module', construct_nn_module(int(in_out_dim / 2), mid_dim, hidden))

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # TODO fill in
        even, odd = split_channels_even_odd(x)

        if self.mask_config:
            transformed = self.nn_module(odd)
            kept = odd

            if reverse:
                transformed = even - transformed
            else:
                transformed = even + transformed
        else:
            transformed = self.nn_module(even)
            kept = even

            if reverse:
                transformed = odd - transformed
            else:
                transformed = odd + transformed


        stacked_cols = stack_data(kept, transformed, self.mask_config)

        return stacked_cols, log_det_J

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        # TODO fill in
        self.mask_config = mask_config
        self.add_module('nn_module', construct_nn_module(int(in_out_dim / 2), mid_dim, hidden, affine=True))
        self.eps = 1e-5

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # TODO fill in
        if self.mask_config:
            first, second = torch.chunk(x, 2, dim=1)
        else:
            second, first = torch.chunk(x, 2, dim=1)

        transformed = self.nn_module(second)
        kept = second
        scale, shift = torch.chunk(transformed, 2, dim=1)
        scale = torch.exp(scale)
        if reverse:
            transformed = torch.mul(first - shift, torch.reciprocal(scale))
            log_det_J = log_det_J - torch.log(scale + self.eps).view(x.shape[0], -1).sum(-1)
        else:
            transformed = torch.mul(first, scale) + shift
            log_det_J = log_det_J + torch.log(scale + self.eps).view(x.shape[0], -1).sum(-1)

        if self.mask_config:
            stacked_cols = torch.cat([transformed, kept], dim=1)
        else:
            stacked_cols = torch.cat([kept, transformed], dim=1)

        return stacked_cols, log_det_J

def construct_nn_module(in_out_dim, mid_dim, num_layers, affine=False):
    module = nn.ModuleList([nn.Linear(in_out_dim, mid_dim)])
    module.append(nn.ReLU())

    for _ in range(num_layers):
        module.append(nn.Linear(mid_dim, mid_dim))
        module.append(nn.ReLU())

    if affine:
        module.append(nn.Linear(mid_dim, in_out_dim * 2))
        module.append(nn.Sigmoid())
    else:
        module.append(nn.Linear(mid_dim, in_out_dim))

    return nn.Sequential(*module)


"""Log-scaling layer.
"""


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale) + self.eps

        # TODO fill in
        log_det_J = torch.sum(self.scale)
        if reverse:
            scale = 1 / scale

        return x * scale, log_det_J

"""Standard logistic distribution.
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""NICE main model.
"""


class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,
                 in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'affine'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type

        # TODO fill in
        self.coupling_layers = []
        for ind, _ in enumerate(range(coupling)):
            is_odd = 0 if ind % 2 == 0 else 1

            if coupling_type == 'additive':
                self.coupling_layers.append(AdditiveCoupling(in_out_dim, mid_dim, hidden, mask_config=is_odd))
            elif coupling_type == 'affine':
                self.coupling_layers.append(AffineCoupling(in_out_dim, mid_dim, hidden, mask_config=is_odd))
            else:
                raise NotImplementedError

            self.coupling_layers = nn.ModuleList(self.coupling_layers)
            # initialize weights:
            for p in self.coupling_layers[-1].parameters():
                if len(p.shape) > 1:
                    nn.init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    nn.init.normal_(p, mean=0., std=0.001)

        self.scaling_layer = Scaling(in_out_dim)

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        # TODO fill in
        x = z
        x, _ = self.scaling_layer(z, reverse=True)

        for layer in reversed(self.coupling_layers):
            x, _ = layer(x, 0, reverse=True)

        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        # TODO fill in
        z = x
        log_det_J = 0
        for layer in self.coupling_layers:
            z, log_det_J = layer(z.contiguous(), log_det_J)

        z, scale_log_det_J = self.scaling_layer(z)

        log_det_J = log_det_J + scale_log_det_J

        return z, log_det_J

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= self.in_out_dim  #np.log(256) * log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z.cpu()), dim=1).to(self.device)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        # TODO
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
