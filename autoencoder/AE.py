"""AE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, device):
        """Initialize an AE.

        Args:
            input_dim: dimension of input
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim // 1.5)),
            nn.ReLU(True),
            nn.BatchNorm1d(int(input_dim // 1.5)),
            nn.Linear(int(input_dim // 1.5), int(input_dim // 2)),
            nn.ReLU(True),
            nn.BatchNorm1d(int(input_dim // 2)),
            nn.Linear(int(input_dim // 2), int(input_dim // 3))
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(input_dim // 3), int(input_dim // 2)),
            nn.ReLU(True),
            nn.BatchNorm1d(int(input_dim // 2)),
            nn.Linear(int(input_dim // 2), int(input_dim // 1.5)),
            nn.ReLU(True),
            nn.BatchNorm1d(int(input_dim // 1.5)),
            nn.Linear(int(input_dim // 1.5), input_dim)
        )

    def loss(self, x, recon):
        ce_loss = F.mse_loss(recon, x, reduction='mean')
        return ce_loss

    def forward(self, x):
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)
        return x_recon
