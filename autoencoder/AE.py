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
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 16),
            nn.SiLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, 4),
            nn.SiLU(True),
            nn.Dropout(0.2)
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 16),
            nn.SiLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, input_dim),
            nn.SiLU(True),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(True)
        )

    def loss(self, x, recon):
        ce_loss = F.mse_loss(recon, x, reduction='mean')
        # kl_loss = F.kl_div(F.log_softmax(x), F.softmax(recon), reduction='batchmean')
        return ce_loss #+ kl_loss

    def forward(self, x):
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)
        return x_recon
