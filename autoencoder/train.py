"""Training procedure for AE.
"""
import torch
import torch.nn.functional as F
from utils import create_batcher
import numpy as np

def train(ae, dataset, optimizer, loss_meter, scheduler, batch_size):
    ae.train()  # set to training mode

    for i, x_orig in enumerate(create_batcher(dataset.trn.x, batch_size=batch_size)):
        x_orig = x_orig.to(ae.device)
        optimizer.zero_grad()
        x_recon = ae(x_orig)
        loss = ae.loss(x_orig, x_recon)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_meter.update(loss.detach().cpu().numpy(), x_orig.size(0))
    return loss_meter

def test(ae, dataset):
    ae.eval()  # set to inference mode
    l1_loss = []
    with torch.no_grad():
        for i, x_orig in enumerate(create_batcher(dataset.tst.x, batch_size=512)):
            x_orig = x_orig.to(ae.device)
            x_recon = ae(x_orig)
            l1_loss.append(F.mse_loss(x_orig/x_orig.max(), x_recon/x_recon.max(), reduction='none').mean(-1).detach().cpu().numpy())

    return np.concatenate(l1_loss)
