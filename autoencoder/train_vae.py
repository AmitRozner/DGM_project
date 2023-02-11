from autoencoder.AE import Model
import torch
from autoencoder.train import train, test
from utils import AverageMeter, plot_save_fig
import matplotlib.pyplot as plt
from pyod.models.vae import VAE
def train_vae(args, dataset):
    vae = VAE(encoder_neurons=[dataset.trn.x.shape[-1], 16, 4], decoder_neurons=[4, 16, dataset.trn.x.shape[-1]],
             batch_size=args.ae_batch_size, epochs=args.epochs, contamination=1e-10, preprocessing=False)
    vae.fit(dataset.trn.x)
    outlier_conf, outlier_labels = vae.predict_proba(dataset.tst.x, return_confidence=True)
    return outlier_conf[:, 1]