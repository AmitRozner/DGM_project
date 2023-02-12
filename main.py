import torch
import scipy
import sklearn
import numpy as np
import argparse
from utils import Dataset, list_str_to_list, normalize_dataset
from nits.train_nits import train_nits
from autoencoder.train_ae import train_ae
from autoencoder.train_vae import train_vae
from nice.train_nice import train_nice
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, default='cardio')
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('--ae_batch_size', type=int, default=64)
parser.add_argument('--nice_batch_size', type=int, default=64)
parser.add_argument('-hi', '--hidden_dim', type=int, default=1024)
parser.add_argument('-nr', '--n_residual_blocks', type=int, default=2)
parser.add_argument('-n', '--patience', type=int, default=-1)
parser.add_argument('-ga', '--gamma', type=float, default=1)
parser.add_argument('-pd', '--polyak_decay', type=float, default=0.995)
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]')
parser.add_argument('-r', '--rotate', action='store_true')
parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-p', '--dropout', type=float, default=-1)
parser.add_argument('-rc', '--add_residual_connections', type=bool, default=False)
parser.add_argument('-bm', '--bound_multiplier', type=float, default=1.0)
parser.add_argument('-pe', '--permute_data', action='store_true')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--normalize_data', action='store_false')
parser.add_argument('--normalize_type', type=str, default='pyod')
parser.add_argument('--use_mod_nits_loss', action='store_false')

args = parser.parse_args()


def get_datasets(args, normalize=False):
    datasets = {}
    datasets = get_single_ds(datasets, args, normalize, ds_name='cardio')
    datasets = get_single_ds(datasets, args, normalize, ds_name='ionosphere')
    datasets = get_single_ds(datasets, args, normalize, ds_name='breastw')
    return datasets


def get_single_ds(datasets, args, normalize, ds_name):

    d_in = scipy.io.loadmat(f'./datasets/{ds_name}.mat')
    Y = d_in['y']
    datasets[ds_name] = Dataset(d_in['X'].astype(np.float), Y.T[0])
    if normalize:
        datasets[ds_name] = normalize_dataset(datasets[ds_name], normalize_type=args.normalize_type)
    return datasets


def main():
    datasets = get_datasets(args, args.normalize_data)
    nits_auc_score = {}
    ae_auc_score = {}
    nice_auc_score = {}
    vae_auc_score = {}
    NUM_RANDOM_SEEDS = 3
    for ds_name, dataset in tqdm(datasets.items()):
        args.dataset = ds_name
        nits_auc_score[ds_name] = []
        ae_auc_score[ds_name] = []
        nice_auc_score[ds_name] = []
        vae_auc_score[ds_name] = []
        for _ in range(NUM_RANDOM_SEEDS):
            nits_score = train_nits(args, dataset)
            nits_auc_score[ds_name].append(sklearn.metrics.roc_auc_score(dataset.y_tst, nits_score))
            ae_score = train_ae(args, dataset)
            ae_auc_score[ds_name].append(sklearn.metrics.roc_auc_score(dataset.y_tst, ae_score))
            nice_score = train_nice(args, dataset)
            nice_auc_score[ds_name].append(sklearn.metrics.roc_auc_score(dataset.y_tst, nice_score))
            vae_score = train_vae(args, dataset)
            vae_auc_score[ds_name].append(sklearn.metrics.roc_auc_score(dataset.y_tst, vae_score))

        print(f'nits_auc_score {ds_name}: {nits_auc_score[ds_name]} avg:{np.mean(nits_auc_score[ds_name])}')
        print(f'ae_auc_score {ds_name}: {ae_auc_score[ds_name]} avg:{np.mean(ae_auc_score[ds_name])}')
        print(f'nice_auc_score {ds_name}: {nice_auc_score[ds_name]} avg:{np.mean(nice_auc_score[ds_name])}')
        print(f'vae_auc_score {ds_name}: {vae_auc_score[ds_name]} avg:{np.mean(vae_auc_score[ds_name])}')

if __name__ == "__main__":
    main()
