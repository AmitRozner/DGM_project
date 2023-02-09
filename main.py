import torch
import scipy
import numpy as np
import argparse
from utils import Dataset, list_str_to_list, calc_auc_score, normalize_dataset
from nits.train_nits import train_nits
from autoencoder.train_ae import train_ae
from nice.train_nice import train_nice
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, default='cardio')
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-s', '--seed', type=int, default=1)
parser.add_argument('-b', '--batch_size', type=int, default=512)
parser.add_argument('--ae_batch_size', type=int, default=32)
parser.add_argument('--nice_batch_size', type=int, default=32)
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
parser.add_argument('--epochs', type=int, default=1024)
parser.add_argument('--normalize_data', action='store_false')

args = parser.parse_args()

def get_datasets(normalize=False):
    datasets = {}
    d_in = scipy.io.loadmat('./datasets/cardio.mat')
    Y = d_in['y']
    datasets['cardio'] = Dataset(d_in['X'].astype(np.float), Y.T[0])
    if normalize:
        datasets['cardio'] = normalize_dataset(datasets['cardio'])
    d_in = scipy.io.loadmat('./datasets/ionosphere.mat')
    Y = d_in['y']
    datasets['ionosphere'] = Dataset(d_in['X'].astype(np.float), Y.T[0])
    if normalize:
        datasets['ionosphere'] = normalize_dataset(datasets['ionosphere'])
    return datasets

def main():
    datasets = get_datasets(args.normalize_data)
    nits_auc_score = {}
    ae_auc_score = {}
    nice_auc_score = {}
    for ds_name, dataset in datasets.items():
        args.dataset = ds_name
        nits_score = train_nits(args, dataset)
        nits_auc_score[ds_name] = calc_auc_score(nits_score, dataset.y_tst)
        ae_score = train_ae(args, dataset)
        ae_auc_score[ds_name] = calc_auc_score(ae_score, dataset.y_tst)
        nice_score = train_nice(args, dataset)
        nice_auc_score[ds_name] = calc_auc_score(nice_score, dataset.y_tst)
        a=1
if __name__ == "__main__":
    main()