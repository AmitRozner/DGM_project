import argparse
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
from nits.model import *
from nits.fc_model import *
from utils import create_batcher, list_str_to_list, plot_save_fig


def train_nits(args, data):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda:' + args.gpu if args.gpu else 'cpu'
    args.patience = args.patience if args.patience >= 0 else 10
    args.dropout = args.dropout if args.dropout >= 0.0 else 0.0
    print(args)

    d = data.trn.x.shape[1]
    max_val = 5
    min_val = -5
    max_val, min_val = torch.tensor(max_val).to(device).float(), torch.tensor(min_val).to(device).float()

    max_val *= args.bound_multiplier
    min_val *= args.bound_multiplier

    nits_model = NITS(d=d, start=min_val, end=max_val, monotonic_const=1e-5,
                      A_constraint='neg_exp', arch=[1] + args.nits_arch,
                      final_layer_constraint='softmax',
                      add_residual_connections=args.add_residual_connections,
                      normalize_inverse=(not args.dont_normalize_inverse),
                      softmax_temperature=False).to(device)

    model = ResMADEModel(
        d=d,
        rotate=args.rotate,
        nits_model=nits_model,
        n_residual_blocks=args.n_residual_blocks,
        hidden_dim=args.hidden_dim,
        dropout_probability=args.dropout,
        use_batch_norm=False,
        zero_initialization=True,
        weight_norm=False
    ).to(device)

    shadow = ResMADEModel(
        d=d,
        rotate=args.rotate,
        nits_model=nits_model,
        n_residual_blocks=args.n_residual_blocks,
        hidden_dim=args.hidden_dim,
        dropout_probability=args.dropout,
        use_batch_norm=False,
        zero_initialization=True,
        weight_norm=False
    ).to(device)

    model = EMA(model, shadow, decay=args.polyak_decay).to(device)

    print_every = 10
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.gamma)

    time_ = time.time()
    epoch = 0
    train_ll = 0.
    max_val_ll = -np.inf
    patience = args.patience
    keep_training = True
    train_ll_list = []
    ema_ll_list = []
    while keep_training:
        model.train()
        for i, x in enumerate(create_batcher(data.trn.x, batch_size=args.batch_size)):
            log_like = model(torch.tensor(x, device=device).float())
            optim.zero_grad()
            (-log_like.mean() + 1 * ((log_like - log_like.mean(dim=0)) ** 2).mean()).backward()
            train_ll += log_like.mean().cpu()
            optim.step()
            scheduler.step()
            model.update()

        epoch += 1

        if epoch % print_every == 0:
            # compute train loss
            train_ll /= len(data.trn.x) * print_every
            lr = optim.param_groups[0]['lr']

            with torch.no_grad():
                model.eval()
                val_ll = 0.
                ema_val_ll = 0.
                for i, x in enumerate(create_batcher(data.val.x, batch_size=args.batch_size)):
                    x = torch.tensor(x, device=device)
                    val_ll += model.model(x).detach().cpu().numpy()
                    ema_val_ll += model(x).detach().cpu().numpy()

                val_ll /= len(data.val.x)
                ema_val_ll /= len(data.val.x)

            # early stopping
            if ema_val_ll > max_val_ll + 1e-7:
                patience = args.patience
                max_val_ll = ema_val_ll
            else:
                patience -= 1

            if patience == 0:
                print("Patience reached zero. max_val_ll stayed at {:.3f} for {:d} iterations.".format(max_val_ll,
                                                                                                       args.patience))
                keep_training = False

            with torch.no_grad():
                model.eval()
                test_ll = 0.
                ema_test_ll = 0.
                for i, x in enumerate(create_batcher(data.tst.x, batch_size=args.batch_size)):
                    x = torch.tensor(x, device=device)
                    test_ll += model.model(x).detach().cpu().numpy()
                    ema_test_ll += model(x).detach().cpu().numpy()

                test_ll /= len(data.tst.x)
                ema_test_ll /= len(data.tst.x)

            fmt_str1 = 'epoch: {:3d}, time: {:3d}s, train_ll: {:.3f},'
            fmt_str2 = ' ema_val_ll: {:.3f}, ema_test_ll: {:.3f},'
            fmt_str3 = ' val_ll: {:.3f}, test_ll: {:.3f}, lr: {:.2e}'

            print((fmt_str1 + fmt_str2 + fmt_str3).format(
                epoch,
                int(time.time() - time_),
                train_ll,
                ema_val_ll,
                ema_test_ll,
                val_ll,
                test_ll,
                lr))
            train_ll_list.append(-train_ll.detach().cpu().numpy())
            ema_ll_list.append(ema_test_ll)
            time_ = time.time()
            train_ll = 0.

        if epoch % (print_every * 10) == 0:
            print(args)

    plot_save_fig(range(0, epoch, print_every), train_ll_list, 'epoch', 'train log like', 'NITS Train Log Likelihood', 'NITS_train_ll.png')
    plt.clf()
    # plot_save_fig(range(0, epoch, 10), ema_ll_list, 'epoch', 'EMA log like', 'NITS EMA Test Log Likelihood', 'NITS_ema_test_ll.png')
    log_like = model.model.forward_vec(torch.tensor(data.tst.x, device=device).float())
    score_all = log_like.detach().cpu().numpy()
    u, s = np.linalg.eigh(np.cov(score_all.T))
    ll_t = score_all @ s[:, -1:]
    ll_score = np.sum(ll_t, axis=1)
    ll_score = ll_score if np.cov(score_all.sum(axis=1), ll_score)[0, 1] > 0 else -ll_score
    return ll_score


def __main__():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='cardio')
    parser.add_argument('-g', '--gpu', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-hi', '--hidden_dim', type=int, default=1024)
    parser.add_argument('-nr', '--n_residual_blocks', type=int, default=2)
    parser.add_argument('-n', '--patience', type=int, default=-1)
    parser.add_argument('-ga', '--gamma', type=float, default=1)
    parser.add_argument('-pd', '--polyak_decay', type=float, default=1 - 5e-5)
    parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]')
    parser.add_argument('-r', '--rotate', action='store_true')
    parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
    parser.add_argument('-l', '--learning_rate', type=float, default=2e-5)
    parser.add_argument('-p', '--dropout', type=float, default=-1)
    parser.add_argument('-rc', '--add_residual_connections', type=bool, default=False)
    parser.add_argument('-bm', '--bound_multiplier', type=float, default=1.0)
    parser.add_argument('-pe', '--permute_data', action='store_true')

    args = parser.parse_args()
    d_in = scipy.io.loadmat('./datasets/cardio.mat')
    Y = d_in['y']
    datasets['cardio'] = Dataset(d_in['X'].astype(np.float), Y.T[0])
    train_nits(args, datasets['cardio'])
