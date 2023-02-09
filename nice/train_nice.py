import torch
from nice import nice
from nice.train import train, test
from utils import AverageMeter, plot_save_fig
import matplotlib.pyplot as plt
import numpy as np


def train_nice(args, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataset.trn.x.shape[-1] % 2 != 0:
        dataset.trn.x = np.concatenate([dataset.trn.x, dataset.trn.x[:, -2:-1]], axis=-1)
        dataset.tst.x = np.concatenate([dataset.tst.x, dataset.tst.x[:, -2:-1]], axis=-1)

    full_dim = dataset.trn.x.shape[-1]
    flow = nice.NICE(
        prior='logistic',
        coupling=4,
        coupling_type='additive',
        in_out_dim=full_dim,
        mid_dim=10,
        hidden=5,
        device=device).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
    train_ll_meter = AverageMeter(f'train_ll', ':6.3f')
    print_every = 10
    loss_list = []
    for epoch in range(args.epochs):
        train_ll_meter = train(flow, dataset, optimizer, train_ll_meter, batch_size=args.nice_batch_size)

        if epoch % print_every == 0:
            print(f'Epoch:{epoch},  Train Loss:{-train_ll_meter.avg}')
            loss_list.append(-train_ll_meter.avg)

    plot_save_fig(range(0, epoch, print_every), loss_list, 'epoch', 'train log like', 'NICE Train Log Likelihood',
                  'NICE_train_ll.png')
    plt.clf()
    return test(flow, dataset)
