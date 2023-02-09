import torch
from nice import nice
from nice.train import train, test
from utils import AverageMeter
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

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    train_ll_meter = AverageMeter(f'train_ll', ':6.3f')

    for epoch in range(args.epochs):
        train_ll_meter = train(flow, dataset, optimizer, train_ll_meter)
        print(f'Epoch:{epoch},  Train Loss:{-train_ll_meter.avg}')

    return test(flow, dataset)
