"""Training procedure for NICE.
"""
import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import nice
from utils import create_batcher

def train(flow, dataset, optimizer, train_ll_meter):
    flow.train()  # set to training mode

    for i, inputs in enumerate(create_batcher(dataset.trn.x, batch_size=32)):
        optimizer.zero_grad()
        ll = flow(inputs.to('cuda:0')).mean()
        (-ll).backward()
        optimizer.step()
        train_ll_meter.update(ll.detach().cpu().numpy(), inputs.size(0))

    return train_ll_meter


def test(flow, dataset):
    flow.eval()  # set to inference mode
    log_like = []
    with torch.no_grad():
        for i, inputs in enumerate(create_batcher(dataset.tst.x, batch_size=32)):
            ll = flow(inputs.to('cuda:0'))
            log_like.append(ll.detach().cpu().numpy())

    return np.concatenate(log_like)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'

    full_dim = trainset.data[0, :].flatten().shape[0]
    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    # TODO fill in
    train_ll_meter = AverageMeter(f'train_ll', ':6.3f')
    test_ll_meter = AverageMeter(f'train_ll', ':6.3f')
    train_ll_list = []
    test_ll_list = []

    for epoch in tqdm(range(args.epochs)):
        train_ll_meter = train(flow, trainloader, optimizer, train_ll_meter)
        train_ll_list.append(train_ll_meter.avg)
        test_ll_meter = test(flow, testloader, model_save_filename, epoch, sample_shape, test_ll_meter)
        test_ll_list.append(test_ll_meter.avg)
        ## Sanity test
        # print(f"train:{train_ll_list}")
        # print(f"test:{test_ll_list}")

    fig, ax = plt.subplots()
    ax.plot(train_ll_list)
    ax.plot(test_ll_list)
    ax.set_title(f"Log Likelihood Loss for {args.dataset}. coupling_type: {args.coupling_type}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ll")
    ax.legend(["Train ll", "Test ll"])
    plt.savefig(os.path.join(os.getcwd(), model_save_filename + "_loss.png"))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='affine')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
