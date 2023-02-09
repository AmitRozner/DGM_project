from autoencoder.AE import Model
import torch
from autoencoder.train import train, test
from utils import AverageMeter
def train_ae(args, dataset):
    device = 'cuda:' + args.gpu if args.gpu else 'cpu'
    ae = Model(input_dim=dataset.trn.x.shape[-1], device=device).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=dataset.trn.x.shape[0] * args.epochs, gamma=0.1)
    train_loss_meter = AverageMeter(f'train_loss', ':6.3f')
    for epoch in range(args.epochs):
        train_loss_meter = train(ae=ae, dataset=dataset, optimizer=optimizer, loss_meter=train_loss_meter,
                                       scheduler=scheduler)
        print(f'Epoch:{epoch},  Train Loss:{train_loss_meter.avg}')

    test_score = test(ae=ae, dataset=dataset)
    return test_score

