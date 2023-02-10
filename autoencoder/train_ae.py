from autoencoder.AE import Model
import torch
from autoencoder.train import train, test
from utils import AverageMeter, plot_save_fig
import matplotlib.pyplot as plt
def train_ae(args, dataset):
    device = 'cuda:' + args.gpu if args.gpu else 'cpu'
    ae = Model(input_dim=dataset.trn.x.shape[-1], device=device).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.003, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int((dataset.trn.x.shape[0] // args.ae_batch_size)
                                                                         * (args.epochs // 2)), gamma=0.1)
    train_loss_meter = AverageMeter(f'train_loss', ':6.3f')
    loss_list = []
    print_every = 10
    for epoch in range(args.epochs):
        train_loss_meter = train(ae=ae, dataset=dataset, optimizer=optimizer, loss_meter=train_loss_meter,
                                       scheduler=scheduler, batch_size=args.ae_batch_size)
        if epoch % print_every == 0:
            print(f'Epoch:{epoch},  Train Loss:{train_loss_meter.avg}')
            loss_list.append(train_loss_meter.avg)

    plot_save_fig(range(0, epoch, print_every), loss_list, 'epoch', 'train loss', 'AE Train Loss',
                  'AE_train_loss.png')
    plt.clf()

    test_score = test(ae=ae, dataset=dataset)
    return test_score

