import sklearn
from nits.fc_model import *
from scipy import stats
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from pyod.utils.utility import standardizer


def list_str_to_list(s):
    print(s)
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')

    s = [int(x) for x in s]

    return s


def create_batcher(x, batch_size=1, device='cuda:0'):
    idx = 0
    p = torch.randperm(len(x))
    x = x[p]

    while idx + batch_size < len(x):
        yield torch.tensor(x[idx:idx + batch_size], device=device).float()
        idx += batch_size
    else:
        yield torch.tensor(x[idx:], device=device).float()


class Dataset:
    def __init__(self, x, Y, permute=False, train_idx=0, val_idx=0):
        # splits x into train, val, and test
        self.n = len(x)
        x = stats.zscore(x, axis=0)
        x = x[:, np.std(x, axis=0) > 0]
        if permute:
            p = np.random.permutation(x.shape[1])

            x = x[:, p]

        x_n = x[Y == 0]
        x_a = x[Y == 1]

        ind = np.random.permutation(x_n.shape[0])

        x_n = x_n[ind, :]
        train_idx = train_idx if train_idx else int(0.5 * x_n.shape[0])
        val_idx = val_idx if val_idx else int(0.5 * x_n.shape[0])

        class DataHolder:
            def __init__(self, x):
                self.x = x

        #
        y_trn = np.zeros(train_idx)
        y_tst = np.zeros(self.n - train_idx)
        y_tst[-x_a.shape[0]:] = 1
        x_t = np.vstack((x_n[train_idx:], x_a))
        self.trn = DataHolder(x_n[:train_idx])
        self.val = DataHolder(x_t)
        self.tst = DataHolder(x_t)
        self.y_trn = y_trn
        self.y_tst = y_tst


def permute_data(dataset):
    d = dataset.trn.x.shape[1]
    train_idx = len(dataset.trn.x)
    val_idx = train_idx + len(dataset.val.x)
    x = np.concatenate([dataset.trn.x, dataset.val.x, dataset.tst.x], axis=0)

    P = np.eye(d)
    P = P[np.random.permutation(d)]
    permuted_x = np.matmul(x, P)
    assert np.allclose(np.matmul(permuted_x, P.T), x)

    return Dataset(permuted_x.astype(np.float), train_idx=train_idx, val_idx=val_idx), P.astype(np.float)


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


def plot_save_fig(x_val, y_val, xlabel, ylabel, title, figname):
    plt.plot(x_val, y_val)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figname)


def normalize_dataset(data, normalize_type='pyod'):
    trn_mean = np.mean(data.trn.x, axis=0)
    trn_std = np.std(data.trn.x, axis=0)

    if np.any(trn_std < 1e-7):
        # Remove any column which has std of zero
        col_to_keep = trn_std >= 1e-7
        trn_mean = trn_mean[col_to_keep]
        trn_std = trn_std[col_to_keep]
        data.trn.x = data.trn.x[:, col_to_keep]
        data.tst.x = data.tst.x[:, col_to_keep]
        data.val.x = data.val.x[:, col_to_keep]

    if normalize_type == 'zscore':
        data.trn.x = (data.trn.x - trn_mean) / trn_std
        data.tst.x = (data.tst.x - trn_mean) / trn_std
        data.val.x = (data.val.x - trn_mean) / trn_std
    elif normalize_type == 'whiten':
        trn_max = np.max(np.abs(data.trn.x), axis=0)
        trn_min = np.min(np.abs(data.trn.x), axis=0)
        data.trn.x = (data.trn.x - trn_min) / (trn_max - trn_min)
        data.val.x = (data.val.x - trn_min) / (trn_max - trn_min)
        data.tst.x = (data.tst.x - trn_min) / (trn_max - trn_min)
    elif normalize_type == 'pyod':
        data.trn.x, data.tst.x = standardizer(data.trn.x, data.tst.x)
    return data
