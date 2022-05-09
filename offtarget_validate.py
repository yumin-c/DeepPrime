import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import GeneInteractionModel

import wandb

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BalancedMSELoss(nn.Module):

    def __init__(self, mode='pretrain', scale=True):
        super(BalancedMSELoss, self).__init__()

        self.factor = [0.2, 1]
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x, pred, actual):
        pred = pred.view(-1, 1)
        y = torch.log1p(actual[:, 0].view(-1, 1))
        idx = actual[:, -1] == 7

        l1 = self.mse(pred[idx], y[idx]) * self.factor[0]
        l2 = self.mse(pred[~idx], y[~idx]) * self.factor[1]
            
        loss = (l1 + l2) / x.size(0)

        return loss


# LOAD & PREPROCESS GENES

off_data = pd.read_csv('data/DeepOff_dataset_220318.csv')
off_data = off_data[off_data['Fold'] != 'Test'].reset_index(drop=True)
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)


if not os.path.isfile('data/g_off.npy'):
    g_off = seq_concat(off_data, col1='WT74_ref', col2='Edited74_On')
    np.save('data/g_off.npy', g_off)
else:
    g_off = np.load('data/g_off.npy')


# FEATURE SELECTION

off_features, off_target = select_cols(off_data)
off_rha = off_data['RHA_len']
off_fold = off_data.Fold


# NORMALIZATION

x_off = (off_features - mean) / std
y_off = off_target
y_off = pd.concat([y_off, off_rha], axis=1)

g_off = torch.tensor(g_off, dtype=torch.float32, device=device)
x_off = torch.tensor(x_off.to_numpy(), dtype=torch.float32, device=device)
y_off = torch.tensor(y_off.to_numpy(), dtype=torch.float32, device=device)


# PARAMS

# wandb.init()
# config = wandb.config

batch_size = 512
learning_rate = 4e-3
weight_decay = 2e-2
hidden_size = 128
n_layers = 1
n_epochs = 10
n_models = 10


# TRAINING & VALIDATION

for m in range(n_models):

    random_seed = m

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    for fold in range(5):

        # if fold != 0:
        #     break

        best_score = [10., 10., 0.]

        model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers).to(device)
        model.load_state_dict(torch.load('models/pretrained/final_model_{}.pt'.format(m)))

        train_set = GeneFeatureDataset(g_off, x_off, y_off, str(fold), 'train', off_fold)
        valid_set = GeneFeatureDataset(g_off, x_off, y_off, str(fold), 'valid', off_fold)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = BalancedMSELoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        n_iters = len(train_loader)

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            train_loss, valid_loss = [], []
            train_count, valid_count = 0, 0

            model.train()

            for i, (g, x, y) in enumerate(train_loader):
                g = g.permute((0, 3, 1, 2))
                y = y.reshape(-1, 2)

                pred = model(g, x)
                loss = criterion(x, pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(x.size(0) * loss.detach().cpu().numpy())
                train_count += x.size(0)

            model.eval()

            pred_, y_ = None, None

            with torch.no_grad():
                for i, (g, x, y) in enumerate(valid_loader):
                    g = g.permute((0, 3, 1, 2))
                    y = y.reshape(-1, 2)

                    pred = model(g, x)
                    loss = criterion(x, pred, y)

                    valid_loss.append(x.size(0) * loss.detach().cpu().numpy())
                    valid_count += x.size(0)

                    if pred_ is None:
                        pred_ = pred.detach().cpu().numpy()
                        y_ = y.detach().cpu().numpy()[:, 0]
                    else:
                        pred_ = np.concatenate((pred_, pred.detach().cpu().numpy()))
                        y_ = np.concatenate((y_, y.detach().cpu().numpy()[:, 0]))

            train_loss = sum(train_loss) / train_count
            valid_loss = sum(valid_loss) / valid_count

            R = stats.spearmanr(pred_, y_).correlation

            # if train_loss < 0.1:
            #     break

            if R > best_score[2]:
                best_score = [train_loss, valid_loss, R]

            #     torch.save(model.state_dict(),'models/offtarget/{:02}_auxiliary.pt'.format(fold))

            # metrics = {'Train loss': train_loss, 'Valid loss': valid_loss, 'Spearman score': R, 'Best R': best_score[2]}
            # wandb.log(metrics)

            pbar.set_description('[FOLD {:02}] [M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(fold, m + 1, n_models, epoch + 1, n_epochs, train_loss, valid_loss, R))

        # os.rename('models/offtarget/{:02}_auxiliary.pt'.format(fold), 'models/offtarget/{:02}_{}_{:.4f}_{:.4f}.pt'.format(fold, m, best_score[1], best_score[2]))