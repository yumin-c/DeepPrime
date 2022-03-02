import sys
import os

import numpy as np
from scipy import stats
import pandas as pd

from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class GeneInteractionModel(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super(GeneInteractionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128,
                      kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=108,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=108,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(128, hidden_size, num_layers,
                        batch_first=True, bidirectional=True)

        self.s = nn.Linear(2 * hidden_size, 12, bias=False)

        self.d = nn.Sequential(
            nn.Linear(27, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128, bias=False)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(0.1),
            nn.Linear(140, 1, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(torch.transpose(g, 1, 2))
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)


class BalancedMSELoss(nn.Module):

    def __init__(self, mode='pretrain'):
        super(BalancedMSELoss, self).__init__()

        if mode == 'pretrain':
            self.factor = [1, 1, 0.7]
        elif mode == 'finetuning':
            self.factor = [1, 0.7, 0.6]

        # self.mse = nn.MSELoss()
        self.mse = ScaledMSELoss()

    def forward(self, pred, actual):
        pred = pred.view(-1, 1)
        y = torch.log1p(actual[:, 0].view(-1, 1))

        l1 = self.mse(pred[actual[:, 1] == 1], y[actual[:, 1] == 1]) * self.factor[0]
        l2 = self.mse(pred[actual[:, 2] == 1], y[actual[:, 2] == 1]) * self.factor[1]
        l3 = self.mse(pred[actual[:, 3] == 1], y[actual[:, 3] == 1]) * self.factor[2]

        return l1 + l2 + l3


class ScaledMSELoss(nn.Module):

    def __init__(self):
        super(ScaledMSELoss, self).__init__()

    def forward(self, pred, y):
        # mu = torch.minimum(torch.exp(7 * (y-2.8)) + 1, torch.ones_like(y) * 25) # inverse
        mu = torch.minimum(torch.exp(6 * (y-3)) + 1, torch.ones_like(y) * 5) # SQRT-inverse

        return torch.mean(mu * (y-pred) ** 2)


class GeneFeatureDataset(Dataset):

    def __init__(
        self,
        gene: torch.Tensor = None,
        features: torch.Tensor = None,
        target: torch.Tensor = None,
        fold: int = None,
        mode: str = 'train',
        fold_list: np.ndarray = None,
    ):
        self.fold = fold
        self.mode = mode
        self.fold_list = fold_list

        if self.fold_list is not None:
            self.indices = self._select_fold()
            self.gene = gene[self.indices]
            self.features = features[self.indices]
            self.target = target[self.indices]
        else:
            self.gene = gene
            self.features = features
            self.target = target

    def _select_fold(self):
        selected_indices = []

        if self.mode == 'valid':  # SELECT A SINGLE GROUP
            for i in range(len(self.fold_list)):
                if self.fold_list[i] == self.fold:
                    selected_indices.append(i)
        elif self.mode == 'train':  # SELECT OTHERS
            for i in range(len(self.fold_list)):
                if self.fold_list[i] != self.fold:
                    selected_indices.append(i)
        else:  # FOR FINALIZING
            for i in range(len(self.fold_list)):
                selected_indices.append(i)

        return selected_indices

    def __len__(self):
        return len(self.gene)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gene = self.gene[idx]
        features = self.features[idx]
        target = self.target[idx]

        return gene, features, target


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = 74

    DATA_X = np.zeros((len(data), 1, length, 4), dtype=float)
    print(np.shape(data), len(data), length)
    for l in tqdm(range(len(data))):
        for i in range(length):

            try:
                data[l][i]
            except Exception:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 0, i, 3] = 1
            elif data[l][i] in "Xx":
                pass
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()

    print("Preprocessed the sequence")
    return DATA_X


def seq_concat(data):
    wt = preprocess_seq(data.WT74_On)
    ed = preprocess_seq(data.Edited74_On)
    g = np.concatenate((wt, ed), axis=1)
    g = 2 * g - 1

    return g


if __name__ == '__main__':

    # LOAD & PREPROCESS GENES

    train_PECV = pd.read_csv('data/DeepPrime_PECV__train_220214.csv')
    test_PECV = pd.read_csv('data/DeepPrime_PECV__test_220214.csv')

    if not os.path.isfile('data/g_train.npy'):
        g_train = seq_concat(train_PECV)
        np.save('data/g_train.npy', g_train)
    else:
        g_train = np.load('data/g_train.npy')

    if not os.path.isfile('data/g_test.npy'):
        g_test = seq_concat(test_PECV)
        np.save('data/g_test.npy', g_test)
    else:
        g_test = np.load('data/g_test.npy')


    # FEATURE SELECTION

    train_features = train_PECV.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                                        'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                                        'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                                        'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]
    train_fold = train_PECV.Fold
    train_target = train_PECV.Measured_PE_efficiency

    test_features = test_PECV.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                                      'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                                      'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                                      'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]
    test_target = test_PECV.Measured_PE_efficiency

    train_type = train_PECV.loc[:, ['type_sub', 'type_ins', 'type_del']]
    test_type = test_PECV.loc[:, ['type_sub', 'type_ins', 'type_del']]


    # NORMALIZATION

    x_train = (train_features - train_features.mean()) / train_features.std()
    y_train = train_target
    y_train = pd.concat([y_train, train_type], axis=1)
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    x_test = (test_features - train_features.mean()) / train_features.std()
    y_test = test_target
    y_test = pd.concat([y_test, test_type], axis=1)
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    g_train = torch.tensor(g_train, dtype=torch.float32, device=device)
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

    g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)


    # PARAMS

    batch_size = 2048
    learning_rate = 5e-3
    weight_decay = 5e-2
    T_0 = 10
    T_mult = 1
    hidden_size = 128
    n_layers = 1
    n_epochs = 10
    n_models = 10

    use_pretrained = False


    # TRAINING & VALIDATION

    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        for fold in range(5):

            best_score = [10., 10., 0.]

            model = GeneInteractionModel(
                hidden_size=hidden_size, num_layers=n_layers).to(device)
            
            if use_pretrained:
                model.load_state_dict(torch.load('models/pretrained/3_1.6923.pt'))

                for name, param in model.named_parameters():
                    if param.requires_grad and name.startswith(('c', 'r', 's')):
                        param.requires_grad = True

            train_set = GeneFeatureDataset(g_train, x_train, y_train, fold, 'train', train_fold)
            valid_set = GeneFeatureDataset(g_train, x_train, y_train, fold, 'valid', train_fold)

            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
            valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

            criterion = BalancedMSELoss(mode='finetuning')
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

            n_iters = len(train_loader)

            for epoch in tqdm(range(n_epochs)):
                train_loss, valid_loss = [], []
                train_count, valid_count = 0, 0

                model.train()

                for i, (g, x, y) in enumerate(train_loader):
                    g = g.permute((0, 3, 1, 2))
                    y = y.reshape(-1, 4)

                    pred = model(g, x)
                    loss = criterion(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(epoch + i / n_iters)

                    train_loss.append(x.size(0) * loss.detach().cpu().numpy())
                    train_count += x.size(0)

                model.eval()

                pred_, y_ = None, None

                with torch.no_grad():
                    for i, (g, x, y) in enumerate(valid_loader):
                        g = g.permute((0, 3, 1, 2))
                        y = y.reshape(-1, 4)

                        pred = model(g, x)
                        loss = criterion(pred, y)

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

                SPR = stats.spearmanr(pred_, y_).correlation

                if valid_loss < best_score[1]:
                    best_score = [train_loss, valid_loss, SPR]

                    torch.save(model.state_dict(),'models/{:02}_auxiliary.pt'.format(fold))

                print('[FOLD {:02}] [M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(fold, m + 1, n_models, epoch + 1, n_epochs, train_loss, valid_loss, SPR))

            os.rename('models/{:02}_auxiliary.pt'.format(fold),
                      'models/{:02}_{}_{:.4f}.pt'.format(fold, m, best_score[1]))
