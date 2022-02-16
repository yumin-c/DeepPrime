import sys
import os
import math

import numpy as np
from numpy.random import shuffle
import scipy
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
from torch.backends import cudnn
from sklearn.model_selection import train_test_split, KFold

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import wandb

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hyperparameter_defaults = dict(
    c_1=64,
    c_2=64,
    c_3=16,
    hidden_size=128,
    learning_rate=5e-4,
    weight_decay=1e-2,
    T_0=30,
    n_layers=1,
    epochs=100
)

wandb.init(config=hyperparameter_defaults, project="DeepSpCas9")
config = wandb.config


class ContinuousBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.from_last_epoch = []

    def __iter__(self):
        idx_from_sampler = set(self.sampler)
        idx_to_exclude = set(self.from_last_epoch)
        idx_after_exclusion = sorted(list(idx_from_sampler - idx_to_exclude))
        shuffle(idx_after_exclusion)
        first_batch = self.from_last_epoch + \
            idx_after_exclusion[:self.batch_size - len(self.from_last_epoch)]
        yield first_batch
        idx_of_left = sorted(
            idx_after_exclusion[self.batch_size - len(self.from_last_epoch):] + list(idx_to_exclude))
        shuffle(idx_of_left)
        batch = []
        for idx in idx_of_left:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last:
            self.from_last_epoch = batch.copy()

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + len(self.from_last_epoch)) // self.batch_size


class ConvNeXt(nn.Module):

    def __init__(self, in_chans=4,
                 depths=[6, 3], dims=[64, 16], drop_path_rate=0.5,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.depths = depths

        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths)-1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item()
                    for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.norm = LayerNorm(dims[-1], eps=1e-6,
                              data_format="channels_first")  # before GRU
        self.head = nn.Linear(dims[-1], 1)
        # self.head = nn.Linear(dims[-1]*7, 1)

        self.r = nn.GRU(dims[-1], 128, 1,
                        batch_first=True, bidirectional=True)
        self.d = nn.Linear(2 * 128, 1, bias=True)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # return self.norm(x.mean([-1])) # global average pooling, (N, C, L) -> (N, C)
        # return x
        return self.norm(x)  # before GRU

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        # x = self.head(x.flatten(1))
        x, _ = self.r(torch.transpose(x, 1, 2))
        x = self.d(x[:, -1, :])
        return x


class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7,
                                padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)

        x = input + self.drop_path(x)
        return x


class ConvGRU(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super(ConvGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.c = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=config.c_1,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=config.c_1, out_channels=config.c_2,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=config.c_2, out_channels=config.c_3,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.r = nn.GRU(config.c_3, hidden_size, num_layers,
                        batch_first=True, bidirectional=True)
        self.d = nn.Linear(2 * hidden_size, 1, bias=True)

    def forward(self, x):
        x = self.c(x)
        x, _ = self.r(torch.transpose(x, 1, 2))
        x = self.d(x[:, -1, :])

        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = 30

    DATA_X = np.zeros((len(data), 1, length, 4), dtype=int)
    print(np.shape(data), len(data), length)
    for l in range(len(data)):
        for i in range(length):

            try:
                data[l][i]
            except:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 0, i, 3] = 1
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()

    print("Preprocessed the sequence")
    return DATA_X


def load_data():
    if not os.path.isfile('data/x_train.npy'):
        data_train = pd.read_excel('aax9249_table_s1.xlsx',
                                   sheet_name=0).iloc[:, [1, 8]]
        data_test = pd.read_excel('aax9249_table_s1.xlsx',
                                  sheet_name=1).iloc[:, [0, 4]]

        x_train = data_train.iloc[:, 0]
        x_test = data_test.iloc[:, 0]
        y_train = data_train.iloc[:, 1].to_numpy()
        y_test = data_test.iloc[:, 1].to_numpy()

        x_train = preprocess_seq(x_train)
        x_test = preprocess_seq(x_test)

        x_train = 2 * x_train - 1
        x_valid = 2 * x_test - 1
        y_train = (y_train - 40) / 100
        y_valid = (y_test - 40) / 100

        np.save('data/x_train.npy', x_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/x_test.npy', x_test)
        np.save('data/y_test.npy', y_test)
    else:
        x_train = np.load('data/x_train.npy')
        y_train = np.load('data/y_train.npy')
        x_test = np.load('data/x_test.npy')
        y_test = np.load('data/y_test.npy')

    return x_train, y_train, x_test, y_test


def train(x=None, y=None, x_test=None, y_test=None, device=None, finalize=False):

    k = 5
    batch_size = 256

    learning_rate = config.learning_rate
    weight_decay = config.weight_decay

    T_0 = config.T_0
    T_mult = 1

    hidden_size = config.hidden_size

    n_layers = config.n_layers
    n_epochs = config.n_epochs
    n_models = 1

    kfold = KFold(n_splits=k, shuffle=False)

    spc = {}

    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        # FOLD PREDICTIONS FOR ENSEMBLE
        preds = np.zeros((n_models, y_test.size(0)))
        spc[m] = []  # TO LOG SPEARMAN CORRELATION SCORES

        if finalize:
            train_set = TensorDataset(x, y)

            train_loader = DataLoader(train_set, batch_sampler=ContinuousBatchSampler(
                sampler=SequentialSampler(range(len(train_set))), batch_size=batch_size, drop_last=False
            ))

            # model = ConvNeXt().to(device)
            model = ConvGRU(hidden_size=hidden_size,
                            num_layers=n_layers).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

            n_iters = len(train_loader)

            for epoch in range(n_epochs):
                train_epoch_loss = []
                train_count = 0

                model.train()
                for i, (_x, _y) in enumerate(train_loader):
                    _x = torch.transpose(_x.squeeze(), 1, 2)
                    _y = _y.reshape(-1, 1)

                    pred = model(_x)
                    loss = criterion(pred, _y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(epoch + i / n_iters)

                    train_epoch_loss.append(
                        _x.size(0) * loss.detach().cpu().numpy())
                    train_count += _x.size(0)

                train_epoch_loss = sum(train_epoch_loss) / train_count

                print('[M {:03}/{:03}] [E {:03}/{:03}] : {:.4f}'.format(m +
                      1, n_models, epoch + 1, n_epochs, train_epoch_loss))

            if x_test is not None:
                model.eval()

                with torch.no_grad():
                    _x = torch.transpose(x_test.squeeze(), 1, 2)
                    pred = model(_x)
                    score = scipy.stats.spearmanr(
                        pred.detach().cpu(), y_test.cpu()).correlation

                    spc[m].append(score)
                    print(spc[m])

                    preds[m] = pred.squeeze().detach().cpu()
        else:
            for f, (train_idx, valid_idx) in enumerate(kfold.split(x)):
                
                if f > 0: continue

                train_set = TensorDataset(x[train_idx], y[train_idx])
                x_valid, y_valid = x[valid_idx], y[valid_idx]

                train_loader = DataLoader(train_set, batch_sampler=ContinuousBatchSampler(
                    sampler=SequentialSampler(range(len(train_set))), batch_size=batch_size, drop_last=False
                ))

                # model = ConvNeXt().to(device)
                model = ConvGRU(hidden_size=hidden_size,
                                num_layers=n_layers).to(device)
                wandb.watch(model)

                criterion = nn.MSELoss()
                optimizer = optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

                n_iters = len(train_loader)

                for epoch in range(n_epochs):
                    train_epoch_loss, valid_epoch_loss = [], []
                    train_count = 0

                    model.train()
                    for i, (_x, _y) in enumerate(train_loader):
                        _x = torch.transpose(_x.squeeze(), 1, 2)
                        _y = _y.reshape(-1, 1)

                        pred = model(_x)
                        loss = criterion(pred, _y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step(epoch + i / n_iters)

                        train_epoch_loss.append(
                            _x.size(0) * loss.detach().cpu().numpy())
                        train_count += _x.size(0)

                    train_epoch_loss = sum(train_epoch_loss) / train_count

                    model.eval()

                    with torch.no_grad():
                        _x = torch.transpose(x_valid.squeeze(), 1, 2)
                        pred = model(_x)
                        valid_epoch_loss = criterion(
                            pred, y_valid.reshape(-1, 1)).detach().cpu()
                        score = scipy.stats.spearmanr(
                            pred.detach().cpu(), y_valid.cpu()).correlation

                    metrics = {'train_loss': train_epoch_loss, 'valid_loss': valid_epoch_loss, 'Spearman score': score}

                    wandb.log(metrics)

                    print('[FOLD {:02}/{:02}] [M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(f + 1, k, m + 1,
                                                                                                                 n_models, epoch + 1, n_epochs, train_epoch_loss, valid_epoch_loss, score))
                
                wandb.log(metrics)

                if x_test is not None:
                    model.eval()

                    with torch.no_grad():
                        _x = torch.transpose(x_test.squeeze(), 1, 2)
                        pred = model(_x)
                        score = scipy.stats.spearmanr(
                            pred.detach().cpu(), y_test.cpu()).correlation

                        spc[m].append(score)
                        print(spc[m])

    print(spc)

    if finalize:
        print(preds.shape)
        preds = np.mean(preds, axis=0) * 100 + 40
        print(preds.shape)
        y_test = y_test.cpu().numpy() * 100 + 40

        print(scipy.stats.spearmanr(preds, y_test))


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x_train, y_train, x_test, y_test = load_data()

    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    train(x_train, y_train, x_test, y_test, device, finalize=False)


if __name__ == '__main__':
    main()
