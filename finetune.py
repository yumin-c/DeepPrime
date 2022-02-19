
import sys
import os
import math
import copy

import numpy as np
from numpy.random import shuffle
import scipy
import pandas as pd

from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
from torch.backends import cudnn
from sklearn.model_selection import train_test_split, KFold

from tqdm import tqdm
import plot
import seaborn as sns

import wandb

def finetune_model(model, fold, pf_loader, valid_loader):

    # PARAMETERS FOR FINETUNING

    learning_rate = 5e-5
    weight_decay = 0e-6
    T_0 = 10
    T_mult = 1
    n_epochs = 10

    best_score = [10., 10., 0.]

    # TO LOCK THE GENE ABSTRACTION MODULE WHILE FINETUNING
    # for name, param in model.named_parameters():
    #     if param.requires_grad and name.startswith('c') or name.startswith('r'):
    #         param.requires_grad = False

    # criterion = BalancedMSELoss(mode='finetuning')

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

    n_iters = len(pf_loader)

    for epoch in tqdm(range(n_epochs)):
        train_loss, valid_loss = [], []
        train_count, valid_count = 0, 0

        model.train()

        for i, (g, x, y) in enumerate(pf_loader):
            g = g.permute((0, 3, 1, 2))
            y = y.reshape(-1, 4)

            pred = model(g, x)
            loss = criterion(pred, y[:, 0].view(-1))

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
                loss = criterion(pred, y[:, 0].view(-1))

                valid_loss.append(x.size(0) * loss.detach().cpu().numpy())
                valid_count += x.size(0)

                if pred_ is None:
                    pred_ = pred.detach().cpu().numpy()
                    y_ = y.detach().cpu().numpy()[:, 0]
                else:
                    pred_ = np.concatenate(
                        (pred_, pred.detach().cpu().numpy()))
                    y_ = np.concatenate((y_, y.detach().cpu().numpy()[:, 0]))

        train_loss = sum(train_loss) / train_count
        valid_loss = sum(valid_loss) / valid_count

        SPR = scipy.stats.spearmanr(pred_, y_).correlation
        
        if SPR > best_score[2]:
            best_score = [train_loss, valid_loss, SPR]

            torch.save(model.state_dict(),
                       'models/final/FM{:02}_auxiliary.pt'.format(fold))

        print('FINETUNING: [FOLD {:02}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(
            fold, epoch + 1, n_epochs, train_loss, valid_loss, SPR))

    os.rename('models/final/FM{:02}_auxiliary.pt'.format(fold),
              'models/final/FM{:02}.pt'.format(fold))