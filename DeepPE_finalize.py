import sys
import os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

from torch.utils.data import DataLoader
from DeepPE import GeneInteractionModel, BalancedMSELoss, GeneFeatureDataset, seq_concat

from tqdm import tqdm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


    # TRAINING

    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        model = GeneInteractionModel(
            hidden_size=hidden_size, num_layers=n_layers).to(device)

        train_set = GeneFeatureDataset(g_train, x_train, y_train)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = BalancedMSELoss(mode='finetuning')
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

        n_iters = len(train_loader)

        for epoch in tqdm(range(n_epochs)):
            train_loss = []
            train_count = 0

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

            train_loss = sum(train_loss) / train_count
            print(train_loss)

        torch.save(model.state_dict(),'models/final_model_{}.pt'.format(m))
