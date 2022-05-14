# An file for on-target DeepPE (pre-)training.

import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import GeneInteractionModel, BalancedMSELoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# LOAD & PREPROCESS GENES

train_file = pd.read_csv('data/DeepPrime_dataset_final_Feat8.csv')
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

gene_path = 'data/genes/DeepPrime_dataset_final_Feat8.npy'
if not os.path.isfile(gene_path):
    g_train = seq_concat(train_file)
    np.save(gene_path, g_train)
else:
    g_train = np.load(gene_path)


# FEATURE SELECTION

train_features, train_target = select_cols(train_file)
train_fold = train_file.Fold
train_type = train_file.loc[:, ['type_sub', 'type_ins', 'type_del']]


# NORMALIZATION

x_train = (train_features - mean) / std
y_train = train_target
y_train = pd.concat([y_train, train_type], axis=1)

g_train = torch.tensor(g_train, dtype=torch.float32, device=device)
x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32, device=device)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32, device=device)


# PARAMS

batch_size = 2048
learning_rate = 5e-3
weight_decay = 5e-2
T_0 = 10
T_mult = 1
hidden_size = 128
n_layers = 1
n_epochs = 10
n_models = 5


# TRAINING

for m in range(n_models):

    random_seed = m

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers).to(device)

    train_set = GeneFeatureDataset(g_train, x_train, y_train, fold_list=train_fold)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = BalancedMSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

    n_iters = len(train_loader)

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
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
        pbar.set_description('M {:02} | {:.4}'.format(m, train_loss))

    torch.save(model.state_dict(),'models/ontarget/final_model_{}.pt'.format(random_seed))