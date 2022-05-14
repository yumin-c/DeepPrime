import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import GeneInteractionModel

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
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

gene_path = 'data/genes/DeepOff_dataset_220318.npy'
if not os.path.isfile(gene_path):
    g_off = seq_concat(off_data, col1='WT74_ref', col2='Edited74_On')
    np.save(gene_path, g_off)
else:
    g_off = np.load(gene_path)


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

batch_size = 512
learning_rate = 4e-3
weight_decay = 1e-4
hidden_size = 128
n_layers = 1
n_epochs = 10
n_models = 5
T_0 = 10
T_mult = 1


# TRAINING & VALIDATION

for m in range(n_models):

    random_seed = m

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers, dropout=0.1).to(device)
    model.load_state_dict(torch.load('models/ontarget/final_model_{}.pt'.format(random_seed % 5)))

    train_set = GeneFeatureDataset(g_off, x_off, y_off, fold_list=off_fold)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    
    criterion = BalancedMSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

    n_iters = len(train_loader)

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        train_loss = []
        train_count = 0

        model.train()

        for i, (g, x, y) in enumerate(train_loader):
            g = g.permute((0, 3, 1, 2))
            y = y.reshape(-1, 2)

            pred = model(g, x)
            loss = criterion(x, pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / n_iters)

            train_loss.append(x.size(0) * loss.detach().cpu().numpy())
            train_count += x.size(0)

        train_loss = sum(train_loss) / train_count
        pbar.set_description('M {:02} | {:.4}'.format(m, train_loss))
    
    torch.save(model.state_dict(), 'models/offtarget/final_model_{}.pt'.format(random_seed))