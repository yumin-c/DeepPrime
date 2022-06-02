import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import BalancedMSELoss


class GeneInteractionModel(nn.Module):

    def __init__(self, num_features=24, dropout=0.1):
        super(GeneInteractionModel, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.cd = nn.Sequential(
            nn.Linear(9 * 128, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 12, bias=False),
        )

        self.d = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128, bias=False)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(dropout),
            nn.Linear(140, 1, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g = self.cd(g.reshape(-1, 9 * 128))

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# LOAD & PREPROCESS GENES

train_file = pd.read_csv('data/DeepPrime_dataset_final_Feat8.csv')
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)
gene_path = 'data/g_final_Feat8.npy'

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
n_layers = 1
n_epochs = 10
n_models = 5

use_pretrained = False


# TRAIN & VALIDATION

for fold in range(5):

    train_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'train', train_fold)
    valid_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'valid', train_fold)

    models, preds = [], []
    
    # Train multiple models to ensemble
    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        model = GeneInteractionModel().to(device)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

        criterion = BalancedMSELoss()
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
            
        models.append(model)

    # Ensemble results (bagging)
    for model in models:

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
        
        preds.append(pred_)
    
    preds = np.squeeze(np.array(preds))
    preds = np.mean(preds, axis=0)
    preds = np.exp(preds) - 1

    R = stats.spearmanr(pred_, y_).correlation

    print('CNN model with additional features.')
    print('Fold {} Spearman score: {}'.format(fold, R))


'''
Fold 0 Spearman score: 0.7990554534308922
Fold 1 Spearman score: 0.7967381820067888
Fold 2 Spearman score: 0.7915377318073978
Fold 3 Spearman score: 0.8002010368476832
Fold 4 Spearman score: 0.759108614298807
'''