# %%
import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import Transformer, BalancedMSELoss

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
train_file['Predicted_PE_efficiency'] = 0


# PARAMS

batch_size = 256
n_epochs = 50
n_models = 5
learning_rate = 2e-3
weight_decay = 1e-2
T_0 = 50
T_mult = 1
hidden_size = 32
nhead = 8
dim_feedforward = 96
num_encoder_layers = 2
dropout = 0.1
layer_norm_eps = 1e-5
num_features = 24

use_trainable_pe = True
use_reg_token = True


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

        model = Transformer(hidden_size=hidden_size,nhead=nhead, dim_feedforward=dim_feedforward, layer_norm_eps=layer_norm_eps, num_encoder_layers=num_encoder_layers,
                            num_features=num_features, dropout=dropout, use_trainable_pe=use_trainable_pe, use_reg_token=use_reg_token).to(device)

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

    R = stats.spearmanr(preds, y_).correlation
    r, _ = stats.pearsonr(preds, y_)

    print('Fold {} Spearman correlation: {}'.format(fold, R))
    print('Fold {} Pearson correlation: {}'.format(fold, r))
    train_file.loc[train_fold == str(fold), 'Predicted_PE_efficiency'] = preds

train_file.to_csv('results/transformer/final_5cv.csv', index=False)


'''
Fold 0 Spearman correlation: 0.8048530119453752
Fold 0 Pearson correlation: 0.7851954066387269

Fold 1 Spearman correlation: 0.807747721637263
Fold 1 Pearson correlation: 0.7845501771532212

Fold 2 Spearman correlation: 0.798446169547196
Fold 2 Pearson correlation: 0.7953866128653181

Fold 3 Spearman correlation: 0.8045821689071677
Fold 3 Pearson correlation: 0.7809100000327571

Fold 4 Spearman correlation: 0.7701770461776856
Fold 4 Pearson correlation: 0.7779827119485514
'''