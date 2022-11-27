# Train code for DeepPrime-Off models.

import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data import GeneFeatureDataset, seq_concat, select_cols
from utils.model import GeneInteractionModel, OffTargetLoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# LOAD & PREPROCESS GENES

fileidx = 0

files = ['DeepOff_dataset_220604.csv',
         'DeepPrime-Off_dataset_293T_PE4max_221115.csv']

for directory in ['data/genes'] + ['models/offtarget_variants/' + x[:-4] for x in files]:
    if not os.path.exists(directory):
        os.makedirs(directory)

for fileidx in range(2):

    file = files[fileidx]

    off_data = pd.read_csv('data/' + file)

    gene_path = 'data/genes/' + file[:-4] + '.npy'

    if not os.path.isfile(gene_path):
        g_off = seq_concat(off_data, col1='WT74_ref', col2='Edited74_On')
        np.save(gene_path, g_off)
    else:
        g_off = np.load(gene_path)


    # FEATURE SELECTION

    off_features, off_target = select_cols(off_data)
    off_rha = off_data['RHA_len'] # RHA_len is used to handle dataset bias on instances with RHA_len of 7 (see OffTargetLoss in models.py)
    off_fold = off_data.Fold


    # NORMALIZATION

    mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
    std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

    x_off = (off_features - mean) / std
    y_off = off_target
    y_off = pd.concat([y_off, off_rha], axis=1)

    g_off = torch.tensor(g_off, dtype=torch.float32, device=device)
    x_off = torch.tensor(x_off.to_numpy(), dtype=torch.float32, device=device)
    y_off = torch.tensor(y_off.to_numpy(), dtype=torch.float32, device=device)


    # PARAMS

    batch_size = 256
    hidden_size = 128
    n_layers = 1
    n_models = 5
    offtarget_mutate = 0.05
    ontarget_mutate = 1.0

    if fileidx == 0:
        learning_rate = 4e-2
        weight_decay = 1e-2
        n_epochs = 5
        T_0 = 5
        T_mult = 1
    elif fileidx == 1:
        learning_rate = 2e-2
        weight_decay = 1e-2
        n_epochs = 8
        T_0 = 8
        T_mult = 1


    # TRAINING & VALIDATION

    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers, dropout=0.2).to(device)
        
        if fileidx == 0: model.load_state_dict(torch.load('models/ontarget/final/model_{}.pt'.format(random_seed)))
        elif fileidx == 1: model.load_state_dict(torch.load('models/ontarget_variants/DP_variant_293T_PE4max_Opti_220728/final_model_{}.pt'.format(random_seed)))

        train_set = GeneFeatureDataset(g_off, x_off, y_off, fold_list=off_fold, offtarget_mutate=offtarget_mutate, ontarget_mutate=ontarget_mutate, random_seed=random_seed)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = OffTargetLoss(dataset=fileidx)
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
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + i / n_iters)

                train_loss.append(x.size(0) * loss.detach().cpu().numpy())
                train_count += x.size(0)

            train_loss = sum(train_loss) / train_count
            pbar.set_description('M {:02} | {:.4}'.format(m, train_loss))
        
        if fileidx==0: torch.save(model.state_dict(), 'models/offtarget/final_model_{}.pt'.format(random_seed))
        elif fileidx==1: torch.save(model.state_dict(), 'models/offtarget_variants/{}/final_model_{}.pt'.format(file[:-4], random_seed))