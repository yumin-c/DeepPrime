import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import GeneInteractionModel, MSLELoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def select_cols_(data):
    features = data.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                            'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    if 'Measured_PE_efficiency' in data.columns:
        target = data['Measured_PE_efficiency']
    else:
        target = data['Predicted_PE_efficiency'] * data['Relative_effi']
        
    return features, target


# LOAD & PREPROCESS GENES

mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

datasets = ['dataset_optim/DeepPrime_dataset01_read200_withOther.csv',
            'dataset_optim/DeepPrime_dataset02_read100_withOther.csv',
            'dataset_optim/DeepPrime_dataset03_read200_OtherX.csv',
            'dataset_optim/DeepPrime_dataset04_read100_OtherX.csv',
            'dataset_optim/DeepPrime_dataset05_read200_OtherX_AddFeat.csv',
            ] # must preserve order

f = open('dataset_optim/optim_cv_log.txt', 'w')

for dataset_idx, dataset in enumerate(datasets):
    print('\nTraining on ', dataset, ' \n')
    train_file = pd.read_csv(dataset)

    if not os.path.isfile('dataset_optim/g_{}.npy'.format(dataset_idx)):
        g_train = seq_concat(train_file)
        np.save('dataset_optim/g_{}.npy'.format(dataset_idx), g_train)
    else:
        g_train = np.load('dataset_optim/g_{}.npy'.format(dataset_idx))


    # FEATURE SELECTION

    if dataset_idx == 4:
        train_features, train_target = select_cols(train_file)
        mean_ = mean[['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                    'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                    'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
        std_ = std[['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    else:
        train_features, train_target = select_cols_(train_file)
        mean_ = mean[['PBSlen', 'RTlen', 'RT-PBSlen', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                        'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
        std_ = std[['PBSlen', 'RTlen', 'RT-PBSlen', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                    'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    train_fold = train_file.Fold


    # NORMALIZATION

    x_train = (train_features - mean_) / std_
    y_train = train_target

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
    n_models = 1


    # TRAIN & VALIDATION

    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        for fold in range(5):

            best_score = [10., 10., 0.]

            if dataset_idx == 4:
                model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers, num_features=24).to(device)
            else:
                model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers, num_features=18).to(device)

            train_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'train', train_fold)
            valid_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'valid', train_fold)

            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
            valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

            criterion = MSLELoss()
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

            n_iters = len(train_loader)

            for epoch in range(n_epochs):
                train_loss, valid_loss = [], []
                train_count, valid_count = 0, 0

                model.train()

                for i, (g, x, y) in enumerate(train_loader):
                    g = g.permute((0, 3, 1, 2))
                    y = y.reshape(-1, 1)

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
                        y = y.reshape(-1, 1)

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

                R = stats.spearmanr(pred_, y_).correlation

                if valid_loss < best_score[1]:
                    best_score = [train_loss, valid_loss, R]

                    # torch.save(model.state_dict(),'models/ontarget/{:02}_auxiliary.pt'.format(fold))

                print('[FOLD {:02}] [M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(fold, m + 1, n_models, epoch + 1, n_epochs, train_loss, valid_loss, R))
                f.write('[FOLD {:02}] [M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(fold, m + 1, n_models, epoch + 1, n_epochs, train_loss, valid_loss, R) + '\n')

            # os.rename('models/ontarget/{:02}_auxiliary.pt'.format(fold), 'models/ontarget/{:02}_{}_{:.4f}.pt'.format(fold, m, best_score[1]))
f.close()