import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import GeneInteractionModel, BalancedMSELoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# LOAD & PREPROCESS GENES

fileidx = 0

files = ['DP_variant_293T_PE2_Conv_220428.csv',
         'DP_variant_293T_NRCH_PE2_Opti_220428.csv',
         'DP_variant_293T_PE2max_Opti_220428.csv',
         'DP_variant_HCT116_PE2_Opti_220428.csv',
         'DP_variant_MDA_PE2_Opti_220428.csv']

file = files[fileidx]

finetune_data = pd.read_csv('data/' + file)

gene_path = 'data/genes/' + file[:-4] + '.npy'

if not os.path.isfile(gene_path):
    g_train = seq_concat(finetune_data)
    np.save(gene_path, g_train)
else:
    g_train = np.load(gene_path)


# FEATURE SELECTION

train_features, train_target = select_cols(finetune_data)
train_fold = finetune_data.Fold
train_type = finetune_data.loc[:, ['type_sub', 'type_ins', 'type_del']]


# NORMALIZATION

# fit to finetuning data distribution.
x_train = (train_features - train_features.mean()) / train_features.std()
y_train = train_target
y_train = pd.concat([y_train, train_type], axis=1)

g_train = torch.tensor(g_train, dtype=torch.float32, device=device)
x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32, device=device)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32, device=device)


# PARAMS

freeze_conv = True
batch_size = 512
hidden_size = 128
n_layers = 1
n_models = 20

if fileidx == 0:
    freeze_conv = False
    learning_rate = 5e-4
    weight_decay = 0e-2
    n_epochs = 100
elif fileidx == 1:
    freeze_conv = False
    learning_rate = 1e-3
    weight_decay = 0e-2
    n_epochs = 100
elif fileidx == 2:
    freeze_conv = False
    learning_rate = 5e-4
    weight_decay = 0e-2
    n_epochs = 100
elif fileidx == 3:
    freeze_conv = False
    learning_rate = 5e-4
    weight_decay = 0e-2
    n_epochs = 50
elif fileidx == 4:
    freeze_conv = False
    learning_rate = 5e-4
    weight_decay = 0e-2
    n_epochs = 100


# TRAINING & VALIDATION

for m in range(n_models):

    random_seed = m

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    for fold in range(5):

        best_score = [10., 10., 0.]

        model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers).to(device)

        model.load_state_dict(torch.load('models/ontarget/mfe34/final_model_{}.pt'.format(m)))

        if freeze_conv:
            for name, param in model.named_parameters():
                if name.startswith('c'):
                    param.requires_grad = False

        train_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'train', train_fold)
        valid_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'valid', train_fold)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = BalancedMSELoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

        n_iters = len(train_loader)

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
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
                # scheduler.step(epoch + i / n_iters)

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

            R = stats.spearmanr(pred_, y_).correlation

            if valid_loss < best_score[1]:
                best_score = [train_loss, valid_loss, R]

                # torch.save(model.state_dict(), 'models/on_ft/{}/{:02}_auxiliary.pt'.format(file[:-4], fold))

            pbar.set_description('[FOLD {:02}] [M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(fold, m + 1, n_models, epoch + 1, n_epochs, train_loss, valid_loss, R))

        # os.rename('models/on_ft/{}/{:02}_auxiliary.pt'.format(file[:-4], fold), 'models/on_ft/{}/{:02}_{}.pt'.format(file[:-4], fold, m))
