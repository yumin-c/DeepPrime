import sys
import os

import numpy as np
from scipy import stats
import pandas as pd

from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class GeneInteractionModel(nn.Module):

    def __init__(self, cdim, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(GeneInteractionModel, self).__init__()

        c1, c2, c3 = cdim

        self.c = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=c1,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c1),
            nn.GELU(),

            nn.Conv1d(in_channels=c1, out_channels=c2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=c2, out_channels=c3,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=c3, out_channels=dim,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 9 + 1, dim))
        self.reg_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


        self.d = nn.Sequential(
            nn.Linear(27, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128, bias=False),
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim + 128),
            # nn.Dropout(0.05),
            nn.Linear(dim + 128, 1, bias=True),
        )

    def forward(self, g, x):
        g = torch.reshape(g, (-1, 8, 74))
        g = self.c(g) # b x c(128) x x l(9)
        g = g.permute(0, 2, 1) # b x l x c

        b, n, _ = g.shape

        reg_tokens = repeat(self.reg_token, '() n d -> b n d', b = b)

        g = torch.cat((reg_tokens, g), dim=1)
        g += self.pos_embedding[:, :(n+1)]
        g = self.dropout(g)

        g = self.transformer(g)

        g = g.mean(dim = 1) if self.pool == 'mean' else g[:, 0]

        g = self.to_latent(g)

        x = self.d(x)
        
        out = self.mlp_head(torch.cat((g, x), dim=1))

        return F.softplus(out)


class BalancedMSELoss(nn.Module):

    def __init__(self, mode='pretrain'):
        super(BalancedMSELoss, self).__init__()

        if mode == 'pretrain':
            self.factor = [1, 1, 0.7]
        elif mode == 'finetuning':
            self.factor = [1, 0.7, 0.6]

        self.mse = nn.MSELoss()
        # self.mse = ScaledMSELoss()

    def forward(self, pred, actual):
        pred = pred.view(-1, 1)
        y = torch.log1p(actual[:, 0].view(-1, 1))

        l1 = self.mse(pred[actual[:, 1] == 1],
                      y[actual[:, 1] == 1]) * self.factor[0]
        l2 = self.mse(pred[actual[:, 2] == 1],
                      y[actual[:, 2] == 1]) * self.factor[1]
        l3 = self.mse(pred[actual[:, 3] == 1],
                      y[actual[:, 3] == 1]) * self.factor[2]

        return l1 + l2 + l3


class ScaledMSELoss(nn.Module):

    def __init__(self):
        super(ScaledMSELoss, self).__init__()

    def forward(self, pred, y):
        mu = torch.minimum(torch.exp(6 * (y-3)) + 1,
                           torch.ones_like(y) * 5)  # SQRT-inverse

        return torch.mean(mu * (y-pred) ** 2)


class GeneFeatureDataset(Dataset):

    def __init__(
        self,
        gene: torch.Tensor = None,
        features: torch.Tensor = None,
        target: torch.Tensor = None,
        fold: int = None,
        mode: str = 'train',
        fold_list: np.ndarray = None,
    ):
        self.fold = fold
        self.mode = mode
        self.fold_list = fold_list

        if self.fold_list is not None:
            self.indices = self._select_fold()
            self.gene = gene[self.indices]
            self.features = features[self.indices]
            self.target = target[self.indices]
        else:
            self.gene = gene
            self.features = features
            self.target = target

    def _select_fold(self):
        selected_indices = []

        if self.mode == 'valid':  # SELECT A SINGLE GROUP
            for i in range(len(self.fold_list)):
                if self.fold_list[i] == self.fold:
                    selected_indices.append(i)
        elif self.mode == 'train':  # SELECT OTHERS
            for i in range(len(self.fold_list)):
                if self.fold_list[i] != self.fold:
                    selected_indices.append(i)
        else:  # FOR FINALIZING
            for i in range(len(self.fold_list)):
                selected_indices.append(i)

        return selected_indices

    def __len__(self):
        return len(self.gene)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gene = self.gene[idx]
        features = self.features[idx]
        target = self.target[idx]

        return gene, features, target


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = 74

    DATA_X = np.zeros((len(data), 1, length, 4), dtype=float)
    print(np.shape(data), len(data), length)
    for l in tqdm(range(len(data))):
        for i in range(length):

            try:
                data[l][i]
            except Exception:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 0, i, 3] = 1
            elif data[l][i] in "Xx":
                pass
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()

    print("Preprocessed the sequence")
    return DATA_X


def seq_concat(data):
    wt = preprocess_seq(data.WT74_On)
    ed = preprocess_seq(data.Edited74_On)
    g = np.concatenate((wt, ed), axis=1)
    g = 2 * g - 1

    return g


if __name__ == '__main__':

    # LOAD & PREPROCESS GENES

    train_PECV = pd.read_csv('data/DeepPrime_PECV__train_220214.csv')

    if not os.path.isfile('data/g_train.npy'):
        g_train = seq_concat(train_PECV)
        np.save('data/g_train.npy', g_train)
    else:
        g_train = np.load('data/g_train.npy')

    # FEATURE SELECTION

    train_features = train_PECV.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                                        'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                                        'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                                        'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]
    train_fold = train_PECV.Fold
    train_target = train_PECV.Measured_PE_efficiency

    train_type = train_PECV.loc[:, ['type_sub', 'type_ins', 'type_del']]

    # NORMALIZATION

    x_train = (train_features - train_features.mean()) / train_features.std()
    y_train = train_target
    y_train = pd.concat([y_train, train_type], axis=1)
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    g_train = torch.tensor(g_train, dtype=torch.float32, device=device)
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

    # PARAMS

    batch_size = 512
    learning_rate = 1e-3
    weight_decay = 2e-2
    T_0 = 8
    T_mult = 1
    n_epochs = 8
    n_models = 1

    use_pretrained = False

    # TRAINING & VALIDATION

    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        for fold in range(5):

            best_score = [10., 10., 0.]

            model = GeneInteractionModel(
                cdim=(144, 144, 144),
                dim=96,
                depth=6,
                heads=16,
                mlp_dim=64,
                pool='reg',
                dim_head=32,
                dropout=0.1,
                emb_dropout=0.05,
            ).to(device)

            train_set = GeneFeatureDataset(
                g_train, x_train, y_train, fold, 'train', train_fold)
            valid_set = GeneFeatureDataset(
                g_train, x_train, y_train, fold, 'valid', train_fold)

            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
            valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

            criterion = BalancedMSELoss(mode='finetuning')
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

                model.eval()

                pred_, y_ = None, None

                with torch.no_grad():
                    for i, (g, x, y) in enumerate(valid_loader):
                        g = g.permute((0, 3, 1, 2))
                        y = y.reshape(-1, 4)

                        pred = model(g, x)
                        loss = criterion(pred, y)

                        valid_loss.append(
                            x.size(0) * loss.detach().cpu().numpy())
                        valid_count += x.size(0)

                        if pred_ is None:
                            pred_ = pred.detach().cpu().numpy()
                            y_ = y.detach().cpu().numpy()[:, 0]
                        else:
                            pred_ = np.concatenate(
                                (pred_, pred.detach().cpu().numpy()))
                            y_ = np.concatenate(
                                (y_, y.detach().cpu().numpy()[:, 0]))

                train_loss = sum(train_loss) / train_count
                valid_loss = sum(valid_loss) / valid_count

                SPR = stats.spearmanr(pred_, y_).correlation

                if valid_loss < best_score[1]:
                    best_score = [train_loss, valid_loss, SPR]

                    torch.save(model.state_dict(),'models/transformers/{:02}_auxiliary.pt'.format(fold))

                print('[FOLD {:02}] [M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(
                    fold, m + 1, n_models, epoch + 1, n_epochs, train_loss, valid_loss, SPR))

                # metrics = {'Train loss': train_loss,
                #            'Valid loss': valid_loss, 'Spearman score': SPR}
                # wandb.log(metrics)

            os.rename('models/transformers/{:02}_auxiliary.pt'.format(fold),
                      'models/transformers/{:02}_{}_{:.4f}.pt'.format(fold, m, best_score[1]))

            # break
