import sys

import numpy as np
import scipy
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class Cas9Dataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]


class ConvGRU(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super(ConvGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.c = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=64, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.r = nn.GRU(16, hidden_size, num_layers,
                        batch_first=True, bidirectional=True)
        self.d = nn.Linear(2 * hidden_size, 1, bias=True)

    def forward(self, x):
        x = self.c(x)
        x, _ = self.r(torch.transpose(x, 1, 2))
        x = self.d(x[:, -1, :])

        return x


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = 30

    DATA_X = np.zeros((len(data), 1, length, 4), dtype=int)
    print(np.shape(data), len(data), length)
    for l in range(len(data)):
        for i in range(length):

            try:
                data[l][i]
            except:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 0, i, 3] = 1
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()

    print("Preprocessed the sequence")
    return DATA_X


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# PARAMS

batch_size = 128
learning_rate = 5e-4
weight_decay = 1e-2
hidden_size = 128
n_layers = 1
n_epochs = 100
n_models = 20

plot = False


# PREPROCESSING

data_train = pd.read_excel('aax9249_table_s1.xlsx',
                           sheet_name=0).iloc[:, [1, 8]]
data_valid = pd.read_excel('aax9249_table_s1.xlsx',
                           sheet_name=1).iloc[:, [0, 4]]

x_train = data_train.iloc[:, 0]
x_valid = data_valid.iloc[:, 0]
y_train = data_train.iloc[:, 1].to_numpy()
y_valid = data_valid.iloc[:, 1].to_numpy()

x_train = preprocess_seq(x_train)
x_valid = preprocess_seq(x_valid)

x_train = 2 * x_train - 1
x_valid = 2 * x_valid - 1
y_train = (y_train - 40) / 100
y_valid = (y_valid - 40) / 100

x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
x_valid = torch.tensor(x_valid, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_valid = torch.tensor(y_valid, dtype=torch.float32, device=device)

train_set = Cas9Dataset(x_train, y_train)

train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

preds = np.zeros((n_models, y_valid.size(0))) # PREDICTIONS FOR ENSEMBLE


# TRAINING & VALIDATION

for m in range(n_models):

    random_seed = m

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    model = ConvGRU(hidden_size=hidden_size, num_layers=n_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-7)

    names = ['Train loss', 'Test loss', 'Spearman score']
    values = {}
    for scheme in names:
        values[scheme] = []

    n_iters = len(train_loader)

    for epoch in range(n_epochs):
        train_epoch_loss, valid_epoch_loss = [], []
        train_count = 0

        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = torch.transpose(x.squeeze(), 1, 2)
            y = y.reshape(-1, 1)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / n_iters)

            train_epoch_loss.append(x.size(0) * loss.detach().cpu().numpy())
            train_count += x.size(0)

        train_epoch_loss = sum(train_epoch_loss) / train_count

        model.eval()
        with torch.no_grad():
            x = torch.transpose(x_valid.squeeze(), 1, 2)
            pred = model(x)
            valid_epoch_loss = criterion(
                pred, y_valid.reshape(-1, 1)).detach().cpu()
            score = scipy.stats.spearmanr(
                pred.detach().cpu(), y_valid.cpu()).correlation

        values[names[0]].append(train_epoch_loss)
        values[names[1]].append(valid_epoch_loss)
        values[names[2]].append(score)

        print('[M {:03}/{:03}] [E {:03}/{:03}] : {:.4f} | {:.4f} | {:.4f}'.format(m + 1,
              n_models, epoch + 1, n_epochs, train_epoch_loss, valid_epoch_loss, score))

    preds[m] = pred.squeeze().detach().cpu().numpy()


preds = np.mean(preds, axis=0) * 100 + 40
y_valid = y_valid.cpu().numpy() * 100 + 40

print(scipy.stats.spearmanr(preds, y_valid))


# PLOTTING 

if plot:
  _, ax = plt.subplots(figsize=(6, 4))

  for scheme in values:
      if scheme != names[2]:
          ax.plot(
              [n for n in range(len(values[scheme]))],
              [p for p in values[scheme]],
              label=scheme,
          )

  ax.legend(loc=4)
  ax.set_title("Train/Test MSE")
  # ax.set_yticks(range(0.0, 1.0, 0.02))
  plt.savefig('Loss.jpg')

  plt.close()

  plt.plot(
      [n for n in range(len(values[names[2]]))],
      [p for p in values[names[2]]],
      label=names[2],
  )
  plt.title("Test Spearman Score")
  plt.ylim(0.5, 0.9)
  plt.savefig('Spearman.jpg')
