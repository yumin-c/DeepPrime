import os
from tkinter import Y
import numpy as np
import pandas as pd
import torch
import yaml
import plot
from scipy import stats
from model import GeneInteractionModel
from utils import seq_concat, select_cols

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# PREPROCESSING

off_data = pd.read_csv('data/DeepOff_dataset_220318.csv')
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

gene_path = 'data/genes/DeepOff_dataset_220318.npy'
if not os.path.isfile(gene_path):
    g_off = seq_concat(off_data, col1='WT74_ref', col2='Edited74_On')
    np.save(gene_path, g_off)
else:
    g_off = np.load(gene_path)

off_features, off_target = select_cols(off_data)
off_fold = off_data.Fold

x_off = (off_features - mean) / std
y_off = off_target

test_idx = off_fold == 'Test'

g_test = g_off[test_idx]
x_test = x_off[test_idx]
y_test = y_off[test_idx]

g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32, device=device)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32, device=device)


# LOAD MODELS

models, preds = [], []

for (path, dir, files) in os.walk('models/offtarget/'):
    for filename in files:
        if filename[-3:] == '.pt':
            models.append('models/offtarget/' + filename)


# TEST

for m in models:
    model = GeneInteractionModel(hidden_size=128, num_layers=1).to(device)
    model.load_state_dict(torch.load(m))
    model.eval()
    with torch.no_grad():
        g = g_test.permute((0, 3, 1, 2))
        x = x_test
        y = y_test.reshape(-1, 1)
        pred = model(g, x).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
    preds.append(pred)


# AVERAGE PREDICTIONS

preds = np.squeeze(np.array(preds))
preds = np.mean(preds, axis=0)
preds = np.exp(preds) - 1
y = y[:, 0]


# SHOW SCORE

print(stats.spearmanr(preds, y).correlation)


# SAVE RESULTS

if True:
    pos_not5 = (off_data['Edit_pos'] != 5)[test_idx]
    plot.plot_spearman(preds[pos_not5], y[pos_not5], 'plots/offtarget/pos_not5.jpg')

plot.plot_spearman(preds, y, 'plots/offtarget/offtarget.jpg')

preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
preds = pd.concat([off_data, preds], axis=1)
preds.to_csv('results/offtarget/offtarget.csv', index=False)