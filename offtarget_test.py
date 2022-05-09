# %%
import os
import numpy as np
import pandas as pd
import torch
import plot
from scipy import stats
from model import GeneInteractionModel
from utils import seq_concat, select_cols

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# PREPROCESSING

off_data = pd.read_csv('data/DeepOff_dataset_220318.csv')
off_data = off_data[off_data['Fold'] == 'Test'].reset_index(drop=True)
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

test_features, test_target = select_cols(off_data)

gene_path = 'data/g_off_test.npy'

if not os.path.isfile(gene_path):
    g_test = seq_concat(off_data, col1='WT74_ref', col2='Edited74_On')
    np.save(gene_path, g_test)
else:
    g_test = np.load(gene_path)


x_test = (test_features - mean) / std
y_test = test_target

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
        g = g_test
        x = x_test
        y = y_test
        g = g.permute((0, 3, 1, 2)) #[:, :, :, 4:34]
        y = y.reshape(-1, 1)
        pred = model(g, x).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
    preds.append(pred)


# AVERAGE PREDICTIONS

preds = np.squeeze(np.array(preds))
preds = np.mean(preds, axis=0)
preds = np.exp(preds) - 1
y = y[:, 0]

# %%

# SHOW SCORE

print(stats.spearmanr(preds, y).correlation)


# SAVE RESULTS

if True:
    pos_not5 = off_data['Edit_pos'] != 5
    plot.plot_spearman(preds[pos_not5], y[pos_not5], 'plots/offtarget/pos_not5.jpg')

plot.plot_spearman(preds, y, 'plots/offtarget/offtarget.jpg')

preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
preds = pd.concat([off_data, preds], axis=1)

preds.to_csv('results/offtarget/offtarget.csv', index=False)
# %%
