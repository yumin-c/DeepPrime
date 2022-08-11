import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from scipy import stats
import plot
from model import GeneInteractionModel
from utils import seq_concat, select_cols

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# PREPROCESSING

use_external = True # if true, evaluate using KimDS (2020) dataset.

if use_external:
    ext_name = 'KimDS_'

    off_data = pd.read_csv('data/Biofeature_output_KimDS_ref_220722_opti_scaffold.csv')
    mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
    std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

    gene_path = 'data/genes/Biofeature_output_KimDS_ref_220722_opti_scaffold.npy'
    if not os.path.isfile(gene_path):
        g_off = seq_concat(off_data, col1='WT74_ref', col2='Edited74_On')
        np.save(gene_path, g_off)
    else:
        g_off = np.load(gene_path)

    off_features, off_target = select_cols(off_data)

    x_off = (off_features - mean) / std
    y_off = off_target

    test_idx = y_off > 0.1

    g_test = g_off #[test_idx]
    x_test = x_off #[test_idx]
    y_test = y_off #[test_idx]

    g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32, device=device)

else:
    ext_name = ''

    off_data = pd.read_csv('data/DeepOff_dataset_220604.csv')
    mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
    std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

    gene_path = 'data/genes/DeepOff_dataset_220604.npy'
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

for m in glob('models/offtarget/*.pt'):
    print(m)
    models.append(m)


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

if use_external:
    zero_indices = []

    for i in range(len(off_data)):
        difference = 0
        on, ref = off_data['WT74_On'].iloc[i], off_data['WT74_ref'].iloc[i]
        rt_len = off_data['RTlen'].iloc[i]

        boundary = 17 + rt_len

        for j in range(boundary):
            if on[4+j] != ref[4+j]:
                difference += 1
        
        if difference > 4:
            zero_indices.append(i)
    
    preds[zero_indices] = 0


# SHOW SCORE

print(stats.spearmanr(preds, y).correlation)


# SAVE RESULTS

if use_external:
    pos_not5 = (off_data['Edit_pos'] != 5)
else:
    pos_not5 = (off_data['Edit_pos'] != 5)[test_idx]

plot.plot_spearman(preds[pos_not5], y[pos_not5], 'plots/offtarget/{}pos_not5.jpg'.format(ext_name), lim=50)
plot.plot_spearman(preds, y, 'plots/offtarget/{}offtarget.jpg'.format(ext_name), lim=50)

preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
if use_external:
    preds = pd.concat([off_data, preds], axis=1)
else:
    preds = pd.concat([off_data.loc[test_idx].reset_index(drop=True), preds], axis=1)
preds.to_csv('results/offtarget/{}offtarget.csv'.format(ext_name), index=False)