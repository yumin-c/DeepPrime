# .
# ├── data # PUT DATA HERE
# │   ├── Biofeature_output_Profiling_220205_PE_effi_for_CYM.csv
# │   ├── DeepPrime_PECV__test_220214.csv
# │   ├── DeepPrime_PECV__train_220214.csv
# │   ├── g_pf.npy
# │   ├── g_test.npy
# │   └── g_train.npy
# ├── test
# │   ├── models # PUT MODELS HERE
# │   │   ├── FM00_0.6410.pt
# │   │   ├── FM01_0.6442.pt
# │   │   ├── FM02_0.6505.pt
# │   │   ├── FM03_0.6333.pt
# │   │   └── FM04_0.5675.pt
# │   ├── plots
# │   │   └── Evaluation of DeepPE2.jpg
# │   └── results
# │       └── 220220.csv
# ├── DeepPE.py
# ├── plot.py
# └── test.py


import os

import numpy as np
import scipy
import pandas as pd

import torch

from torch.utils.data import DataLoader

from DeepPE import GeneFeatureDataset, GeneInteractionModel, seq_concat

import plot

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# PREPROCESSING

train_PECV = pd.read_csv('data/DeepPrime_PECV__train_220214.csv')
test_PECV = pd.read_csv('data/DeepPrime_PECV__test_220214.csv')

train_features = train_PECV.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                                    'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                                    'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                                    'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]

test_features = test_PECV.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                                  'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                                  'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                                  'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]
test_target = test_PECV.Measured_PE_efficiency

if not os.path.isfile('data/g_test.npy'):
    g_test = seq_concat(test_PECV)
    np.save('data/g_test.npy', g_test)
else:
    g_test = np.load('data/g_test.npy')

x_test = (test_features - train_features.mean()) / train_features.std()
y_test = test_target
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

# MODEL PARAMS

batch_size = 2048
hidden_size = 128
n_layers = 1

test_set = GeneFeatureDataset(g_test, x_test, y_test)
test_loader = DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

models, preds = [], []

# LOAD MODELS 

for (path, dir, files) in os.walk('test/models/'):
        for filename in files:
            if filename[-3:] == '.pt':
                models.append('test/models/' + filename) 

# TEST

for m in models:
    
    model = GeneInteractionModel(
        hidden_size=hidden_size, num_layers=n_layers).to(device)

    model.load_state_dict(torch.load(m))

    pred_, y_ = None, None

    model.eval()
    with torch.no_grad():
        for i, (g, x, y) in enumerate(test_loader):
            g = g.permute((0, 3, 1, 2))
            y = y.reshape(-1, 1)

            pred = model(g, x)

            if pred_ is None:
                pred_ = pred.detach().cpu().numpy()
                y_ = y.detach().cpu().numpy()
            else:
                pred_ = np.concatenate(
                    (pred_, pred.detach().cpu().numpy()))
                y_ = np.concatenate((y_, y.detach().cpu().numpy()))

    preds.append(pred_)

# AVERAGE PREDICTIONS

preds = np.squeeze(np.array(preds))
preds = np.mean(preds, axis=0)
preds = np.exp(preds) - 1
y_ = y_

# SHOW SCORE

print(scipy.stats.spearmanr(preds, y_).correlation)

# SAVE RESULTS

preds = pd.DataFrame(preds, columns=['Predicted PE efficiency'])
preds.to_csv('test/results/220220.csv', index=False)

plot.plot_spearman(preds, y_, 'test/plots/Evaluation of DeepPE2.jpg')