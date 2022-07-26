# Sample test code for model performance evaluation.
# Locations for test_file, mean, std, model_folder should be redefined.

import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from model import GeneInteractionModel
from utils import seq_concat, select_cols

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# PREPROCESSING

test_file = 'data/DeepPrime_dataset_final_Feat8.csv' # Enter test file location here
test_file = pd.read_csv(test_file)
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True) # Train set mean (made with preprocessing.py)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True) # Train set std (made with preprocessing.py)

if 'PE2' in test_file.columns:
    test_file = test_file[test_file['PE2'] == 'O'].reset_index(drop=True)
    
test_features, test_target = select_cols(test_file)
test_fold = test_file.Fold

gene_path = 'data/genes/DeepPrime_dataset_final_Feat8.npy'

if not os.path.isfile(gene_path):
    g_test = seq_concat(test_file)
    np.save(gene_path, g_test)
else:
    g_test = np.load(gene_path)

x_test = (test_features - mean) / std
y_test = test_target

test_idx = test_fold == 'Test' # select 'Test' fold data

g_test = g_test[test_idx]
x_test = x_test[test_idx]
y_test = y_test[test_idx]

g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32, device=device)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32, device=device)


# LOAD MODELS

models, preds = [], []

model_dir = 'models/ontarget/final/' # Put models here (example)

for file in glob(model_dir + '*.pt'):
    models.append(file)


# TEST

for m in models:
    model = GeneInteractionModel(hidden_size=128, num_layers=1).to(device)
    model.load_state_dict(torch.load(m))
    model.eval()
    with torch.no_grad():
        g, x = g_test, x_test
        g = g.permute((0, 3, 1, 2))
        pred = model(g, x).detach().cpu().numpy()
    preds.append(pred)


# AVERAGE PREDICTIONS

preds = np.squeeze(np.array(preds))
preds = np.mean(preds, axis=0)
preds = np.exp(preds) - 1


# SAVE RESULTS

preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
preds.to_csv('prediction.csv', index=False)