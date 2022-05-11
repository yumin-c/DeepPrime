# Test script for on-target performance evaluation.

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

mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

file_list = [
             'DeepPrime_dataset_final_Feat8.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_220303.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_TriAve_220303.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_TriAve_ExtFig5_220303.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_TriAve_ExtFig6_220303.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig2a_220303.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig2b_220303.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig3b_220303.csv',
             'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig3c_220303.csv',
             'DeepPrime_input_PEmax_220303.csv',
             'DeepPrime_input_HT-56K_220304.csv',
            #  'Biofeature_output_Profiling_220205_PE_effi_for_CYM.csv',
            #  'DP_variant_293T_PE2max_Opti_220428.csv',
             ]

for file in file_list:

    test_file = pd.read_csv('data/' + file)

    if 'PE2' in test_file.columns:
        test_file = test_file[test_file['PE2'] == 'O'].reset_index(drop=True)

    test_features, test_target = select_cols(test_file)
    test_fold = test_file.Fold

    gene_path = 'data/genes/' + file[:-4] + '.npy'

    if not os.path.isfile(gene_path):
        g_test = seq_concat(test_file)
        np.save(gene_path, g_test)
    else:
        g_test = np.load(gene_path)

    x_test = (test_features - mean) / std
    y_test = test_target

    test_idx = test_fold == 'Test'

    g_test = g_test[test_idx]
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32, device=device)


    # LOAD MODELS

    models, preds = [], []

    for (path, dir, files) in os.walk('models/ontarget/final/'):
        for filename in files:
            if filename[-3:] == '.pt':
                models.append('models/ontarget/final/' + filename)


    # TEST

    for m in models:

        model = GeneInteractionModel(hidden_size=128, num_layers=1).to(device)
        model.load_state_dict(torch.load(m))

        pred_, y_ = None, None

        model.eval()

        with torch.no_grad():
            g = g_test
            x = x_test
            y = y_test

            g = g.permute((0, 3, 1, 2))
            y = y.reshape(-1, 1)

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

    plot.plot_spearman(preds, y, 'plots/' + file[:-4] + '.jpg')
    preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
    preds = pd.concat([test_file, preds], axis=1)
    preds.to_csv('results/' + file[:-4] + '.csv', index=False)