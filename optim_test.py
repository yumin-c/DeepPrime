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


def select_cols_(data):
    features = data.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                            'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    if 'Measured_PE_efficiency' in data.columns:
        target = data['Measured_PE_efficiency']
    else:
        target = data['Predicted_PE_efficiency'] * data['Relative_effi']

    return features, target


# PREPROCESSING

file_list = [
    #  'DeepPrime_PECV__test_220214.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_220303.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_TriAve_220303.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_TriAve_ExtFig5_220303.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_TriAve_ExtFig6_220303.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig2a_220303.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig2b_220303.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig3b_220303.csv',
    'DeepPrime_Nat_Liu_endo_PE2only_TriAve_Fig3c_220303.csv',
    'DeepPrime_input_PEmax_220303.csv',
    #  'DeepPrime_input_HT-56K_220304.csv',
    #  'Biofeature_output_Profiling_220205_PE_effi_for_CYM.csv'
]

dataset_idx = 4

for file in file_list:

    test_file = pd.read_csv('data/' + file)
    mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
    std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)

    gene_path = 'data/genes/' + file[:-4] + '.npy'

    if not os.path.isfile(gene_path):
        g_test = seq_concat(test_file)
        np.save(gene_path, g_test)
    else:
        g_test = np.load(gene_path)

    if dataset_idx == 4:
        test_features, test_target = select_cols(test_file)
        mean = mean[['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                    'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                    'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
        std = std[['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    else:
        test_features, test_target = select_cols_(test_file)
        mean = mean[['PBSlen', 'RTlen', 'RT-PBSlen', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                     'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
        std = std[['PBSlen', 'RTlen', 'RT-PBSlen', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                   'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    test_fold = test_file.Fold


    # NORMALIZATION

    x_test = (test_features - mean) / std
    y_test = test_target

    g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32, device=device)


    # LOAD MODELS

    models, preds = [], []

    for (path, dir, files) in os.walk('dataset_optim/models/{}/'.format(dataset_idx)):
        for filename in files:
            if filename[-3:] == '.pt':
                models.append('dataset_optim/models/{}/'.format(dataset_idx) + filename)


    # TEST

    for m in models:

        if dataset_idx == 4:
            model = GeneInteractionModel(hidden_size=128, num_layers=1, num_features=24).to(device)
        else:
            model = GeneInteractionModel(hidden_size=128, num_layers=1, num_features=18).to(device)
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

    print('Scores for ' + str(dataset_idx))
    print(stats.spearmanr(preds, y).correlation)


    # SAVE RESULTS

    plot.plot_spearman(preds, y, 'dataset_optim/plots/{}_{}.jpg'.format(dataset_idx, file))
    preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
    preds = pd.concat([test_file.reset_index(drop=True), preds], axis=1)
    preds.to_csv('dataset_optim/results/{}_{}.csv'.format(dataset_idx, file), index=False)
