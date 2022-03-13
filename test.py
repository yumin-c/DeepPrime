# .
# ├── data # PUT DATA HERE
# │   ├── genes
# │   ├── Biofeature_output_Profiling_220205_PE_effi_for_CYM.csv
# │   ├── DeepPrime_input_PEmax_220228.csv
# │   ├── DeepPrime_Nat_Liu_endo_PE2only_220303.csv
# │   ├── ...
# │   ├── DeepPrime_PECV__train_220214.csv
# │   ├── DeepPrime_PECV__test_220214.csv
# │   ├── g_pf.npy
# │   └── g_train.npy
# ├── models
# │   ├── pretrained
# │   └── test # PUT MODELS HERE
# │       ├── final_model_0.pt
# │       ├── final_model_1.pt
# │       ├── final_model_2.pt
# │       └── ...
# ├── plots
# ├── results
# ├── DeepPE.py
# ├── DeepPE_finalize.py
# ├── plot.py
# └── test.py # THIS FILE

import os

import numpy as np
import scipy
import pandas as pd

import torch

from DeepPE import GeneInteractionModel, seq_concat
import plot


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def select_cols(data):
    features = data.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                            'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                            'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                            'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]
    target = data.Measured_PE_efficiency

    return features, target


# PREPROCESSING

file_list = ['DeepPrime_PECV__test_220214.csv',
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
             'Biofeature_output_Profiling_220205_PE_effi_for_CYM.csv'
             ]

for file in file_list:

    train_PECV = pd.read_csv('data/DeepPrime_PECV__train_220214.csv')
    test_file = pd.read_csv('data/' + file)

    if 'PE2' in test_file.columns:
        test_file = test_file[test_file['PE2'] == 'O'].reset_index(drop=True)

    train_features, _ = select_cols(train_PECV)
    test_features, test_target = select_cols(test_file)

    gene_path = 'data/genes/' + file[:-4] + '.npy'

    if not os.path.isfile(gene_path):
        g_test = seq_concat(test_file)
        np.save(gene_path, g_test)
    else:
        g_test = np.load(gene_path)

    x_test = (test_features - train_features.mean()) / train_features.std()
    y_test = test_target
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    g_test = torch.tensor(g_test, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    # LOAD MODELS

    models, preds = [], []

    for (path, dir, files) in os.walk('models/test/'):
        for filename in files:
            if filename[-3:] == '.pt':
                models.append('models/test/' + filename)


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

            pred = model(g, x)

            if pred_ is None:
                pred_ = pred.detach().cpu().numpy()
                y_ = y.detach().cpu().numpy()
            else:
                pred_ = np.concatenate((pred_, pred.detach().cpu().numpy()))
                y_ = np.concatenate((y_, y.detach().cpu().numpy()))

        preds.append(pred_)

    # AVERAGE PREDICTIONS

    preds = np.squeeze(np.array(preds))
    preds = np.mean(preds, axis=0)
    preds = np.exp(preds) - 1
    y_ = y_[:, 0]


    # SHOW SCORE

    print(scipy.stats.spearmanr(preds, y_).correlation)

    Ts = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    Rs = []

    for i in range(len(Ts)):
        indices = []

        for j in range(len(y_)):
            if y_[j] > Ts[i]:
                indices.append(j)

        y_t = y_[indices]
        preds_t = preds[indices]
        corr = scipy.stats.spearmanr(preds_t, y_t).correlation
        print('Thr = {:4}, n = {:05} | {:2.5}'.format(Ts[i], len(y_t), corr))
        Rs.append(corr)


    # SAVE RESULTS

    plot.plot_spearman(preds, y_, 'plots/' + file[:-4] + '.jpg')

    preds = pd.DataFrame(preds, columns=['Predicted_PE_efficiency'])
    preds = pd.concat([test_file, preds], axis=1)

    preds.to_csv('results/' + file[:-4] + '.csv', index=False)