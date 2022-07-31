import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import GeneInteractionModel, BalancedMSELoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# LOAD & PREPROCESS GENES

files = [
    'DP_variant_293T_PE2_Conv_220428.csv',
    'DP_variant_293T_NRCH_PE2_Opti_220428.csv',
    'DP_variant_293T_PE2max_Opti_220428.csv',
    'DP_variant_HCT116_PE2_Opti_220428.csv',
    'DP_variant_MDA_PE2_Opti_220428.csv',
    'DP_variant_DLD1_PE4max_Opti_220728.csv',
    'DP_variant_DLD1_NRCHPE4max_Opti_220728.csv',
    'DP_variant_A549_PE4max_Opti_220728.csv',
    'DP_variant_293T_PE4max_Opti_220728.csv',
    ]

for fileidx in range(9):

    file = files[fileidx]

    finetune_data = pd.read_csv('data/' + file)

    gene_path = 'data/genes/' + file[:-4] + '.npy'

    if not os.path.isfile(gene_path):
        g_train = seq_concat(finetune_data)
        np.save(gene_path, g_train)
    else:
        g_train = np.load(gene_path)


    # FEATURE SELECTION

    train_features, train_target = select_cols(finetune_data)
    train_fold = finetune_data.Fold
    train_type = finetune_data.loc[:, ['type_sub', 'type_ins', 'type_del']]


    # NORMALIZATION

    x_train = (train_features - train_features.mean()) / train_features.std() # fit to finetuning data distribution.
    y_train = train_target
    y_train = pd.concat([y_train, train_type], axis=1)

    g_train = torch.tensor(g_train, dtype=torch.float32, device=device)
    x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32, device=device)


    # PARAMS

    freeze_conv = False
    use_scheduler = False
    batch_size = 512
    hidden_size = 128
    n_layers = 1
    n_models = 20

    if fileidx != 8:
        continue
    
    if fileidx == 0:
        learning_rate = 5e-4
        weight_decay = 0e-2
        n_epochs = 100
    elif fileidx == 1:
        learning_rate = 1e-3
        weight_decay = 0e-2
        n_epochs = 100
    elif fileidx == 2:
        learning_rate = 5e-3
        weight_decay = 1e-2
        n_epochs = 100
        use_scheduler = True
    elif fileidx == 3:
        learning_rate = 1e-2
        weight_decay = 1e-2
        n_epochs = 50
        use_scheduler = True
    elif fileidx == 4:
        learning_rate = 4e-3
        weight_decay = 1e-2
        n_epochs = 50
        use_scheduler = True
    elif fileidx == 5:
        learning_rate = 8e-3
        weight_decay = 1e-2
        n_epochs = 50
        use_scheduler = True
    elif fileidx == 6:
        learning_rate = 1e-3
        weight_decay = 0e-2
        n_epochs = 100
    elif fileidx == 7:
        learning_rate = 4e-3
        weight_decay = 2e-2
        n_epochs = 100
    elif fileidx == 8:
        learning_rate = 5e-3
        weight_decay = 1e-2
        n_epochs = 100


    # TRAINING

    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers).to(device)

        model.load_state_dict(torch.load('models/ontarget/final/model_{}.pt'.format(m % 5)))
        
        if freeze_conv:
            for name, param in model.named_parameters():
                if name.startswith('c'):
                    param.requires_grad = False

        train_set = GeneFeatureDataset(g_train, x_train, y_train, fold_list=train_fold)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = BalancedMSELoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            train_loss = []
            train_count = 0

            for i, (g, x, y) in enumerate(train_loader):
                g = g.permute((0, 3, 1, 2))
                y = y.reshape(-1, 4)

                pred = model(g, x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(x.size(0) * loss.detach().cpu().numpy())
                train_count += x.size(0)

            train_loss = sum(train_loss) / train_count
            pbar.set_description('M {:02} | {:.4}'.format(random_seed, train_loss))

        torch.save(model.state_dict(),'models/ontarget_variants/{}/final_model_{}.pt'.format(file[:-4], random_seed))
