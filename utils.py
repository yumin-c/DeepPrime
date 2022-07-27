import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm
import random


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = 74

    seq_onehot = np.zeros((len(data), 1, length, 4), dtype=float)
    print(np.shape(data), len(data), length)
    for l in tqdm(range(len(data))):
        for i in range(length):

            try:
                data[l][i]
            except Exception:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                seq_onehot[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                seq_onehot[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                seq_onehot[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                seq_onehot[l, 0, i, 3] = 1
            elif data[l][i] in "Xx":
                pass
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()

    print("Preprocessed the sequence")
    return seq_onehot


def seq_concat(data, col1='WT74_On', col2='Edited74_On'):
    wt = preprocess_seq(data[col1])
    ed = preprocess_seq(data[col2])
    g = np.concatenate((wt, ed), axis=1)
    g = 2 * g - 1

    return g


def select_cols(data):
    features = data.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                            'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                            'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score']]
    if 'Measured_PE_efficiency' in data.columns:
        target = data['Measured_PE_efficiency']
    elif 'real_Off-target_%' in data.columns:
        target = data['real_Off-target_%']
    else:
        target = []
        
    return features, target


class GeneFeatureDataset(Dataset):

    def __init__(
        self,
        gene: torch.Tensor = None,
        features: torch.Tensor = None,
        target: torch.Tensor = None,
        fold: int = None,
        mode: str = 'train',
        fold_list: np.ndarray = None,
        offtarget: bool = False,
        random_seed: int = 0,
    ):
        random.seed(random_seed)
        self.fold = fold
        self.mode = mode
        self.fold_list = fold_list
        self.offtarget = offtarget

        if self.fold_list is not None:
            self.indices = self._select_fold()
            self.gene = gene[self.indices]
            self.features = features[self.indices]
            self.target = target[self.indices]
        else:
            self.gene = gene
            self.features = features
            self.target = target

    def _select_fold(self):
        selected_indices = []

        if self.mode == 'valid':  # Select a single group
            for i in range(len(self.fold_list)):
                if self.fold_list[i] == self.fold:
                    selected_indices.append(i)
        elif self.mode == 'train':  # Select others
            for i in range(len(self.fold_list)):
                if self.fold_list[i] != self.fold and self.fold_list[i] != 'Test':
                    selected_indices.append(i)
        elif self.mode == 'finalizing':
            for i in range(len(self.fold_list)):
                selected_indices.append(i)

        return selected_indices
        

    def __len__(self):
        return len(self.gene)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gene = self.gene[idx]
        features = self.features[idx]
        target = self.target[idx]
        
        if self.offtarget:
            prob = random.random()
            
            if prob < 0.05: # Transform 5% of data to create dummy data with off-target efficiency of 0%
                mutated_sequence = gene[0, :, :]

                proportion = random.uniform(0.2, 1.0)
                replace_indices = random.sample(range(74), int(proportion * 74))

                for i in replace_indices:
                    mutated_sequence[i] = torch.tensor(random.choice([[1., -1., -1., -1.], [-1., 1., -1., -1.], [-1., -1., 1., -1.], [-1., -1., -1., 1.]]), dtype=torch.float32, device=gene.device)

                gene[0] = mutated_sequence
                target = torch.zeros_like(target)

        return gene, features, target
