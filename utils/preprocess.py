# Save mean and standard deviation of training set.


import pandas as pd
from utils.data import select_cols

train_file = pd.read_csv('data/DeepPrime_dataset_final_Feat8.csv')

train_features, _ = select_cols(train_file[train_file['Fold']!='Test'])

mean, std = train_features.mean(), train_features.std()

mean.to_csv('data/mean.csv', header=False)
std.to_csv('data/std.csv', header=False)