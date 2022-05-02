# Save mean and standard deviation of training set.

import pandas as pd
from utils import select_cols

train_PECV = pd.read_csv('data/DeepPrime_PECV__train_220214.csv')

train_features, _ = select_cols(train_PECV)

mean, std = train_features.mean(), train_features.std()

mean.to_csv('data/mean.csv', header=False)
std.to_csv('data/std.csv', header=False)