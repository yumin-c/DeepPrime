

from tqdm import tqdm
import plot
import seaborn as sns

import wandb

def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length = 74

    DATA_X = np.zeros((len(data), 1, length, 4), dtype=float)
    print(np.shape(data), len(data), length)
    for l in tqdm(range(len(data))):
        for i in range(length):

            try:
                data[l][i]
            except:
                print(data[l], i, length, len(data))

            if data[l][i] in "Aa":
                DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 0, i, 3] = 1
            elif data[l][i] in "Xx":
                pass
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()

    print("Preprocessed the sequence")
    return DATA_X


def seq_concat(data):
    wt = preprocess_seq(data.WT74_On)
    ed = preprocess_seq(data.Edited74_On)
    g = np.concatenate((wt, ed), axis=1)
    g = 2 * g - 1

    return g

DIR = r'E:\Dropbox\Paired lib_PE2\Fig source data\PECV_550K\DeepPrime_dataset\for_CYM'
data = pd.read_csv('%s/DeepPrime_PECV__test_220214_for_study.csv' % DIR)
WT_data = data.WT74_On
ED_data = data.Edited74_On

WT_processed = preprocess_seq(WT_data)
ED_processed = preprocess_seq(ED_data)

print(WT_processed)
print(ED_processed)

seq_processed_cat = seq_concat(data)
print(seq_processed_cat)