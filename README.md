## Introduction
DeepPrime is a deep-learning-based prime editing efficiency prediction tool developed in [Laboratory of Genome Editing, Yonsei University](https://sites.google.com/site/hyongbumkimlab/home). It is a successor to [DeepPE](https://www.nature.com/articles/s41587-020-0677-y), which was developed to predict prime editing efficiencies of a limited number of length combinations.

## Directory tree
```bash
├── data # PUT .csv files here
│   ├── genes # folder for preprocessed gene files (.npy)
│   │   └── ...
│   └── ...
│
├── models # Trained models are stored here
│   ├── ontarget
│   ├── ontarget_variants
│   ├── offtarget
│   └── offtarget_variants
│
├── utils # Utilities for..
│   ├── data.py # Data preprocessing & Dataset
│   ├── model.py # Models and losses
│   └── preprocess.py # Save reference mean and standard deviation for normalization.
│
├── train_base.py
├── train_ft.py
└── train_off.py
```