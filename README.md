## Introduction
DeepPrime is a deep-learning-based prime editing efficiency prediction tool developed in [Laboratory of Genome Editing, Yonsei University](https://sites.google.com/site/hyongbumkimlab/home). It is a successor to [DeepPE](https://www.nature.com/articles/s41587-020-0677-y), which was developed to predict prime editing efficiencies of a limited number of length combinations.

## What's New
* DeepPrime is designed to predict efficiencies of a wide variety of pegRNA combinations.
* We integrated CNN & RNN to extract inter-sequence features between target DNA and corresponding pegRNA.
* DeepPrime was trained using 259K pegRNAs with PBS lengths ranging from 1 to 17, RT lengths ranging from 1 to 50, Edit positions ranging from 1 to 30, and editing lengths ranging from 1 to 3.

## Progress
* DeepPrime for pegRNA-based on-target prime editing efficiency prediction on HEK293T cell line is now available (`models/ontarget/final/`, `plots/ontarget/`, `results/ontarget/`).
* DeepPrime performance on various external data were released (`data/`).
* DeepPrime for different cell lines and different PE techniques is now available (`models/ontarget_variants/`, `plots/ontarget_variants/`, `results/ontarget_variants/`).
* DeepPrime for off-target edit prediction is now available (`models/offtarget/`, `plots/offtarget/`, `results/offtarget/`).

## Directory tree
```bash
├── data # PUT .csv files here
│   ├── genes # folder for preprocessed gene files (.npy)
│   │   └── ...
│   └── ...
│
├── models
│   ├── offtarget
│   ├── ontarget
│   └── ontarget_variants
│
├── plots
│   ├── offtarget
│   ├── ontarget
│   └── ontarget_variants
│
├── results
│   ├── offtarget
│   ├── ontarget
│   └── ontarget_variants
│
├── model.py
├── utils.py
├── plot.py
├── train.py # full-dataset train without cross validation.
├── test.py
│
│   # FINETUNING (DeepPrime-FT)
├── finetune.py
├── finetune_test.py
│
│   # OFF-TARGET MODEL
├── offtarget_train.py
└── offtarget_test.py
```

## Performance
DeepPrime performance evaluation result is now available in `plots/`.