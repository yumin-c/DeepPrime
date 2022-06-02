## Introduction
DeepPrime is a deep-learning-based prime editing efficiency prediction tool developed in [Laboratory of Genome Editing, Yonsei University](https://sites.google.com/site/hyongbumkimlab/home). It is a successor to [DeepPE](https://www.nature.com/articles/s41587-020-0677-y), which was developed to predict prime editing efficiencies of a limited number of length combinations.

## What's New
* DeepPrime is designed to predict the efficiency of a wide variety of pegRNA combinations.
* We integrated a convolutional neural network with GRU to extract inter-sequence features between target DNA and corresponding pegRNA.
* The model was trained using 259K pegRNAs with PBS lengths ranging from 1 to 17, RT lengths ranging from 1 to 50, Edit positions ranging from 1 to 30, and editing lengths ranging from 1 to 3.

## Progress
* DeepPrime for pegRNA-based on-target prime editing efficiency prediction on HEK293T cell line is now available (`models/ontarget/final/`, `plots/ontarget/`, `results/ontarget/`).
* DeepPE performance on various external data were released (`data/`).
* DeepPrime for different cell lines and different PE techniques is now available (`models/ontarget_variants/`, `plots/ontarget_variants/`, `results/ontarget_variants/`).
* DeepPrime to predict off-target edit rates is now available (`models/offtarget/`, `plots/offtarget/`, `results/offtarget/`).

## Directory tree
```bash
├── data # PUT DATA HERE
│   ├── genes # folder for preprocessed gene files (.npy)
│   │   └── ...
│   ├── DeepPrime_dataset_final_Feat8.csv # main train/test dataset.
│   ├── DeepPrime_Nat_Liu_endo_PE2only_TriAve_220303.csv # external evaluation data.
│   ├── ...
├── models
│   ├── offtarget
│   ├── ontarget
│   └── ontarget_variants
├── plots
│   ├── offtarget
│   ├── ontarget
│   └── ontarget_variants
├── results
│   ├── offtarget
│   ├── ontarget
│   └── ontarget_variants
│
│   # BASE MODEL
├── model.py
├── utils.py
├── plot.py
├── train.py # full-dataset train without cross validation.
├── test.py
├── validate.py # train code with 5-fold cv.
│
│   # FINETUNING (VARIANTS)
├── finetune.py
├── finetune_validate.py
├── finetune_test.py
│
│   # OFF-TARGET MODEL
├── offtarget_train.py
├── offtarget_validate.py
└── offtarget_test.py
```

## Performance
DeepPrime on/off-target performance is now available in `plots/`.