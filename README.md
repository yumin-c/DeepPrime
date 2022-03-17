## Introduction
DeepPE2 is a deep-learning-based prime editing efficiency prediction tool developed in [Laboratory of Genome Editing, Yonsei University](https://sites.google.com/site/hyongbumkimlab/home). It is a successor to [DeepPE](https://www.nature.com/articles/s41587-020-0677-y), which was developed to predict prime editing efficiencies of a limited number of length combinations.

## What's New
* DeepPE2 is designed to predict the efficiency of a wide variety of pegRNA combinations.
* We integrated a convolutional neural network with GRU to extract inter-sequence features between target DNA and corresponding pegRNA.
* The model was trained using 259K pegRNAs with PBS lengths ranging from 1 to 17, RT lengths ranging from 1 to 50, Edit positions ranging from 1 to 30, and editing lengths ranging from 1 to 3.

## Progress
* DeepPE2 for pegRNA-based on-target prime editing efficiency prediction on HEK293T cell line is available (`DeepPE.py`).
* DeepPE performance on various external data was released (`data/`).
* Currently developing DeepPE for different cell lines and different PE techniques.
* Off-target edit probability prediction model will be developed.

## Directory tree
```bash
├── data # PUT DATA HERE
│   ├── genes # folder for preprocessed gene files (.npy)
│   ├── Biofeature_output_Profiling_220205_PE_effi_for_CYM.csv # PE profiling dataset with a small diversity of target genes.
│   ├── DeepPrime_input_PEmax_220228.csv
│   ├── DeepPrime_Nat_Liu_endo_PE2only_220303.csv
│   ├── ...
│   ├── DeepPrime_PECV__train_220214.csv # PECV train dataset.
│   ├── DeepPrime_PECV__test_220214.csv  # PECV test dataset.
│   ├── g_pf.npy
│   └── g_train.npy
├── models
│   ├── test # models for performance evaluation.
│   └── pretrained
├── plots
├── results
├── DeepPE.py # main train file with 5-fold cv.
├── DeepPE_finalize.py # final train code without cross-validation.
├── plot.py # code for plotting.
└── test.py # code for testing.
```

## Performance
DeepPE2 on-target performance on HEK293T cell line is now available in `plots/`.
