## Predicting the efficiency of CRISPR-based prime editing.

### Currently working on:
* Improving [DeepPE](http://deepcrispr.info/DeepPE/) using a Conv2D-RNN-based model.
* Evaluating DeepPE performance on various datasets.
### Directory tree
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
### Current model performance
Check out in `plots/`.