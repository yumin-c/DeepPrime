## Predicting the efficiency of CRISPR-based prime editing.

### Currently working on:
* Improving [DeepPE](http://deepcrispr.info/DeepPE/) using a Conv2D-RNN-based model.
* Fine-tuning using a biased external data that tested various pegRNA combinations
```bash
├── DeepPE.py # main file including train, valid, test code.
├── DeepPE_3outputs.py # experimental code with three separate outputs.
├── data
│   ├── Biofeature_output_Profiling_220205_PE_effi_for_CYM.csv # PE profiling dataset with a small diversity of target genes.
│   ├── DeepPrime_PECV__test_220214.csv # PECV test dataset.
│   └── DeepPrime_PECV__train_220214.csv # PECV train dataset.
├── finetune.py # code for fine-tuning.
├── plot.py # code for plotting.
├── lightning
│   ├── DeepPE_sweep_lightning.py
│   └── PL_DeepPE.py
```
### Current model performance
**DeepPE2 performance plot**
![spearmanPlot](./plots/Evaluation%20of%20DeepPE2.jpg)

**Prediction for substitution only**
![subPlot](./plots/Evaluation%20of%20substitution.jpg)

**Prediction for insertion only**
![insPlot](./plots/Evaluation%20of%20insertion.jpg)

**Prediction for deletion only**
![delPlot](./plots/Evaluation%20of%20deletion.jpg)
