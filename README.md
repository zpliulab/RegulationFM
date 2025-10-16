# RegulationFM: A Foundation Model for Generating Gene Regulatory Networks from Single-Cell Multi-Omics Data # 

RegulationFM is a foundation model for accurate and interpretable Gene Regulatory Network reconstruction directly from single-cell multi-omics data. It integrates scRNA-seq, scATAC-seq, and other omics to learn joint representations, then uses a discrete diffusion generative module to generate GRNs at scale.

> Abbreviations used:  
> GRN - Gene Regulatory Network  
> scRNA-seq - single-cell RNA sequencing  
> scATAC-seq - single-cell Assay for Transposase-Accessible Chromatin


<div align="center">
  <img src="https://github.com/zpliulab/RegulationFM/blob/main/images/images1.jpg" alt="Schematic diagram of RegulationFM generation network" style="width: 1500px; height: 200px;"/>
</div>

## Requirements ![Python](https://img.shields.io/badge/python-3.10-blue "Python3.10")

Before you start, ensure the following packages are installed.

Key packages
- python 3.10.14
- torch 1.13.1
- torch-geometric 2.5.3
- r-base 4.3.3

Other packages
- networkx
- joblib
- numpy
- pandas
- scikit-learn
- scipy
- r-cicero


You can create a minimal environment with:

```bash
conda create -n RegulationFM python=3.10 -y
conda activate RegulationFM
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
## Project Structure

```bash
RegulationGPT/
│
├── config/                         # Configuration files
│   ├── Consctructed_base_GRN_config.py   # Parameters for building baseline GRN
│   ├── pathway_config.py                 # Options for graph segmentation into subgraphs
│   ├── RegulationGPT_config.py           # MoDGM training and testing parameters
│   ├── Multiomics_integration_config.py  # MoFRM multimodal integration parameters
│
├── diffusion_model/                # Core model code
│   ├── __init__.py
│   ├── discrete_diffusion_model.py       # Discrete diffusion model
│   ├── RegulationGPT.py                  # MoDGM module
│   ├── Building_training_dataset/        # Utilities for constructing train and test sets
│   │   ├── pathway_segmentation.py
│   │   ├── Standardized_train_test_sets.py
│   └── discrete/
│       ├── diffusion_utils.py
│       ├── network_preprocess.py
│       ├── noise_predefined.py
│
├── multiomics_integration/         # MoFRM - multi-omics joint learning
│   ├── __init__.py
│   ├── Multiomics_joint.py
│   ├── Multiomics_process.py
│   └── VAE_attention.py
│
├── scATAC_co_accessibility/        # scATAC-seq co-accessibility to baseline GRN
│   ├── __init__.py
│   ├── motif_analysis.rar          # Motif databases
│   ├── step1_preprocess_atac2peak.R # Note: The R environment requires additional configuration!
│   └── Preprocess_scATAC_to_GRN.py
│
├── Data/                           # Data utilities and examples
│   ├── datatype_trans.py
│
├── results_checkpoint/             # Saved model weights and results
│
├── run_RegulationFM.py             # Main script for training and testing on simulation data
├── run_RegulationFM_MI.py          # Example script for myocardial infarction case
└── README.md
```
## Data Preparation 

### Download the simulation datasets

The simulation datasets used to reproduce our results is hosted on Zenodo:

- Record: https://doi.org/10.5281/zenodo.17367292 
- Contents: pre-split training and test sets plus the gold-standard network
- 
### Other datasets

1. Standardize multi-omics data into an HDF5 file named `standard_input.h5`.  
   It should contain at least the datasets `mRNA` and `Gene_id`.  
2. Process scATAC-seq with Signac or Cicero to generate gene activity matrices.  
3. See:
   - `scATAC_co_accessibility/step1_preprocess_atac2peak.R`
   - `Data/datatype_trans.py`  
   for example transformations and file formats.

- If you have any other questions, please contact zpliu@sdu.edu.cn or cywang@xzhmu.edu.cn


## Quick Start

<div align="center">
  <img src="https://github.com/zpliulab/RegulationFM/blob/main/images/network.gif" alt="Schematic diagram of RegulationFM generation network" style="width: 500px; height: 500px;"/>
</div>

### A. Train the model

Use `run_RegulationFM.py` as a reference.

1. Configure parameters in `config/RegulationGPT_config.py`.
2. (Optional) Build a baseline GRN from scATAC-seq co-accessibility.
3. (Optional) Segment large networks into subgraphs.
4. Run multi-omics integration `Multiomics_learning` to produce joint representations.
5. Initialize the trainer:

```python
from diffusion_model import RegulationGPT
trainer = RegulationGPT(args)
```

6. Train:

```python
best_mean_AUC, train_model_file, printf = trainer.train(train_filename, base_GRN_link)
```

`train_model_file` is a checkpoint path saved under `results_checkpoint`.

### B. Test the model

1. Load a checkpoint:

```python
import torch
diffusion_pre = torch.load(train_model_file, map_location=trainer.device)
```

2. Load test data. If the HDF5 file contains a single dataset, set `num='default'`.  
   `network_key_list` controls which gold standards to evaluate against  
   supported options include `network` RegNetwork, `DoRothEA`, `KEGG`, `TRRUST`.

```python
testdata, truelabel = trainer.load_test_data(test_filename, num=test_num, network_key_list='network')
```

3. Run inference:

```python
adj_final = trainer.test(diffusion_pre, testdata, truelabel)
```

4. Evaluate:

```python
perf = trainer.evaluation(adj_final, truelabel)
print(perf['AUC'], perf['AUPR'], perf['F1'])
```

## Configuration Tips

- `config/RegulationGPT_config.py` controls training epochs, batch sizes, graph size limits, intervals for testing and checkpointing.
- `config/pathway_config.py` controls subgraph segmentation such as metacell number and pathway options.
- `config/Multiomics_integration_config.py` sets hyperparameters for the multi-omics fusion modules.
- `scATAC_co_accessibility/*` provides scripts and databases to construct a baseline GRN that can guide evaluation or serve as priors.

## Citation

The manuscript is currently in preparation.  
For questions please contact: zpliu@sdu.edu.cn.

## License
RegulationFM is released under an MIT License.
