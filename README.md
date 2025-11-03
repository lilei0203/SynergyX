# SynergyX: a Multi-Modality Mutual Attention Network for interpretable drug synergy prediction

## Introduction
In the "Comparison between different types of models" experiment of the study "A Review of Deep Learning Approaches for Drug Synergy Prediction in Cancer", SynergyX is retrained using the DrugComb dataset, and the hyperparameters are readjusted to ensure a fair comparison under consistent experimental conditions:

batchsize=32
learning rate=0.0001
epoch=500


## Overview
The repository is organised as follows:
- `data/` contains data files and data processing files;
- `dataset/` contains the necessary files for creating the dataset;
- `models/` contains different modules of SynergyX;
- `saved_model/` contains the trained weights of SynergyX;
- `experiment/` contains log files and output files;
- `utils.py` contains the necessary processing subroutines;
- `metrics.py` contains the necessary functions to calculate model evaluation metrics;
- `main.py` main function for SynergyX.


## Requirements
The SynergyX network is built using PyTorch and PyTorch Geometric. You can use following commands to create conda env with related dependencies.

```
conda create -n synergyx python=3.8 pip=22.1.2
conda activate synergyx
pip install torch==1.10.1+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.2
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install pandas
pip install subword-nmt
pip install rdkit
```

## Implementation
### Model Training

Run the following commands to train SynergyX. 

``` 
python main.py --mode train  > './experiment/'$(date +'%Y%m%d_%H%M').log
``` 




