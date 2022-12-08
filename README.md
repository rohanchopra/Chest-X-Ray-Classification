# COMP6721_Group_Q
## Lung Disease Classification
Detect the lung disease from chest X-Ray scans using CNN and deep learning.

## Directory Structure

```
├── documentation/              <- All project related documentation and reports
├── deliverables/               <- All project deliverables
├── data/                       <- All project related data files
├── latex/                      <- All latex files for the project
├── models/                     <- Trained models, predictions, and summaries
├── notebooks/                  <- Jupyter notebooks
│  ├── Pneumonia/               <- Notebooks for the pneumonia dataset
│  ├── covid-pneumonia/         <- Notebooks for the COVID dataset
│  ├── xray8/                   <- Notebooks for the Chest X-ray 8 dataset
│  ├── Grad-CAM/                <- Notebooks to generate gradCAM plots
│  ├── t-SNE/                   <- Notebooks to generate t-SNE plots
├── src/                        <- Source code for the project
│  ├── multilabel/              <- Scripts for the multilabel dataset
│  ├── __init__.py              <- Makes src a Python module
├── .gitignore                  <- List of files and folders git should ignore
├── LICENSE                     <- Project's License
├── README.md                   <- The top-level README for developers using this project
├── environment.yml             <- Conda environment file
└── requirements.txt            <- The requirements file for reproducing the environment
```

## Creating the environment
Load conda environment as:
```
conda env create -f environment.yml
```
Install torch in conda environment:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

