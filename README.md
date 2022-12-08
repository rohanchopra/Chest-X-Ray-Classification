# COMP6721_Group_Q
## Lung Disease Classification
Chest X-rays scans are among the most accessible ways to diagnose lung diseases. This study tries to compare the detection of lung diseases using these scans from  three different datasets using deep neural networks. Three different backbone architectures, ResNet34, MobileNet V3 Large and EfficientNet B1 were used along with a set of models trained using transfer learning. It is observed that MobileNet takes the least amount of time to train while ResNet converges the fastest. Also, EfficientNet performs the best most of the times on Chest X-ray scans. An F1 score of 0.8 for the pneumonia dataset was obtained, 0.98 for the COVID-19 dataset and 0.46 for the multilabel chest X-ray 8 dataset. Finally, models are visualized using t-SNE and gradCAM to understand the features learned by the models and correlate them with the actual effect of the diseases on the lungs.

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

## Training and validating the models
To train and validate the models
1. Create the conda environment
2. Copy the data to the data folder
3. Run the relevant script from the notebooks folder. For example to train the EfficientNet B1 model for the Chest X-ray 8 dataset, run the notebooks/xray8/Multilabel_Training_efficient_net_b1_100_epoch_32_batch.ipynb notebook.

## Running the pre-trained model on the sample dataset
To train and validate the models
1. Create the conda environment
2. Copy the sample data to the data folder
3. Run the relevant script from the notebooks folder. For example to train the transfer learning EfficientNet B1 model for the Chest X-ray 8 dataset, run the notebooks/xray8/Multilabel_Training_efficient_net_b1_100_epoch_32_batch_transfer.ipynb notebook.

## Source code package in PyTorch
Not required. ALl dependencies are present in the environment file.

## Dataset
A sample of the dataset can be downloaded from the following [link](https://drive.google.com/file/d/1OpvSkIDzOlJUSCLJ-wfXSVsBbfhntun4/view?usp=sharing).
