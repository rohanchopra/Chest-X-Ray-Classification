# Lung Disease Classification
Detect the lung disease from chest X-Ray scans using CNN and deep learning.

## Directory Structure

```
├── documentation/              <- All project related documentation and reports
├── config/                     <- All project related config files
│  ├── config.ini               <- Configuration for the testing/production system
├── data/                       <- All project related data files
│  ├── raw/                     <- The original, immutable data dump
│  └── processed/               <- The final data sets for modeling
├── logs/                       <- Project logs
├── models/                     <- Trained models, predictions, and summaries
├── notebooks/                  <- Jupyter notebooks
├── references/                 <- Data dictionaries, and all other explanatory material
├── src/                        <- Source code for the project
│  ├── modeling/                <- Scripts to train the models
│  ├── processing/              <- Scripts to process the data to create the final dataset
│  ├── scoring/                 <- Scripts to score the models
│  ├── utils/                   <- Scripts for utility functions
│  │  └── utils.py              <- Script for common utility functions
│  ├── __init__.py              <- Makes src a Python module
│  └── main.py                  <- Project's entry script
├── .gitignore                  <- List of files and folders git should ignore
├── LICENSE                     <- Project's License
├── README.md                   <- The top-level README for developers using this project
└── requirements.txt            <- The requirements file for reproducing the environment
```

## Naming Conventions

### Jupyter Notebooks
```
yyyyMMdd-developer_initials-description_of_notebook.ipynb
```

### Log Files
```
yyyy-MM-ddTHHmm.log
```

### Model Files
```
yyyyMMdd-description_of_model.h5
```

### Data Files
```
yyyyMMdd-description_of_data.h5
```