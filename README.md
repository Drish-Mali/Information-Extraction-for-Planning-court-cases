# Information Extraction for Planning Court Cases

## Introduction
This project focuses on extracting key information from planning court cases using machine learning techniques, including Named Entity Recognition (NER) and multi-label classification. The goal is to analyze legal documents and automatically identify relevant entities and labels within the text.

## Installation
To install the required dependencies, run the following command:

```
python -m pip install -r requirements.txt
```
## Structure 

├── README.md               # Project overview and instructions
├── LICENSE                 # License for the project
├── requirements.txt        # List of dependencies
├── .gitignore              # Files to ignore in Git
├── data                    # Directory for data files
├── src                     # Source code
│   ├── data                # Scripts to download, generate, or process data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── models              # Scripts to train models and make predictions
│   │   ├── __init__.py
│   │   └── train_model.py
│   └── notebooks           # Jupyter notebooks
│       ├── __init__.py
│       └── helpers.py
├── models                  # Trained models or model checkpoints
