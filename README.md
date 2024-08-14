# Information Extraction for Planning Court Cases

## Introduction
This project aims to extract key information from planning court cases using advanced machine learning techniques, specifically Named Entity Recognition (NER) and multi-label classification. The objective is to automatically identify relevant entities and labels within legal documents, streamlining the analysis process. This work is part of a Master of Science (MSc) thesis in Data Science Technology and Innovation (DSTI) at the EDINA research centre, University of Edinburgh. 

## Installation
To install the required dependencies, run the following command:

```
python -m pip install -r requirements.txt
```
## Structure 
```
├── README.md                                   # Project overview and instructions
├── LICENSE                                     # License for the project
├── requirements.txt                            # List of dependencies
├── .gitignore                                  # Files to ignore in Git
├── data                                        # Directory for data files
├── src                                         # Source code
│   ├── data                                    # Scripts to download, generate, or process data
│   │   ├── data_scrap.py                       # Script to scrap cases meta data
│   │   └── download_pdf.py                     # Script to download pdf files for cases
│   │   ├── download_xml.py                     # Script to download XML files for cases
│   │   └── create_paragraph_annotation_data.py # Script to create case wise data for paragraph annotation
│   │   ├── make_dataset.py                     # Script to divide XML file into cover and main section
│   │   └── make_dataset_new_main_text.py       # Script to divide main section using new paragraph seperation technique
│   │   ├── llama3_prompt_citation_label.py     # Script to label citation using Llama 3 70B model
│   │   └── open_ai_prompt_fact_label.py        # Script to label fact using ChatGPT 3.5
│   ├── model                                   # Scripts to train models and make predictions
│   │   ├── multi_label_training.py             # Script to train multi-label classification models
│   │   └── NER_training.py                     # Script to train NER models
│   └── notebooks                               # Jupyter notebooks
│       ├── EDA_data.ipynb                      # Notebook to perform EDA on the data
│       └── consolidate_data.ipynb              # Notebook to consolidate multi-label data 
│       └── consolidate_re_annotated_data.ipynb # Notebook to consolidate re-annotated multi-label data 
├── models                                      # Trained models or model checkpoints