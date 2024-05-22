# PubLabeler

## Introduction

PubLabeler is a tool for automatically labeling scientific papers based on their abstracts and titles. It is designed to help researchers to quickly and accurately classify papers into different categories according to their research topics related to proteins. The tool uses a machine learning algorithm to train a model based on a large dataset of labeled papers, and then predicts the category of new papers based on their abstracts and titles.

## Usage

To use PubLabeler, you need to follow these steps:

1. Install the required packages:

```
pip install pandas
pip install numpy   
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers
```

2. prepare the dataset:

The data directory must contains the test.csv file. The test.csv file should contain the following columns:

- pid: the UniProtKB ID of the protein
- pmid: the PubMed ID of the paper
- labels: optional, the manually labeled categories of the paper

The data directory should also contain the abstracts and titles of the papers in the following format:

- paper_dict.pkl: a dictionary containing the abstracts and titles of the papers, i.e., {pmid: {'title': title, 'abstract': abstract}}
- pro_name.pkl: a dictionary containing the textual descriptinos of the proteins extracted from the UniProtKB database, i.e., {pid: name}

3. run the code:

- To train the model:

```
python main.py
```

- To predict the categories of new protein-paper pairs:

```
python main.py --model_type test --best_model_idx 3 --init_weight 0
```
