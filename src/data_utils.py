import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
import os
import pandas as pd
from IPython.display import display
from torch.utils.data import TensorDataset, DataLoader
import transformers
transformers.logging.set_verbosity_error()


def get_data(meta_data, paper_path='./data/paper_dict.pkl', protein_path='./data/pro_name.pkl'):
    paper_dict = pickle.load(open(paper_path, 'rb'))
    pro_name = pickle.load(open(protein_path, 'rb'))
    text_list = []
    label_list = []
    pair_list = []
    for i in range(len(meta_data)):
        paper_id = str(meta_data.iloc[i]['pmid'])
        pro_id = str(meta_data.iloc[i]['pid'])
        label = meta_data.iloc[i][2:].values
        try:
            text = (pro_name[pro_id], paper_dict[paper_id]['title'] + ' ' + paper_dict[paper_id]['abstract'])
        except KeyError:
            continue
        pair_list.append((pro_id, paper_id))
        text_list.append(text)
        label_list.append(label)
    return pair_list, text_list, label_list

def encode_text(texts, labels, tokenizer, max_length=512):
    input_ids = []
    attention_masks = []
    for text_pair in texts:
        encoded_dict = tokenizer.encode_plus(
            text=text_pair[0],
            text_pair=text_pair[1],
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    labels = torch.tensor(labels, dtype=torch.float32)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks, labels


def get_dataloader(input_ids, attention_masks, labels, shuffle=True, batch_size=32):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_label_cooccurrence(meta_data, label_list):
    labels = meta_data[label_list].values
    # 计算标签的共现频率矩阵
    cooccurrence_matrix = np.dot(labels.T, labels)

    # 计算每个标签出现的总次数
    label_counts = np.sum(labels, axis=0)

    # 计算条件概率矩阵
    cond_prob_matrix = cooccurrence_matrix / label_counts[:, np.newaxis]
    return cond_prob_matrix