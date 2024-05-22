import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
transformers.logging.set_verbosity_error()

class PubClassifier(nn.Module):
    def __init__(self, num_labels, bert_model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', hidden_dim=128, 
                 freeze_bert_layers=8, init_weights=True, label_cooccurrence=None):
        super(PubClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.freeze_bert_layers = freeze_bert_layers
        self.num_labels = num_labels
        # 冻结前 freeze_bert_layers 层的参数
        for param in self.bert_model.parameters():
            param.requires_grad = False
        for param in self.bert_model.encoder.layer[-freeze_bert_layers:].parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self.init_weights(label_cooccurrence)

    def init_weights(self, label_cooccurrence_matrix):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

        # selected_weight = self.out.weight[:self.num_labels]
        # # selected_bias = self.out.bias[:self.num_labels]
        # nn.init.xavier_uniform_(selected_weight)
        # # nn.init.constant_(selected_bias, 0)
        self.out.weight.data[:, :self.num_labels] = torch.from_numpy(label_cooccurrence_matrix)
        # self.out.bias.data[:k] = selected_bias

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        x = self.fc(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.out(x)
        return self.sigmoid(logits)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.functional.binary_cross_entropy(input, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        # self.gamma = self.gamma.to(input.device)
        

        if self.weight is not None:
            weight = self.weight.to(input.device) 
            wce_loss = nn.functional.binary_cross_entropy(input, target, weight=weight, reduction='none')
            focal_loss = (1 - p_t) ** self.gamma * wce_loss
        else:
            focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(input.device)
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
