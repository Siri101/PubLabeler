import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
import os
import sys
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve
import transformers
from logzero import logger
import argparse
from src.torch_utils import PubClassifier, FocalLoss
from src.data_utils import get_data, encode_text, get_dataloader, get_label_cooccurrence
transformers.logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='train', help='training or testing')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory must contain test.csv')
    parser.add_argument('--model_dir', type=str, default='./models', help='model directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='output directory')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', help='model name')
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--device', type=int, default=0, help='cuda device')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--init_weight', type=int, default=1, help='initialize weights by label co-occurrence')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size')
    parser.add_argument('--focal_loss_gamma', type=float, default=2, help='focal loss gamma')
    parser.add_argument('--best_model_idx', type=int, default=3, help='epoch with the best validation f1 score')

    args = parser.parse_args()
    label_list = ['sequences','expression','subcellular location','function','pathology & biotech','ptm/processing','names','family & domains','interaction','structure','unclassified']

    # Load data
    if args.model_type == 'train':
        train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
        train_pairs, train_texts, train_labels = get_data(train_df)
        val_df = pd.read_csv(os.path.join(args.data_dir, 'val.csv'))
        val_pairs, val_texts, val_labels = get_data(val_df)
    test_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    test_pairs, test_texts, test_labels = get_data(test_df)

    # Get label co-occurrence
    label_cooccurrence = None
    if args.init_weight and args.model_type == 'train':
        label_cooccurrence = get_label_cooccurrence(train_df, train_df.columns[2:])
        logger.info('Label co-occurrence: {}'.format(label_cooccurrence.shape))

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    config = BertConfig.from_pretrained(args.model_name)

    # Tokenize and encode the data
    if args.model_type == 'train':
        train_input_ids, train_attention_masks, train_labels = encode_text(train_texts, train_labels, tokenizer, args.max_seq_length)
        val_input_ids, val_attention_masks, val_labels = encode_text(val_texts, val_labels, tokenizer, args.max_seq_length)
    test_input_ids, test_attention_masks, test_labels = encode_text(test_texts, test_labels, tokenizer, args.max_seq_length)
    logger.info('Data encoded')

    # Create datasets and dataloaders
    if args.model_type == 'train':
        train_dataloader = get_dataloader(train_input_ids, train_attention_masks, train_labels, args.batch_size)
        val_dataloader = get_dataloader(val_input_ids, val_attention_masks, val_labels, args.batch_size)
    test_dataloader = get_dataloader(test_input_ids, test_attention_masks, test_labels, shuffle=False, batch_size=args.batch_size)

    # Create model
    model = PubClassifier(num_labels=test_labels.shape[1], hidden_dim=args.hidden_dim, init_weights=args.init_weight, label_cooccurrence=label_cooccurrence)
    device = torch.device('cuda:{}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    logger.info('Model created')

    # Train model
    if args.model_type == 'train':
        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = FocalLoss(gamma=args.focal_loss_gamma)
        best_val_f1 = 0
        best_val_epoch = 0
        for epoch in range(args.num_epochs):
            model.train()
            train_loss = 0
            for batch in train_dataloader:
                input_ids, attention_masks, labels = batch
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            logger.info('Epoch {}/{} - Train loss: {:.4f}'.format(epoch+1, args.num_epochs, train_loss))

            # Evaluate model on validation set
            model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            val_thresholds = []
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_masks, labels = batch
                    input_ids = input_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    labels = labels.to(device)

                    outputs = model(input_ids, attention_masks)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    val_preds.append(outputs.cpu().numpy())
                    val_true.append(labels.cpu().numpy())

            val_loss /= len(val_dataloader)
            val_preds = np.concatenate(val_preds, axis=0)
            val_true = np.concatenate(val_true, axis=0)
            val_f1 = f1_score(val_true, val_preds, average='micro')
            # val_precision, val_recall, _ = precision_recall_curve(val_true, np.max(val_preds, axis=1))
            # val_auc = auc(val_recall, val_precision)
            val_f1_macro = []
            for i, label in enumerate(train_df.columns[2:]):
                val_fpr, val_tpr, _ = roc_curve(val_true[:, i], val_preds[:, i])
                val_auc = auc(val_fpr, val_tpr)
                # logger.info(f"Label {i}: AUC = {val_auc:.2f}")
                # plt.plot(val_fpr, val_tpr, label=f"Label {i} (AUC = {val_auc:.2f})")
                val_precision, val_recall, thresholds = precision_recall_curve(val_true[:, i], val_preds[:, i])
                val_aupr = auc(val_recall, val_precision)
                logger.info(f"Label {label}: AUC = {val_auc:.3f}, AUPR = {val_aupr:.3f}")
                best_threshold = -100
                best_f1 = -100
                for threshold in thresholds:
                    pred = (val_preds[:, i] > threshold).astype(int)
                    f1 = f1_score(val_true[:, i], pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                val_f1_macro.append(best_f1)
                val_thresholds.append(best_threshold)
                logger.info(f"Label {label}: Best F1 = {best_f1:.3f}, Best Threshold = {best_threshold:.3f}")
            logger.info('Epoch {}/{} - Val loss: {:.4f} - Val micro F1: {:.4f} - Val macro F1: {:.4f}'.format(epoch+1, args.num_epochs, val_loss, val_f1, np.mean(val_f1_macro)))
            if val_f1+np.mean(val_f1_macro) > 2 *best_val_f1:
                best_val_f1 = val_f1+np.mean(val_f1_macro)
                best_val_epoch = epoch+1
                torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_{epoch+1}.pt'))
                pickle.dump(val_thresholds, open(os.path.join(args.output_dir, f'best_thresholds.pkl'), 'wb'))

    # Evaluate model on test set
    if args.model_type == 'train':
        model.load_state_dict(torch.load(os.path.join(args.model_dir, f'model_{best_val_epoch}.pth')))
    else:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, f'model_{args.best_model_idx}.pth')))
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_masks)
            test_preds.append(outputs.cpu().numpy())
            test_true.append(labels.cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_true = np.concatenate(test_true, axis=0)
    if os.path.exists(os.path.join(args.output_dir, f'best_thresholds.pkl')):
        best_thresholds = pickle.load(open(os.path.join(args.output_dir, f'best_thresholds.pkl'), 'rb'))
    else:
        best_thresholds = [0.622, 0.334, 0.476, 0.472, 0.434, 0.402, 0.285, 0.327, 0.418, 0.281, 0.502]
    for i in range(test_preds.shape[1]):
        test_preds[:, i] = (test_preds[:, i] > best_thresholds[i]).astype(int)
    assert len(test_pairs)==test_preds.shape[0]
    with open(os.path.join(args.output_dir, 'test_predictions.txt'), 'w') as fout:
        for i, (protein_id, paper_id) in enumerate(test_pairs):
            for j, label in enumerate(label_list):
                if test_preds[i, j] == 1:
                    print(protein_id, paper_id, label, sep='\t', file=fout)


if __name__ == '__main__':
    main()
