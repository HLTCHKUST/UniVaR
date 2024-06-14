import os, sys
sys.path.append('../univar_trainer')

import argparse
import glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import evaluate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from training_utils.utils import encode_qas

target_label = 'value' # 'lang' or 'model'
model_path = 'FacebookAI/xlm-roberta-base'
training_type = 'knn' # 'full' or 'cls' or 'knn'

seen_value_list = [
    'aya-101$eng', 'aya-101$fra', 'aya-101$arb', 'aya-101$deu', 'aya-101$ita', 'aya-101$jpn',
    'aya-101$hin', 'aya-101$zho', 'aya-101$vie', 'aya-101$tur', 'aya-101$spa', 'aya-101$ind',
    'Mixtral-8x7B-Instruct-v0.1$fra', 'Mixtral-8x7B-Instruct-v0.1$deu', 'Mixtral-8x7B-Instruct-v0.1$spa', 
    'Mixtral-8x7B-Instruct-v0.1$ita', 'Mixtral-8x7B-Instruct-v0.1$eng', 'SeaLLM-7B-v2$eng', 'SeaLLM-7B-v2$zho', 
    'SeaLLM-7B-v2$vie', 'SeaLLM-7B-v2$ind', 'bloomz-rlhf$eng', 'bloomz-rlhf$zho', 'bloomz-rlhf$fra', 'bloomz-rlhf$spa',
    'bloomz-rlhf$arb', 'bloomz-rlhf$vie', 'bloomz-rlhf$hin', 'bloomz-rlhf$ind', 'chatglm3-6b$zho',
    'chatglm3-6b$eng', 'Nous-Hermes-2-Mixtral-8x7B-DPO$fra', 'Nous-Hermes-2-Mixtral-8x7B-DPO$deu',
    'Nous-Hermes-2-Mixtral-8x7B-DPO$spa', 'Nous-Hermes-2-Mixtral-8x7B-DPO$ita', 'Nous-Hermes-2-Mixtral-8x7B-DPO$eng',
    'SOLAR-10.7B-Instruct-v1.0$eng', 'Mistral-7B-Instruct-Aya-101$fra', 'Mistral-7B-Instruct-Aya-101$deu', 
    'Mistral-7B-Instruct-Aya-101$spa', 'Mistral-7B-Instruct-Aya-101$ita', 'Mistral-7B-Instruct-Aya-101$eng',
]

# Evaluation Metric
metric_acc = evaluate.load("accuracy")
metric_prec = evaluate.load("precision")
metric_rec = evaluate.load("recall")
metric_f1 = evaluate.load("f1")

@torch.inference_mode()
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    _, preds = torch.from_numpy(logits).topk(5, dim=-1)

    n_samples = labels.shape[0]
    acc_1 = ((preds[:,0] == torch.from_numpy(labels)).sum() / n_samples).item()
    acc_3 = (((preds[:,:3] == torch.from_numpy(labels).unsqueeze(dim=-1)).sum(dim=-1) > 0).sum() / n_samples).item()
    acc_5 = (((preds[:,:5] == torch.from_numpy(labels).unsqueeze(dim=-1)).sum(dim=-1) > 0).sum() / n_samples).item()

    acc = metric_acc.compute(predictions=preds[:,0], references=labels)['accuracy']    
    prec = metric_prec.compute(predictions=preds[:,0], references=labels, average='weighted')['precision']
    rec = metric_rec.compute(predictions=preds[:,0], references=labels, average='weighted')['recall']
    f1 = metric_f1.compute(predictions=preds[:,0], references=labels, average='weighted')['f1']
    return {'acc': acc, 'acc@1': acc_1, 'acc@3': acc_3, 'acc@5': acc_5, 'prec': prec, 'rec': rec, 'f1': f1}

def compute_metrics_knn(eval_pred):
    preds, labels = eval_pred
    acc = metric_acc.compute(predictions=preds, references=labels)['accuracy']
    prec = metric_prec.compute(predictions=preds, references=labels, average='weighted')['precision']
    rec = metric_rec.compute(predictions=preds, references=labels, average='weighted')['recall']
    f1 = metric_f1.compute(predictions=preds, references=labels, average='weighted')['f1']
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

if __name__ == '__main__':
    ###
    # Argument Parser
    ###
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--model_path', type=str, default='FacebookAI/xlm-roberta-base', help='Model path')
    parser.add_argument('--training_type', type=str, default='knn', help='Training type')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Cache directory')
    parser.add_argument('--ouput_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of kNN neighbors')
    parser.add_argument('--task_name', type=str, default='text-only', help='Input task representation (text-only, translation, paraphrase)')
    parser.add_argument('--num_qas', type=int, default=1, help='Number of QA in the input')
    args = parser.parse_args()

    model_path = args.model_path
    training_type = args.training_type
    cache_dir = args.cache_dir
    out_dir = args.ouput_dir
    bs = args.batch_size
    n_epochs = args.n_epochs
    lr = args.learning_rate
    n_neighbors = args.n_neighbors
    num_qas = args.num_qas
    task_name = args.task_name
    model_name = model_path.split('/')[-1]
    
    # Make dirs if not there
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Set Random Seed
    random.seed(14045)

    # Load EuroParl Dataset    
    train_df = pd.read_csv('./europarl_data/train_para.csv')
    valid_df = pd.read_csv('./europarl_data/valid_para.csv')
    test_df = pd.read_csv('./europarl_data/test_para.csv')
    dset_name = 'europarl'

    # Perform Evaluation
    if os.path.exists(f'{out_dir}/{training_type}_{dset_name}_{task_name}_{model_name}_{num_qas}_result_log.csv'):
        print(f'Skipping {training_type}_{dset_name}_{task_name}_{model_name}')
        sys.exit(0)

    label = None
    target_label = 'original_language'
    if 'text-only' in task_name:
        label = 'text'
    elif 'translation' in task_name:
        label = 'qa'
    elif 'paraphrase' in task_name:
        label = 'para_qa'
    else:
        raise ValueError(f'Unknown task name: {task_name}')
        
    # Encode QAs and prepare labels
    train_qas = train_df[label].tolist()
    valid_qas = valid_df[label].tolist()
    test_qas = test_df[label].tolist()

    # Encode QAs
    cache_path  = f'{cache_dir}/{task_name}_{dset_name}_{model_name}_{num_qas}.pt'  
    if os.path.exists(cache_path):
        train_reps, valid_reps, test_reps = torch.load(cache_path)
    else:
        model = SentenceTransformer(model_path, trust_remote_code=True).to('cuda')
        with torch.inference_mode():
            enc_bs = 6 if 'nomic' in model_path else 512
            train_reps = encode_qas(train_qas, model, qa_per_view=num_qas, convert_to_numpy=True, batch_size=enc_bs, show_progress_bar=True)
            valid_reps = encode_qas(valid_qas, model, qa_per_view=num_qas, convert_to_numpy=True, batch_size=enc_bs, show_progress_bar=True)
            test_reps = encode_qas(test_qas, model, qa_per_view=num_qas, convert_to_numpy=True, batch_size=enc_bs, show_progress_bar=True)
            torch.save((train_reps, valid_reps, test_reps), cache_path)
        del model

    # Prepare Labels
    label_to_idx = {x:i for i,x in enumerate(sorted(list(train_df[target_label].unique())))}
    train_labels = list(map(lambda x: label_to_idx[x], train_df[target_label].tolist()))
    test_labels = list(map(lambda x: label_to_idx[x], test_df[target_label].tolist()))
    num_labels = train_df[target_label].nunique()
    
    if training_type == 'cls':
        data = {
            'model': [], 'epoch': [], 
            'train_loss': [], 'test_loss': [],
            'test_acc@1': [], 'test_acc@3': [], 'test_acc@5': [],
            'test_prec': [], 'test_rec': [], 'test_f1': [],
        }
        
        # Define the model
        model = nn.Linear(in_features=train_reps.shape[1], out_features=num_labels).cuda()
        criterion = nn.CrossEntropyLoss()
        
        # Define your optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Define your learning rate scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - (epoch / n_epochs))
    
        # Define your data loader
        train_loader = DataLoader(list(zip(train_reps, train_labels)), batch_size=bs, shuffle=True, num_workers=128)
        test_loader = DataLoader(list(zip(test_reps, test_labels)), batch_size=bs, shuffle=False, num_workers=128)
        
        # Train the model for multiple epochs
        for epoch in range(n_epochs):
            # training the model on the training set
            model.train()
            train_loss = 0.0
            for reps, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} training"):
                # Move to CUDA
                reps, labels = reps.to('cuda'), labels.to('cuda')
                
                # Zero the gradients
                optimizer.zero_grad()
        
                # Forward Backward
                outputs = model(reps)
                loss = criterion(outputs, labels)
                loss.backward()
                
                train_loss += loss.item()
                
                # Update the weights
                optimizer.step()
            
            # Print the loss after each epoch
            train_loss /= len(train_loader)
            print(f"Epoch {epoch + 1} | training loss: {train_loss:.4f}")
        
            # Update the learning rate
            scheduler.step()

            # Evaluate the model on the test set
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_logits, test_labels = [], []
                for reps, labels in test_loader:
                    # Move to CUDA
                    reps, labels = reps.to('cuda'), labels.to('cuda')
                    
                    # Forward pass
                    outputs = model(reps)
        
                    # Compute the loss
                    test_loss += criterion(outputs, labels).item()
            
                    # Store the prediction & label
                    logits = outputs.data.squeeze().cpu().numpy()
                    test_logits.append(logits)
                    test_labels.append(labels.squeeze().cpu().numpy())
                    
                # Print the test loss and accuracy
                test_metrics = compute_metrics(eval_pred=(np.concatenate(test_logits, axis=0), np.concatenate(test_labels, axis=0)))
                test_loss /= len(test_loader)
                test_acc, test_f1 = test_metrics['acc'] * 100, test_metrics['f1'] * 100
                print(f"Epoch {epoch + 1} | test loss: {test_loss:.4f}, acc: {test_acc:.2f}%, f1: {test_f1:.2f}%")
    
            # Add log
            data['model'].append(model_path)
            data['epoch'].append(epoch)
            data['train_loss'].append(train_loss)
            data['test_loss'].append(test_loss)
            data['test_acc@1'].append(test_metrics['acc@1'])
            data['test_acc@3'].append(test_metrics['acc@3'])
            data['test_acc@5'].append(test_metrics['acc@5'])
            data['test_prec'].append(test_metrics['prec'])
            data['test_rec'].append(test_metrics['rec'])
            data['test_f1'].append(test_metrics['f1'])
        
        # Log Dataframe
        log_df = pd.DataFrame(data)
        log_df.to_csv(f'{out_dir}/cls_{dset_name}_{task_name}_{model_name}_{num_qas}_result_log.csv', index=False)
        print(log_df.to_dict(orient='records'))

        # Save Prediction & Label
        torch.save(
            (np.concatenate(test_logits, axis=0), np.concatenate(test_labels, axis=0)), 
            f'{out_dir}/cls_{dset_name}_{task_name}_{model_name}_{num_qas}_pred_labels.pt'
        )
    else: # training_type == 'knn'
        data = {
            'model': [], 
            'test_acc': [], 'test_prec': [], 'test_rec': [], 'test_f1': [],
        }
        
        # Run kNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_reps, train_labels)
        test_preds = knn.predict(test_reps)
        test_metrics = compute_metrics_knn(eval_pred=(test_preds, test_labels))
        
        data['model'].append(model_path)
        data['test_acc'].append(test_metrics['acc'])
        data['test_prec'].append(test_metrics['prec'])
        data['test_rec'].append(test_metrics['rec'])
        data['test_f1'].append(test_metrics['f1'])
        
        # Log Dataframe
        log_df = pd.DataFrame(data)
        torch.save(
            (test_preds, test_labels), 
            f'{out_dir}/knn_{dset_name}_{task_name}_{model_name}_{num_qas}_pred_labels.pt'
        )
        log_df.to_csv(f'{out_dir}/knn_{dset_name}_{task_name}_{model_name}_{num_qas}_result_log.csv', index=False)
        print(log_df.to_dict(orient='records'))