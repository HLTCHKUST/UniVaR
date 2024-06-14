import glob
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers.readers import InputExample

from torch.utils.data import Dataset
    
# Dataset Class for ValueBERT's paired Value QA data
class PairValueQADataset(Dataset):
    def __init__(self, value_qa_data, per_value_sample_truncation, qa_per_view=1):
        if per_value_sample_truncation <= 0:
            self.value_qa_data = value_qa_data # {"value": <list["qa"]>}
        else:
            self.value_qa_data = {k: v[:per_value_sample_truncation] for k, v in value_qa_data.items()}
        
        self.value_idx_map = {i: k for i,k in enumerate(value_qa_data.keys())}
        self.qa_per_view = qa_per_view
        
        num_pair = []
        for k, v in self.value_qa_data.items():
            num_pair.append(len(v))
            print(f"\t{len(v)} samples in {k}")
        self.end_value_idx = np.array(num_pair).cumsum()
        
    def __len__(self):
        return self.end_value_idx[-1]

    def __getitem__(self, idx):
        # Get value QA data
        value_idx = np.argmax(self.end_value_idx > idx)       
        value_key = self.value_idx_map[value_idx]
        qa_data = self.value_qa_data[value_key]
        
        # Get QA index
        qa1_idx = idx - self.end_value_idx[value_idx]
        if self.qa_per_view==1:
            qa2_idx = np.random.randint(len(qa_data))
            return InputExample(texts=[qa_data[qa1_idx], qa_data[qa2_idx]], label=value_idx)
        else:
            view_1 = '\n'.join(random.sample(qa_data, self.qa_per_view))
            view_2 = '\n'.join(random.sample(qa_data,self.qa_per_view))
            
            return InputExample(texts=[view_1,view_2], label=value_idx)
        

        # return InputExample(texts=[qa_data[qa1_idx], qa_data[qa2_idx]], label=1 if 'no_value' in value_key else 0)

def split_train_val_data(samples, per_value_train_samples, per_value_val_samples):
    assert per_value_train_samples + per_value_val_samples <= len(samples)
    assert per_value_train_samples > 0 or per_value_val_samples > 0

    if per_value_train_samples <= 0:
        per_value_train_samples = len(samples) - per_value_val_samples
    if per_value_val_samples <= 0:
        per_value_val_samples = len(samples) - per_value_train_samples

    train_samples = samples[:per_value_train_samples]
    validation_samples = samples[per_value_train_samples:]

    return train_samples, validation_samples


def get_data(dataset_path, per_value_train_samples, per_value_val_samples, args):
    
    if '.csv' not in dataset_path[-8:]:
        dfs = []
        for path in sorted(glob.glob(f'{dataset_path}/*.csv*')):
            if not args.include_no_value and 'no_value' in path:
                continue
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs)
    else:
        df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['answer']).reset_index(drop=True)
    df['qa'] = df.apply(lambda x: f"Q: {x['question'].strip()} A: {x['answer'].strip()}", axis='columns')

    samples = {f'{k[0]}${k[1]}': tdf['qa'].tolist() for k, tdf in df.groupby(['model', 'lang'])}

    train_samples = {}
    validation_samples = {}
    for value_id, value_samples in samples.items():
        random.shuffle(value_samples)
        train_samples[value_id], validation_samples[value_id] = split_train_val_data(value_samples, per_value_train_samples, per_value_val_samples)

    print(f"Dataset loaded from {dataset_path}, total available samples: {len(df)}")

    print('Generating Training Split...')
    train_dataset = PairValueQADataset(train_samples, per_value_train_samples, args.qa_per_view)
    print(f" -> Train Dataset: {len(train_dataset)} samples (per value truncation={per_value_train_samples})")

    print('Generating Validation Split...')
    validation_dataset = PairValueQADataset(validation_samples, per_value_val_samples, args.qa_per_view)
    print(f" -> Validation Dataset: {len(validation_dataset)} samples (per value truncation={per_value_val_samples})")

    return train_dataset, validation_dataset

def save_analysis_data(train_dataset_path, test_dataset_path, val_split_ratio=0.1, include_no_value=False, outpath='./'):
    # Load the training dataset
    print(f"Loading dataset from {train_dataset_path}")
    if '.csv' not in train_dataset_path[-8:]:
        dfs = []
        for path in sorted(glob.glob(f'{train_dataset_path}/*.csv*')):
            if not include_no_value and 'no_value' in path:
                continue
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs)
    else:
        df = pd.read_csv(train_dataset_path)
    df = df.dropna(subset=['answer']).reset_index(drop=True)
    df['qa'] = df.apply(lambda x: f"Q: {x['question'].strip()} A: {x['answer'].strip()}", axis='columns')

    # Generate Training & Validation dataset
    samples = {f'{k[0]}${k[1]}': tdf.index.tolist() for k, tdf in df.groupby(['model', 'lang'])}
    for k, v in samples.items():
        random.shuffle(v)
        
    train_samples = {k: v[:int(len(v)*(1-val_split_ratio))] for k, v in samples.items()}
    validation_samples = {k: v[int(len(v)*(1-val_split_ratio)):] for k, v in samples.items()}

    # Get Dataframe from Index
    for k, v in train_samples.items():
        train_samples[k] = df.loc[v,:]
    for k, v in validation_samples.items():
        validation_samples[k] = df.loc[v,:]
        
    # Save Training and Validation
    train_file_name = f'seen_qa/samples_with_no_value' if include_no_value else f'seen_qa/samples_value_only'
    valid_file_name = f'unseen_qa/samples_with_no_value' if include_no_value else f'unseen_qa/samples_value_only'
    torch.save(train_samples, f'{outpath}/{train_file_name}.pt')
    torch.save(validation_samples, f'{outpath}/{valid_file_name}.pt')
    
    # Load the test datasets
    if '.csv' not in test_dataset_path[-8:]:
        dfs = []
        for path in sorted(glob.glob(f'{test_dataset_path}/*.csv*')):
            if not include_no_value and 'no_value' in path:
                continue
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs)
    else:
        df = pd.read_csv(test_dataset_path)
    df = df.dropna(subset=['answer']).reset_index(drop=True)
    df['qa'] = df.apply(lambda x: f"Q: {x['question'].strip()} A: {x['answer'].strip()}", axis='columns')

    # Generate Large Test & Small Probe Test dataset
    samples = {f'{k[0]}${k[1]}': tdf.index.tolist() for k, tdf in df.groupby(['model', 'lang'])}
    for k, v in samples.items():
        random.shuffle(v)
        
    train_samples = {k: v[:int(len(v)*(1-val_split_ratio))] for k, v in samples.items()}
    validation_samples = {k: v[int(len(v)*(1-val_split_ratio)):] for k, v in samples.items()}
    
    # Get Dataframe from Index
    for k, v in train_samples.items():
        train_samples[k] = df.loc[v,:]
    for k, v in validation_samples.items():
        validation_samples[k] = df.loc[v,:]
    
    # Save Large Test and Small Probe Test dataset
    train_file_name = f'unseen_model/samples_with_no_value' if include_no_value else f'unseen_model/samples_value_only'
    valid_file_name = f'unseen_model_qa/samples_with_no_value' if include_no_value else f'unseen_model_qa/samples_value_only'
    torch.save(train_samples, f'{outpath}/{train_file_name}.pt')
    torch.save(validation_samples, f'{outpath}/{valid_file_name}.pt')

# Inference Code
model_to_data_mapping = {
    'multi-value-embedding': [
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/seen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_model/samples_value_only.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_model_qa/samples_value_only.pt',
    ],
    'multi-value-embedding-expand': [
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/seen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_model/samples_value_only.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_model_qa/samples_value_only.pt',
    ],
    'multi-value-embedding-no-value': [
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/seen_qa/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_qa/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_model/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240314_gen_qa_pairs/analysis/unseen_model_qa/samples_with_no_value.pt',
    ],
    'multi-value-embedding-expand-no-value': [
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/seen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model_qa/samples_value_only.pt',
    ],
    'la-multi-value-embedding': [
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/seen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model_qa/samples_value_only.pt',
    ],
    'la-multi-value-embedding-expand': [
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/seen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_qa/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model/samples_value_only.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model_qa/samples_value_only.pt',
    ],
    'la-multi-value-embedding-no-value': [
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/seen_qa/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_qa/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model_qa/samples_with_no_value.pt',
    ],
    'la-multi-value-embedding-expand-no-value': [
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/seen_qa/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_qa/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model/samples_with_no_value.pt',
        '/share/value-embedding/datasets/20240319_gen_qa_pairs_translated/analysis/unseen_model_qa/samples_with_no_value.pt',
    ],
}

def load_data_from_model(model_path):
    model_name = model_path.split('/')[-1].split('-ep')[0]
    data_path = model_to_data_mapping[model_name]
    seen_qas, unseen_qas = torch.load(data_path[0]), torch.load(data_path[1])
    unseen_model_cluster_qas, unseen_model_probe_qas = torch.load(data_path[2]), torch.load(data_path[3])
    return seen_qas, unseen_qas, unseen_model_cluster_qas, unseen_model_probe_qas
