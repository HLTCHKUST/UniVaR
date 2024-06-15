import random
import numpy as np
import torch
from copy import deepcopy

def encode_value_dataset(value_dset, model, batch_size=128, truncation=None, qa_per_view=1):
    value_reps = {}
    for value, qa_df in value_dset.items():
        print(f'Encoding {value}...')
        all_qa_pairs = qa_df['qa'].tolist()#[:truncation]
        input_samples = []
        for i in range(min(len(all_qa_pairs), truncation)):
            if qa_per_view==1:
                input_samples.append(all_qa_pairs[i])
            else:
                qa_pairs = random.sample(all_qa_pairs, qa_per_view)
                input_samples.append('\n'.join(qa_pairs))

        value_reps[value] = model.encode(input_samples, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
    return value_reps

def encode_qas(qas, model, batch_size=128, truncation=None, qa_per_view=1, convert_to_numpy=True, show_progress_bar=True):
    input_samples = []
    for i in range(len(qas) // qa_per_view):
        if qa_per_view==1:
            input_samples.append(qas[i])
        else:
            qa_pairs = random.sample(qas, qa_per_view)
            input_samples.append('\n'.join(qa_pairs))

    return model.encode(input_samples, convert_to_numpy=convert_to_numpy, batch_size=batch_size, show_progress_bar=show_progress_bar)

def expand_sentence_transformer_encoder(sentence_transformer, num_extended_layers=6, freeze_prior_modules=False):
    num_hidden_layers = sentence_transformer[0].auto_model.config.num_hidden_layers
    
    if sentence_transformer[0].auto_model.config.architectures[0] == 'BertModel':
        sentence_transformer[0].auto_model.encoder.layer.extend([
            deepcopy(sentence_transformer[0].auto_model.encoder.layer[-(i+1)]) for i in range(num_extended_layers)
        ])
        
        if freeze_prior_modules:
            for params in sentence_transformer[0].auto_model.embeddings.parameters():
                params.requires_grad = False
            for params in sentence_transformer[0].auto_model.encoder.layer[:num_hidden_layers].parameters():
                params.requires_grad = False

        sentence_transformer[0].auto_model.config.num_hidden_layers = num_hidden_layers + num_extended_layers
    else:
        raise NotImplementedError
        
    return sentence_transformer

def get_params_count(model, max_name_len: int = 64):
    params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
    total_trainable_params = sum([x[1] for x in params if x[-1]])
    total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
    return params, total_trainable_params, total_nontrainable_params


def get_params_count_summary(model, max_name_len: int = 64):
    padding = 64
    params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
    param_counts_text = ''
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    for name, param_count, shape, trainable in params:
        truncated_name = name[:max_name_len]  # Truncate the name if it's too long
        param_counts_text += f'| {truncated_name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
    param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    return param_counts_text


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)