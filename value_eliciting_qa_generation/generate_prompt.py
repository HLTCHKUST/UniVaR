import os,sys
import datasets
from nltk import word_tokenize
import pandas as pd
import torch
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

def load_model():
    # Load Tokenizer & Model
    model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_path, token='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', trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_path, token='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', torch_dtype=torch.float16, trust_remote_code=True, load_in_8bit=True
    )
    model = torch.compile(model)
    return tokenizer, model

def get_situation_qa(batch, model, tokenizer):
    # Encode Batch
    formatted_prompts = []
    for situation in batch['situation']:
        chats = [
            {"role": "user", "content": f'Given a premise about "{situation}", write a question asking whether the speaker should do or not do the aforementioned premise.'},
            {"role": "assistant", "content": "\nSure, here is a concise question asking whether the speaker should proceed with the given premise or not:"},
        ]
        formatted_prompts.append(tokenizer.apply_chat_template(chats, tokenize=False))

    # Get Response
    inputs = tokenizer(formatted_prompts, return_tensors='pt', add_special_tokens=False, padding='longest').to(model.device)
    input_ids = inputs['input_ids']
    output_ids = model.generate(
        **inputs, eos_token_id=tokenizer.eos_token_id, top_p=0.9, min_length=5, max_length=input_ids.shape[1] + 128, do_sample=True
    )

    # Decode Batch
    outputs = []
    for i, out in enumerate(output_ids):
        qa_prompt = tokenizer.decode(out[len(input_ids[i]):], skip_special_tokens=True)
        qa_prompt = qa_prompt.strip().split('\n')[0]
        if qa_prompt[0] == '#':
            qa_prompt = qa_prompt[1:].strip()
        if qa_prompt[-1] == '"':
            qa_prompt = ''.join(qa_prompt.split('"')[1:-1])
        outputs.append(qa_prompt)

    return {'qid': batch['qid'], 'question': outputs}

if __name__ == "__main__":
    batch_size = 32
    
    # Load Tokenizer & Model
    tokenizer, model = load_model()

    for file_path in glob.glob('./data/*/*_situation.csv'):
        out_path = file_path.replace('_situation.csv', '_question.csv')
        if os.path.exists(out_path):
            print(f'Skipping {file_path}...')
            continue    
        print(f'Processing {file_path}...')
        
        HISTORY_IDS = []
        if os.path.isfile(out_path):
            HISTORY_IDS = pd.read_csv(out_path).index.tolist()

        # Load Dataset
        df = pd.read_csv(file_path)
        df = df.drop_duplicates('situation')
        df = df.loc[~df.index.isin(HISTORY_IDS),:]
        dset = datasets.Dataset.from_pandas(df)

        # LLM Inference    
        for i, batch in tqdm(enumerate(dset.iter(batch_size=batch_size)), total=dset.num_rows//batch_size):
            if batch['qid'][0] in HISTORY_IDS:
                continue
            batch = get_situation_qa(batch, model, tokenizer)
            tdf = pd.DataFrame(batch)
            if HISTORY_IDS or i:
                tdf.to_csv(out_path, mode='a', index=False, header=False)
            else:
                tdf.to_csv(out_path, mode='a', index=False, header=True)  