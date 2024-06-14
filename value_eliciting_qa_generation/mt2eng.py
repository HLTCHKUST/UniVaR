import os, sys

import glob
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
import datasets
import json
from copy import deepcopy
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import set_seed

lang_map = {
    'zho': 'zho_Hans', 'kor': 'kor_Hang', 'jpn': 'jpn_Jpan', 'deu': 'deu_Latn', 'fin': 'fin_Latn', 
    'swe': 'swe_Latn', 'fra': 'fra_Latn', 'ita': 'ita_Latn', 'por': 'por_Latn', 'spa': 'spa_Latn', 
    'tha': 'tha_Thai', 'vie': 'vie_Latn', 'zsm': 'zsm_Latn', 'tgl': 'tgl_Latn', 'hat': 'hat_Latn', 
    'quy': 'quy_Latn', 'rus': 'rus_Cyrl', 'ron': 'ron_Latn', 'bul': 'bul_Cyrl', 'ind': 'ind_Latn', 
    'arb': 'arb_Arab', 'hin': 'hin_Deva', 'swh': 'swh_Latn', 'pes': 'pes_Arab', 'eng': 'eng_Latn'
}

if __name__ == "__main__":
    # Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.bfloat16).to('cuda')

    lang_dfs = []
    for file_path in glob.glob('./data//*/qa/*.csv'):
        file_name = file_path.split('/')[-1]
        out_path = file_path.replace('/qa/', '/qa_translated/')
        if os.path.exists(out_path):
            print(f'Skipping {file_name}')
            continue
        print(f'Processing {file_name}')
        df = pd.read_csv(file_path).fillna('-')

        mt_dfs = []
        for lang, tdf in df.groupby('lang'):
            print(f'Processing {file_name} | {lang_map[lang]} to {lang_map["eng"]} translation')
            # Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=lang_map[lang])

            if lang == 'eng':
                tdf['mt_lang'] = 'eng'
                mt_dfs.append(tdf)
            else:
                # Translate Question & Answer
                questions = tdf['question'].tolist()
                answers = tdf['answer'].tolist()
                batch_size = 32

                # Translate Questions
                mt_questions = []
                for i in tqdm(range(0,len(questions),batch_size)):
                    inputs = tokenizer(questions[i:i+batch_size], padding='longest', truncation=True, return_tensors="pt", max_length=256).to('cuda')
                    translated_tokens = model.generate(
                        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_map['eng']], 
                        min_length=int(inputs['input_ids'].shape[1] * 0.25), max_length=int(inputs['input_ids'].shape[1] * 3)
                    )
                    mt_questions += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                    del inputs, translated_tokens

                # Translate Answers
                mt_answers = []
                for i in tqdm(range(0,len(answers),batch_size)):
                    inputs = tokenizer(answers[i:i+batch_size], padding='longest', truncation=True, return_tensors="pt", max_length=256).to('cuda')
                    translated_tokens = model.generate(
                        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_map['eng']], 
                        min_length=int(inputs['input_ids'].shape[1] * 0.25), max_length=int(inputs['input_ids'].shape[1] * 3)
                    )
                    mt_answers += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                    del inputs, translated_tokens

                # Fill the MT data frame buffer
                tdf['question'] = mt_questions
                tdf['answer'] = mt_answers
                tdf['mt_lang'] = 'eng'
                mt_dfs.append(tdf)
        mt_df = pd.concat(mt_dfs)
        mt_df.to_csv(out_path, index=False)