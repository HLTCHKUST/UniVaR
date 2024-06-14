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

target_langs = [
    'zho', 'kor', 'jpn', 'deu', 'fin', 'swe', 'fra', 'ita', 'por', 'spa', 'tha', 'vie',
    'zsm', 'tgl', 'hat', 'quy', 'rus', 'ron', 'bul', 'ind', 'arb', 'hin', 'swh', 'pes'
]

if __name__ == "__main__":
    # Load Model & Tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=lang_map['eng'])
    batch_size = 32
    
    for file_path in glob.glob('./data/*/*_question.csv'):
        out_path = file_path.replace('_question.csv', '_question_mt.csv')
        if os.path.exists(out_path):
            print(f'Skipping {file_path}...')
            continue    
        print(f'Processing {file_path}...')
        
        df = pd.read_csv(file_path)
        en_df = df.copy()
        en_df['lang'] = 'eng'

        lang_dfs = [en_df]
        qa_prompts = df['question'].tolist()

        for lang in target_langs:
            # Translate Answer
            mt_lang_code = lang_map[lang]
            print(f'Processing {lang_map["eng"]} to {mt_lang_code} translation')

            tdf = df.copy() # Copy the data frame
            mt_questions = []
            for i in range(0,len(qa_prompts),batch_size):
                # Translate Answers
                inputs = tokenizer(qa_prompts[i:i+batch_size], padding='longest', return_tensors="pt").to('cuda')
                translated_tokens = model.generate(
                    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[mt_lang_code], 
                    min_length=int(inputs['input_ids'].shape[1] * 0.25), max_length=int(inputs['input_ids'].shape[1] * 3)
                )
                mt_questions += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                del inputs, translated_tokens

            # Fill the MT data frame buffer
            lang_code = mt_lang_code.split('_')[0]
            tdf['question'] = mt_questions
            tdf['lang'] = lang_code

            # Append to final buffer
            lang_dfs.append(tdf)

            mt_df = pd.concat(lang_dfs)
            mt_df = mt_df.sort_values(['lang','qid']).reset_index(drop=True)
            mt_df.to_csv(out_path, index=False)