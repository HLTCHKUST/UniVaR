import os, sys
import glob

import numpy as np
import pandas as pd
import json

from tqdm import tqdm
import json
from copy import deepcopy
import torch

import openai
import cohere
import anthropic

model_lang_map = {
    'gpt-3.5-turbo': [
        'eng_Latn', 'zho_Hans', 'kor_Hang', 'jpn_Jpan', 'deu_Latn', 'fin_Latn', 'swe_Latn',
        'fra_Latn', 'spa_Latn', 'ita_Latn', 'por_Latn', 'tha_Thai', 'vie_Latn', 'zsm_Latn',
        'tgl_Latn', 'hat_Latn', 'quy_Latn', 'rus_Cyrl', 'ron_Latn', 'bul_Cyrl', 'ind_Latn',
        'arb_Arab', 'swh_Latn', 'hin_Deva', 'pes_Arab'
    ],
    'gpt-4-turbo': [
        'eng_Latn', 'zho_Hans', 'kor_Hang', 'jpn_Jpan', 'deu_Latn', 'fin_Latn', 'swe_Latn', 
        'fra_Latn', 'spa_Latn', 'ita_Latn', 'por_Latn', 'tha_Thai', 'vie_Latn', 'zsm_Latn', 
        'tgl_Latn', 'hat_Latn', 'quy_Latn', 'rus_Cyrl', 'ron_Latn', 'bul_Cyrl', 'ind_Latn', 
        'arb_Arab', 'swh_Latn', 'hin_Deva', 'pes_Arab'
    ],
    'claude-haiku': [
        'eng_Latn', 'por_Latn', 'fra_Latn', 'deu_Latn'
    ],
    'claude-sonnet': [
        'eng_Latn', 'por_Latn', 'fra_Latn', 'deu_Latn'
    ],
    'claude-opus': [
        'eng_Latn', 'por_Latn', 'fra_Latn', 'deu_Latn'
    ],
    'command-r': [
        'eng_Latn', 'fra_Latn', 'spa_Latn', 'ita_Latn', 'deu_Latn', 'por_Latn', 'jpn_Jpan', 
        'kor_Hang', 'arb_Arab', 'zho_Hans'
    ],
    'command-r-plus': [
        'eng_Latn', 'fra_Latn', 'spa_Latn', 'ita_Latn', 'deu_Latn', 'por_Latn', 'jpn_Jpan', 
        'kor_Hang', 'arb_Arab', 'zho_Hans'
    ],
}

###
# LLM API
###
def openai_api(openai_client, input_text, system_text = '', model_name = 'gpt-4-1106-preview'):
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system", 
                "content": system_text,
                "role": "user",
                "content": input_text,
            },
        ],
        max_tokens=128
    )
    return completion.choices[0].message.content

def claude_api(anthropic_client, input_text, system_text = '', model_name='claude-3-opus-20240229'):
    message = anthropic_client.messages.create(
        model=model_name,
        system=system_text,
        messages=[
            {"role": "user", "content": input_text}
        ],
        max_tokens=128
    )
    return message.content

def cohere_api(cohere_client, input_text, system_text = '', model_name='command-r-plus'):
    response = cohere_client.chat(
        model=model_name,
        chat_history=[],
        message=input_text,
        preamble=system_text,
        connectors=[],
        max_tokens=128
    )
    return response.text

if __name__ == "__main__":
    openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    anthropic_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    cohere_client = cohere.Client(os.environ['COHERE_API_KEY'])

    for file_path in glob.glob('./data/*/*_question_mt.csv'):
        out_folder = "/".join(file_path.split('/')[:-1]) + '/qa'
        cache_folder = "/".join(file_path.split('/')[:-1]) + '/cache'
        
        df = pd.read_csv(file_path)        
        for model_name in model_lang_map.keys():
            out_path = f'{out_folder}/{model_name}.csv'
            cache_path = f'{cache_folder}/{model_name}.pt'

            langs = model_lang_map[model_name]
            langs = list(map(lambda x: x.split('_')[0], langs))

            df = df.loc[df['lang'].isin(langs),:]
            questions = df['question'].tolist()

            print(f'starting generation for `{model_name}` | num data: {len(questions)}')
            if os.path.exists(cache_path):
                answers = torch.load(cache_path)
            else:
                answers = []
            print(f'load cached answers | num data: {len(answers)}')

            for idx in tqdm(range(len(answers),len(questions))):
                text = questions[idx]
                
                # Get LLM Response from API
                if 'gpt-3.5-turbo' == model_name:
                    out = openai_api(openai_client, text, model_name=model_name)
                elif 'gpt-4-turbo' == model_name:
                    out = openai_api(openai_client, text, model_name=model_name)
                elif 'claude-haiku' == model_name:
                    out = claude_api(anthropic_client, text, model_name=model_name)
                elif 'claude-sonnet' == model_name:
                    out = claude_api(anthropic_client, text, model_name=model_name)
                elif 'claude-opus' == model_name:
                    out = claude_api(anthropic_client, text, model_name=model_name)
                elif 'command-r' == model_name:
                    out = cohere_api(cohere_client, text, model_name=model_name)    
                elif 'command-r-plus' == model_name:
                    out = cohere_api(cohere_client, text, model_name=model_name)  
                else:
                    raise ValueError(f'Unknown `model_name`: "{model_name}"')
                answers.append(out)

                if len(answers) % 16 == 0:
                    print(f'saving intermediate answers for `{model_name}` | idx: {idx}')
                    torch.save(answers, cache_path)

            print(f'finish generating for `{model_name}`')
            torch.save(answers, cache_path)

            df['answer'] = answers
            df['model'] = model_name
            df.to_csv(out_path, index=False)
