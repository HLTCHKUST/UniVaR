import os, sys

import numpy as np
import pandas as pd
import json
import glob

from tqdm import tqdm
import datasets
import json
from copy import deepcopy
import torch

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import set_seed

model_lang_map = {
    'CohereForAI/aya-101': [
        'arb_Arab', 'bul_Cyrl', 'deu_Latn', 'eng_Latn', 'fin_Latn', 'tgl_Latn', 'fra_Latn', 'ind_Latn',
        'ita_Latn', 'hin_Deva', 'jpn_Jpan', 'kor_Hang', 'por_Latn', 'ron_Latn', 'rus_Cyrl', 'spa_Latn',
        'swe_Latn', 'tha_Thai', 'vie_Latn', 'zho_Hans', 'pes_Arab', 'zsm_Latn'
    ],
    'mistralai/Mixtral-8x7B-Instruct-v0.1': [
        'fra_Latn', 'deu_Latn', 'spa_Latn', 'ita_Latn', 'eng_Latn'
    ],
    'SeaLLMs/SeaLLM-7B-v2': [
        'eng_Latn', 'zho_Hans', 'vie_Latn', 'ind_Latn', 'tha_Thai', 'zsm_Latn', 'tgl_Latn'
    ],
    'keyfan/bloomz-rlhf': [
        'eng_Latn', 'zho_Hans', 'fra_Latn', 'spa_Latn', 'por_Latn', 'arb_Arab', 'vie_Latn', 'ind_Latn'
    ],
    'THUDM/chatglm3-6b': [
        'zho_Hans', 'eng_Latn'
    ],
    'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO': [
        'fra_Latn', 'deu_Latn', 'spa_Latn', 'ita_Latn', 'eng_Latn'
    ],
    'upstage/SOLAR-10.7B-Instruct-v1.0': [
        'eng_Latn'
    ],
    'MaziyarPanahi/Mistral-7B-Instruct-Aya-101': [
        'fra_Latn', 'deu_Latn', 'spa_Latn', 'ita_Latn', 'eng_Latn'
    ],
    'core42/jais-30b-chat-v3': [
        'arb_Arab', 'eng_Latn'
    ],
    'MaralGPT/Maral-7B-alpha-1': [
        'pes_Arab', 'eng_Latn'
    ],
    'meta-llama/Llama-2-13b-chat-hf': [
        'eng_Latn', 'deu_Latn', 'fra_Latn', 'swe_Latn', 'zho_Hans', 'spa_Latn', 'rus_Cyrl', 'ita_Latn', 
        'jpn_Jpan', 'por_Latn', 'vie_Latn', 'kor_Hang', 'ind_Latn', 'fin_Latn', 'ron_Latn', 'bul_Cyrl'
    ],
    '01-ai/Yi-34B-Chat': [
        'zho_Hans', 'eng_Latn'
    ],
    'meta-llama/Meta-Llama-3-8B-Instruct':[
        'eng_Latn', 'deu_Latn', 'fra_Latn', 'swe_Latn', 'zho_Hans', 'spa_Latn', 'rus_Cyrl', 'ita_Latn', 
        'jpn_Jpan', 'por_Latn', 'vie_Latn', 'kor_Hang', 'ind_Latn', 'fin_Latn', 'ron_Latn', 'bul_Cyrl'
    ]
}

###
# SEALLM Template
###
TURN_TEMPLATE = "<|im_start|>{role}\n{content}</s>"
TURN_PREFIX = "<|im_start|>{role}\n"

def seallm_chat_convo_format(conversations, add_assistant_prefix: bool, system_prompt=None):
    # conversations: list of dict with key `role` and `content` (openai format)
    if conversations[0]['role'] != 'system' and system_prompt is not None:
        conversations = [{"role": "system", "content": system_prompt}] + conversations
    text = ''
    for turn_id, turn in enumerate(conversations):
        prompt = TURN_TEMPLATE.format(role=turn['role'], content=turn['content'])
        text += prompt
    if add_assistant_prefix:
        prompt = TURN_PREFIX.format(role='assistant')
        text += prompt    
    return text

if __name__ == "__main__":    
    for model_path in model_lang_map.keys():
        langs = model_lang_map[model_path]
        model_name = model_path.split('/')[-1]

        print(model_name)
        if 'aya-101' in model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
            bs=16
        elif '30b' in model_path or '8x7B' in model_path or '8x7B' in model_path or '40b' in model_path or '34B' in model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True,
                     quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0)
            )
            bs=32
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
            bs=32

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f'loading model `{model_name}` completed')
        
        for file_path in sorted(glob.glob('./data/*/*_question_mt.csv')):
            out_path = "/".join(file_path.split('/')[:-1]) + f'/qa/{model_name}.csv'
            cache_path = "/".join(file_path.split('/')[:-1]) + f'/cache/{model_name}.pt'

            langs = list(map(lambda x: x.split('_')[0], langs))
            df = pd.read_csv(file_path)
            df = df.loc[df['lang'].isin(langs),:]
            questions = df['question'].tolist()

            print(f'starting generation for `{model_name}` | num data: {len(questions)}')
            if os.path.exists(cache_path):
                answers = torch.load(cache_path)
            else:
                answers = []
            print(f'load cached answers | num data: {len(answers)}')

            for idx in tqdm(range(len(answers),len(questions),bs)):
                texts = questions[idx:idx+bs]

                # Format input text
                prompts = []
                for text in texts:
                    # Format Prompt
                    if 'aya-101' in model_name: 
                        prompts.append(text)
                    elif 'jais' in model_name: 
                        prompts.append(f"### Input: [|Human|] {text}. Please provide a concise explanation\n### Response: [|AI|]")
                    elif 'SeaLLM' in model_name: 
                        chats = [{"role": "user", "content": f'{text}'}]
                        prompts.append(seallm_chat_convo_format(chats, add_assistant_prefix=True))
                    elif 'bloomz-rlhf' in model_name: 
                        prompts.append(f"USER: {text}\nASSISTANT:")
                    elif 'chatglm' in model_name: 
                        prompts.append(f"<|user|>{text}<|assistant|>")
                    elif 'falcon' in model_name: 
                        prompts.append(f"<|prompter|>{text}<|endoftext|><|assistant|>")
                    elif 'Maral' in model_name:
                        prompts.append(f"### Human:{text}\n### Assistant:")
                    else:  
                        # Llama-2-13b-chat-hf, Mistral-7B-Instruct-Aya-101, SOLAR-10.7B-Instruct-v1.0,
                        # Yi-34B-Chat, Mixtral-8x7B-Instruct-v0.1 Nous-Hermes-2-Mixtral-8x7B-DPO
                        chats = [{"role": "user", "content": f'{text}'}]
                        prompts.append(tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True))

                # Tokenize and padding
                inputs = tokenizer(prompts, padding='longest', return_tensors="pt").to(model.device)
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']

                input_len = inputs['input_ids'].shape[1]
                outputs = model.generate(**inputs, do_sample=True, top_k=50, top_p=0.95, max_new_tokens=128)
                if model.config.is_encoder_decoder:
                    answers += tokenizer.batch_decode(outputs, skip_special_tokens=True)
                else:
                    answers += tokenizer.batch_decode(outputs[:,input_len:], skip_special_tokens=True)

                if len(answers) % 1024 == 0:
                    print(f'saving intermediate answers for `{model_name}` | idx: {idx}')
                    torch.save(answers, cache_path)

            print(f'finish generating for `{model_name}`')
            torch.save(answers, cache_path)

            df['answer'] = answers
            df['model'] = model_name
            df.to_csv(out_path, index=False)

        del tokenizer, model
