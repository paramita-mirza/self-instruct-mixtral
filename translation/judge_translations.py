"""
deploy llama 3.1 70b instruct via vLLM

adapt prompt from mt-bench / alpacaEval

get intersection of all data_ids

"""
import sys
import os
from time import time
import jsonlines
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
import argparse
from d_trans.fewshot_data import fewshot_data as fs_data

sys.path.append('../self_instruct_alpaca')

prompt_alpacaEval_inspired = """Select the translation (a) or (b) that is the best translation of the given source text into {target_language}. Choose your preferred translation, which can be subjective. Your answer should ONLY contain: Translation (a) or Translation (b) and the stop-token ##EOS##. Here are two examples:

{fewshot_data}

# Task:
Now is the real task, do not explain your answer, just say Translation (a) ##EOS## or Translation (b) ##EOS##.

## Source text:
{source_text}

## Translation (a):
{translation_1}

## Translation (b):
{translation_2}

## Which is best, Translation (a) or Translation (b)?"""


abbreviation_map = {
    "EN": "English",
    "BG": "Bulgarian",
    "DA": "Danish",
    "DE": "German",
    "ET": "Estonian",
    "FI": "Finnish",
    "FR": "French",
    "EL": "Greek",
    "IT": "Italian",
    "LV": "Latvian",
    "LT": "Lithuanian",
    "NL": "Dutch",
    "PL": "Polish",
    "PT": "Portuguese",
    "RO": "Romanian",
    "SV": "Swedish",
    "SK": "Slovak",
    "SL": "Slovenian",
    "ES": "Spanish",
    "CS": "Czech",
    "HU": "Hungarian"
}

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="trans_evalset_v1", help="Name of translated dataset")
parser.add_argument("--deployment", type=str, default='vllm_local', choices=['vllm_local', 'hf_local', 'openai'], help="Choose the deployment method")
parser.add_argument("--source_file", type=str, default="/raid/s3/opengptx/lucas/data/trans_evalset_v1.jsonl")
parser.add_argument("--judge", type=str, default='meta-llama/Meta-Llama-3.1-70B-Instruct')
parser.add_argument("--model1", type=str, default='Meta-Llama-3.1-70B-Instruct')
parser.add_argument("--model2", type=str, default='Mixtral-8x22B-Instruct-v0.1')
parser.add_argument("--out_dir", type=str, default="/home/lucas-weber/synthetic-instruction-data-generation/translation/output")
parser.add_argument("--lang", type=str)
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--max_model_len", type=int, default=4096)
args = parser.parse_args()

cache_dir = os.environ['HF_HOME'] + '/hub'

    

def judge_data(source_texts: str | list[str],
            translations_1: str | list[str],
            translations_2: str | list[str],
            lang_pair: tuple[str],
            ):
        if lang_pair[1]=="en":
            return source_texts

        source_texts =   [source_texts]     if not isinstance(source_texts, list)        else source_texts   
        translations_1 = [translations_1]   if not isinstance(translations_1, list)      else translations_1   
        translations_2 = [translations_2]   if not isinstance(translations_2, list)      else translations_2  
        
        source_lang, target_lang = abbreviation_map[lang_pair[0].upper()], abbreviation_map[lang_pair[1].upper()]

        prompts = [format_prompt(target_language=target_lang, source_text=source_text, translation_1=translation_1, 
                                 translation_2=translation_2, fewshot_data=fs_data[lang_pair[1].upper() + "_judge"]) 
                   for source_text, translation_1, translation_2 in zip(source_texts, translations_1, translations_2)] 

        request_start = time()
        results = request_api.make_requests(prompts, max_tokens=10, do_sample=True, temperature=.0, top_p=1., stop_sequences = ["##EOS##"])
        request_duration = time() - request_start

        if isinstance(results[0], dict):
            results = [res["response"] for res in results]
            
        results = [res.split("##EOS##")[0].strip() for res in results]

        return results
    
def format_prompt(**kwargs):
        """  
        """ 
        return prompt_alpacaEval_inspired.format(**kwargs).strip()

def format_datapoint(doc):
    if all(key in doc for key in ["instruction", "input", "output"]):
        return "{instruction}\n{input}\n{output}".format(**doc)
    elif all(key in doc for key in ["messages"]):
        formatted = "\n".join([message["content"] for message in doc["messages"]])
        return formatted

def convert_id_to_lang(data_id, lang):    
    split_id = data_id.split('_')
    split_id[2] = lang_pair[0].lower()
    return "_".join(split_id)

if __name__ == "__main__":
    path_source_texts = args.source_file
    path_translations_1 = f'output/evalset_translations/{args.task_name}_{args.lang.upper()}_{args.model1}.jsonl'
    path_translations_2 = f'output/evalset_translations/{args.task_name}_{args.lang.upper()}_{args.model2}.jsonl'
    
    with jsonlines.open(path_source_texts, "r") as f:
        source_texts = [l for l in f.iter()]
    with jsonlines.open(path_translations_1, "r") as f:
        translations_1 = [l for l in f.iter()]
    with jsonlines.open(path_translations_2, "r") as f:
        translations_2 = [l for l in f.iter()]

    if args.deployment == 'vllm_local':
        from llm.vllm_api import VLLM
        request_api = VLLM(model_name=args.judge, num_devices=args.num_devices, max_model_len=args.max_model_len, cache_dir=cache_dir, enforce_eager=True, 
                            gpu_memory_utilization=0.95)
    elif args.deployment == 'hf_local':
        from llm.huggingface_api import HuggingFaceLLM
        request_api = HuggingFaceLLM(model_name=args.judge, cache_dir=cache_dir)
    elif args.deployment == 'openai':
        from llm.openai_api import OpenAILLM
        request_api = OpenAILLM(model=args.judge)

    judge_name = args.judge.split('/')[1] if '/' in args.judge else args.judge

    lang_pair = ("EN", args.lang.upper())
    
    id_intersection = sorted(list(set([doc['data_id'] for doc in translations_1]).intersection(set([doc['data_id'] for doc in translations_2]))))
    id_intersection_source = [convert_id_to_lang(data_id, lang_pair[0]) for data_id in id_intersection]
    
    translations_1 = [doc for doc in translations_1 if doc['data_id'] in id_intersection]
    translations_2 = [doc for doc in translations_2 if doc['data_id'] in id_intersection]
    source_texts = [doc for doc in source_texts if doc['data_id'] in id_intersection_source]

    judgements = {'data_id': [], 
                    'winner': [],
                    'model1': [],
                    'model2': [],
                    'language': [],
                    'judge': []}
    
    for i, datapoint in tqdm(enumerate(zip(source_texts, translations_1, translations_2)), 
                        total=len(source_texts),
                        desc=f"Judging {args.task_name} translations from models {args.model1} and {args.model2}"):
        
        (judgement, ) = judge_data(format_datapoint(datapoint[0]), format_datapoint(datapoint[1]), format_datapoint(datapoint[2]), lang_pair)
        
        winner = args.model1 if judgement == "Translation (a)" else args.model2 if judgement == "Translation (b)" else judgement
        
        judgements['data_id'].append(datapoint[0]['data_id'])
        judgements['winner'].append(winner)
        judgements['model1'].append(args.model1)
        judgements['model2'].append(args.model2)
        judgements['language'].append(lang_pair[1])
        judgements['judge'].append(judge_name)
        
    
    save_path = os.path.join(args.out_dir, f'judge_{judge_name}')
    os.makedirs(save_path, exist_ok=True)
    
    pd.DataFrame(judgements).to_csv(os.path.join(save_path, f'{args.task_name}_judgements_{args.model1}_vs_{args.model2}.csv'), index=False)
