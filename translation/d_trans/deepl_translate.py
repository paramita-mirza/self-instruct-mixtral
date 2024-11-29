import jsonlines
import os
import time
import random
import math
from collections import defaultdict
from itertools import islice
from pathlib import Path
from tqdm.auto import tqdm

from d_trans import (tasks, 
                        utils,
                        base,
                        )

import d_trans.translation_apis as tutils

def translate_tasks(tasks_to_translate: str,
                    split: str,
                    limit: int,
                    sample: bool,
                    langs: list[str],
                    dummy: bool,
                    out_dir: str,
                    compute_costs: bool):
    
    out_dir = Path(out_dir)
    if split!="all":
        splits = split.split(",")
    else:
        splits="train,test,validation".split(",")
    chars = 0
    tasks_to_translate, data_paths = utils.get_matches(tasks.task_registry.keys(),tasks_to_translate)
    tasks_to_translate = {f'{task}{"_" + data_path.split("/")[-1].split(".")[0] if data_path else ""}': tasks.task_registry[task](data_path) 
                            for task, data_path in zip(tasks_to_translate, data_paths)}
    '''
    tasks_to_translate, data_paths = utils.get_matches(d_trans.tasks.task_registry.keys(),tasks_to_translate)
    tasks_to_translate = {task:d_trans.tasks.task_registry[task]() \
        for task in zip(tasks_to_translate, data_paths)}'''
    for task_name, task in tasks_to_translate.items():
        if task_name == "instruction":
            assert len(splits)==1
        for lang in langs.split(","):
            for split in splits:
                chars += translate_single_task(task=task,
                                                task_name=task_name,
                                                lang=lang,
                                                dummy=dummy,
                                                split=split,
                                                limit=limit,
                                                sample=sample,
                                                out_dir=out_dir,
                                                compute_costs=compute_costs)
    return chars/50000

def translate_single_task(task: base.Task,
                        task_name: str,
                        lang: str,
                        dummy: bool,
                        split: str,
                        limit: int,
                        sample: bool,
                        out_dir: str,
                        compute_costs: bool=False):
    t_fun = tutils.deepl_translate if not dummy else tutils.dummy_translate
    if "instruction" in task_name:
        assert split=='no_splits'
        task.add_metrics()
        fpath = out_dir/f"{task.dataset_name}_{lang}.jsonl"
    else:
        fpath = out_dir/f"{task_name}_{lang}_{split}.jsonl"
    docs = task.get_docs(split)
    
    if sample:
        strategy = task.sample_strat()
        if strategy=="topics":
            assert limit>1
            groups = utils.group_samples(docs, task)
            docs = utils.stratified_sample(limit, groups)
        elif strategy=="simple":
            assert limit<1
            docs = random.sample(docs,math.floor(limit*len(docs)))
    else:
        if limit is not None:
            docs = list(islice(docs,int(limit))) if limit > 1 else docs
    
    chars = 0
    already_translated = set()
    if os.path.exists(fpath):
        with jsonlines.open(fpath, "r") as f:
            already_translated = {task.doc_id(l) for l in f.iter()}

    if not compute_costs:
        writer = jsonlines.open(fpath, "a")
    for i, doc in tqdm(enumerate(docs),
                        total=len(docs) if limit is None else limit,
                        desc=f"Translating {task_name} to {lang}",
                        disable=compute_costs):
        
        doc = task.convert_data_id_to_target_lang(doc, lang)
        id = task.doc_id(doc)
        
        translatable = task.doc_to_combined(doc)
        if compute_costs:
            chars += len(translatable)
        elif id not in already_translated:
            # translate
            (translated_str,) = t_fun(texts=translatable, lang_pair=("en",lang))
            translated_d = task.combined_to_doc(translatable, translated_str, doc, model_name='deepl', lang=lang)
            #resdict[task_name].append({id:translated_str})

            # save
            if not compute_costs:
                writer.write(translated_d)
    
    if not compute_costs:
        writer.close()
    
    return chars
                    
            
            
    
            
