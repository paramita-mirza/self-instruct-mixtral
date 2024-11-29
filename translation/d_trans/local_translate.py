import jsonlines
import os
import time
import random
import math
from itertools import islice
from pathlib import Path
from tqdm.auto import tqdm

from d_trans import (tasks, 
                     utils,
                     base,
                     )
from d_trans.translation_apis import LocalTranslator

# TODO: 
# - handle cases without separators
# - use prompt per setfit category?
# - find a good way to filter 'stopping short' translations
# - test the possibility to do 'postprocessing' with an LLM (e.g. 'check if the output makes sense given the input. If not, reformulate.')

import fasttext
model = fasttext.load_model('/raid/s3/opengptx/lucas/models/lid.176.bin')

def get_language(text: str):
    predictions = model.predict(text, k=1)  # k=1 means returning top 1 prediction
    language_code = predictions[0][0].replace("__label__", "")  # Extract the language code
    confidence = predictions[1][0]  # Extract the confidence score
    return language_code, confidence

def package_docs(docs):
    """ In case of kto data, two datapoints share the same prompt, so we package them and translate them together """
    if all(key in docs[0] for key in ["prompt", "completion", "label"]):
        packed_docs = []
        for i in range(0, len(docs), 2):
            _ = docs[i].pop("label")
            prompt = docs[i].pop("prompt")
            completion1 = docs[i].pop("completion")
            completion2 = docs[i+1].pop("completion")
            packed_docs.append({"prompt": prompt, "completion": [completion1, completion2], **docs[i]})
        return packed_docs
    else:
        return docs

def unpack_doc(doc):
    """ Undo the package_docs operation after translation """
    if all(key in doc for key in ["prompt", "completion"]):
        data_id = doc.pop("data_id").split("_")
        return [{"prompt": doc["prompt"], "completion": doc["completion"][0], "label": True, "data_id": "_".join(data_id), **doc}, 
                {"prompt": doc["prompt"], "completion": doc["completion"][1], "label": False, "data_id": "_".join(data_id[:-1] + [str(1+int(data_id[-1]))]), **doc}]
    else:
        return doc


def translate_tasks(tasks_to_translate: str,
                    split: str,
                    limit: int,
                    sample: bool,
                    langs: list[str],
                    out_dir: str,
                    request_api,
                          ):
    
    Translator = LocalTranslator(request_api)
    
    out_dir = Path(out_dir)
    if split!="all":
        splits = split.split(",")
    else:
        splits="train,test,validation".split(",")

    tasks_to_translate, data_paths = utils.get_matches(tasks.task_registry.keys(),tasks_to_translate)
    tasks_to_translate = {f'{task}{"_" + data_path.split("/")[-1].split(".")[0] if data_path else ""}': tasks.task_registry[task](data_path) 
                          for task, data_path in zip(tasks_to_translate, data_paths)}
    for task_name, task in tasks_to_translate.items():
        for lang in langs.split(","):
            for split in splits:
                translate_single_task(task=task,
                                    task_name=task_name,
                                    lang=lang,
                                    split=split,
                                    limit=limit,
                                    sample=sample,
                                    out_dir=out_dir,
                                    translator=Translator
                                    )

def translate_single_task(task: base.Task,
                        task_name: str,
                        lang: str,
                        split: str,
                        limit: int,
                        sample: bool,
                        out_dir: str,
                        translator
                        ):
    batch_size = 16
    
    model_name = translator.request_api.model_name
    
    if "instruction" in task_name:
        assert split=='no_splits'
        task.add_metrics()
        fpath = out_dir/f"{task.dataset_name}_{lang}_{model_name.split('/')[1]}.jsonl"
    else:
        fpath = out_dir/f"{task_name}_{lang}_{split}.jsonl"
    docs = task.get_docs(split)
    
    # In case that the data is from kto, we package the docs
    docs = package_docs(docs)
    
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
    
    already_translated = set()
    if os.path.exists(fpath):
        with jsonlines.open(fpath, "r") as f:
            already_translated = {task.doc_id(l) for l in f.iter()}

    failed_translations = []
    batch_texts = []
    batch_docs = []
    n_translations = 0
    n_skips = 0
    writer = jsonlines.open(fpath, "a")
    for i, doc in tqdm(enumerate(docs), 
                        total=len(docs) if limit is None else limit,
                        desc=f"Translating {task_name} to {lang}"):
        
        doc = task.convert_data_id_to_target_lang(doc, lang)
        id = task.doc_id(doc)

        if len(batch_texts) != batch_size and id not in already_translated:
            translatable = task.doc_to_combined(doc)
            batch_texts += [translatable]
            batch_docs += [doc]
            if i != len(docs)-1:
                continue
        elif id in already_translated:
            n_skips += 1
            continue
    
        # translate
        batch_texts_translated = translator(texts=batch_texts, lang_pair=("en",lang))
        for text, translated_text, t_doc in zip(batch_texts, batch_texts_translated, batch_docs):
            pred_lang, conf = get_language(translated_text.replace("\n", " "))
            if pred_lang.lower() != lang.lower() or conf < 0.8:
                failed_translations.append(t_doc['data_id'])
                continue
            translated_d = task.combined_to_doc(text, translated_text, t_doc, model_name, lang)
            translated_d = unpack_doc(translated_d)
            if isinstance(translated_d, str):
                failed_translations.append(translated_d)
                continue
            # save
            if isinstance(translated_d, list):
                for d in translated_d:
                    writer.write(d)
            else:
                writer.write(translated_d)
            n_translations += 1
        batch_texts = []
        batch_docs = []
        
    print(f"Successfully translated {n_translations}/{len(docs)-n_skips} {task_name} datapoints to {lang}.")
    writer.close()


