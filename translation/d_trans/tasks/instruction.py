import d_trans.base as base
import os
import json


class instructions(base.Task):
    def __init__(self, data_path) -> None:
        self.DATASET_PATH = data_path
        self.dataset_name = self.get_task_name()
        self.metric_names = {'embedding_distance': 'embedding_distance', # how the folder is called vs. how the key is called
                             'complexity_scores': 'complexity', 
                             'quality_scores': 'quality', 
                             'instagger': 'ins_tags', 
                             'reward_scores': 'reward',
                             'categories': 'setfit_label'
                             }
        super().__init__()
        self.sep = "<x>SEP</x>"     
        self.id_counter = 0
    
    def download(self):
        with open(self.DATASET_PATH) as f: 
            self.dataset = [json.loads(line) for line in f]

    def add_metrics(self):
        for metric_name, metric_clean_name in self.metric_names.items():
            try:
                if metric_name == "instagger":
                    with open(f"../analysis/{metric_name}/{self.dataset_name}.jsonl") as f:
                        for sample, line in zip(self.dataset, f.readlines()): 
                            sample.update({metric_clean_name: json.loads(line) if line.strip() != 'None' else None}) 
                else:     
                    with open(f"../analysis/{metric_name}/{self.dataset_name}.csv") as f:
                        for sample, line in zip(self.dataset, f.readlines()): 
                            if metric_name == "categories": #if line.strip() != 'None':
                                sample.update({metric_clean_name: line.strip() if line.strip() != 'None' else None})
                            else:
                                sample.update({metric_clean_name: float(line.strip()) if line.strip() != 'None' else None})
            except FileNotFoundError:
                print(f"Could not find {metric_clean_name} for {self.dataset_name} at default path.")
                
    def get_task_name(self):
        return self.DATASET_PATH.split("/")[-1].split(".")[0]
    
    def convert_data_id_to_target_lang(self, doc, lang):
        if "data_id" in doc:
            split_id = doc["data_id"].split('_')
            split_id[2] = lang.lower()
            doc["data_id"] =  "_".join(split_id)
        return doc

    def doc_to_combined(self, doc):
        # combine a document into a single translatable string
        if all(key in doc for key in ["instruction", "instances"]):
            return doc["instruction"] + self.sep + doc["instances"][0]['input'] + self.sep + doc["instances"][0]['output']
        elif all(key in doc for key in ["instruction", "input", "output"]):
            return doc["instruction"] + self.sep + doc["input"] + self.sep + doc["output"]
        elif all(key in doc for key in ["messages"]):
            conversation = ''
            for i, message in enumerate(doc["messages"]):    
                conversation += message['content'] + (" " + self.sep + " " if i < len(doc["messages"])-1 else '')
            return conversation
        elif all(key in doc for key in ["prompt", 'completion']):
            return self.sep.join([d['content'] for d in doc["prompt"]] 
                                 + [d['content'] for d in doc["completion"][0]]
                                 + [d['content'] for d in doc["completion"][1]])
        else:
            breakpoint()
            raise KeyError("Document does not have any expected key combinations")
        
    
    def combined_to_doc(self, text, text_translated, doc, model_name, lang):
        # separate translated string into original format
        components = text.split(self.sep)
        components_translated = text_translated.split(self.sep)
        outdoc = doc.copy()
        
        outdoc["translator"] = model_name

        if len(components_translated) != len(components):
            # failed to translate in correct format
            return outdoc['data_id']

        if all(key in doc for key in ["instruction", "instances"]):
            outdoc["instruction"] = components_translated[0]
            outdoc["instances"][0]['input'] = components_translated[1]
            outdoc["instances"][0]['output'] = components_translated[2]
        elif all(key in doc for key in ["instruction", "input", "output"]):
            outdoc["instruction"] = components_translated[0]
            outdoc["input"] = components_translated[1]
            outdoc["output"] = components_translated[2]
        elif all(key in doc for key in ["messages"]):
            roles = ["user", "assistant"]
            outdoc["messages"] = [{"role": roles[i%2], "content": components_translated[i]} for i in range(len(components_translated))]
        elif all(key in doc for key in ["prompt", 'completion']):
            # take the role from original prompt; original prompt might contain multiple prompts (e.g. system prompt and user prompt)
            outdoc["prompt"] = [{'content': components_translated[i], 'role': d['role']} for i, d in enumerate(doc["prompt"])]
            outdoc["completion"] = [{'content': components_translated[i+len(doc["prompt"])], 'role': d[0]['role']} for i, d in enumerate(doc["completion"])]
        else:
            raise KeyError("Document does not have any expected key combinations")
        return outdoc
        
    def doc_to_stack(self, doc):
        #TODO (optional): return a list of document substrings without separation
        pass
    
    def stack_to_doc(self, doc):
        #TODO (optional): combine translated stack of document substrings into original format
        pass
    
    def get_docs(self, split):
        #TODO: get the documents belonging to split
        return self.dataset
    
    def doc_id(self, doc):
        # return unique identifier of datapoint
        try:
            # check if datapoint as id value
            return next((doc[key] for key in ['data_id', 'id'] if key in doc))
        except KeyError:
            id = self.id_counter
            self.id_counter += 1
            return id
    
    def sample_strat(self) -> str:
        return "simple"