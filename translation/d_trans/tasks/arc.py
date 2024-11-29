from collections import defaultdict
import d_trans.base as base

def generate_tasks():
    tasks = {"arc_challenge": make_task('ARC-Challenge'),
             "arc_easy": make_task('ARC-Easy')}
    return tasks

def make_task(subs):
    class arctask(arc):
        def __init__(self) -> None:
            self.DATASET_NAME = subs
            super().__init__()
            
    return arctask
            
class arc(base.Task):
    def __init__(self) -> None:
        self.DATASET_PATH = "ai2_arc"
        super().__init__()
        self.sep = "<x>SEP</x>"
        self.id_col = "id"
    
    def doc_to_combined(self, doc):
        combined_prompt = doc["question"]+self.sep
        for choice in doc["choices"]["text"]:
            combined_prompt += choice + self.sep
        return combined_prompt
    
    def combined_to_doc(self, docstr, doc):
        split_str = docstr.split(self.sep)
        doc["question"] = split_str[0]
        for i in range(1,len(split_str)-1):
            doc["choices"]["text"][i-1] = split_str[i]
        return doc
        
    def doc_to_stack(self, doc):
        pass
    
    def get_docs(self, split):
        return list(self.dataset[split])
    
    def doc_id(self, doc):
        return doc["id"]
    
    def sample_strat(self) -> str:
        return "simple"