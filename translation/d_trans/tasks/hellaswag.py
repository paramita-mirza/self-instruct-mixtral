import re

import d_trans.base as base

class Hellaswag(base.Task):
    def __init__(self) -> None:
        self.DATASET_PATH = "hellaswag"
        super().__init__()
        self.sep = "<x>SEP</x>"     #TODO: set separator as desired
        self.id_col = "id"
    
    def doc_to_combined(self, doc):
        combined = ""
        for key in ["activity_label", "ctx_a", "ctx_b"]:
            combined += doc[key] + self.sep
        for e in doc["endings"]:
            combined += e + self.sep
        return combined
    
    def combined_to_doc(self, docstr, doc):
        outdoc = doc
        split_str = docstr.split(self.sep)[:-1]
        for i, key in enumerate(["activity_label", "ctx_a", "ctx_b"]):
            outdoc[key] = split_str[i]
        outdoc["ctx"] = outdoc["ctx_a"] + " " + outdoc["ctx_b"]
        outdoc["endings"] = split_str[3:]
        return outdoc
        
    def doc_to_stack(self, doc):
        pass
    
    def stack_to_doc(self, doc):
        #TODO (optional): combine translated stack of document substrings into original format
        pass
    
    def get_docs(self, split):
        return [self._process_doc(doc,i) for i,doc in enumerate(self.dataset[split])]
    
    def doc_id(self, doc):
        return doc["source_id"]

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
    
    def _process_doc(self, doc, i):
        out_doc = doc
        for key in ["ctx_a", "ctx_b", "ctx"]:
            out_doc[key] = self.preprocess(out_doc[key])
        out_doc["endings"] = [self.preprocess(e) for e in out_doc["endings"]]
        out_doc["id"] = i
        return out_doc

    def get_topic(self, doc):
        t = doc["source_id"].split("~")[0]
        return t

    def sample_strat(self):
        return "topics"