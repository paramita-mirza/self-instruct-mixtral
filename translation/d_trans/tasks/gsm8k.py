import d_trans.base as base

class GSM8k(base.Task):
    def __init__(self) -> None:
        self.DATASET_PATH = "gsm8k"
        self.DATASET_NAME = "main"
        super().__init__()
        self.sep = "<x>SEP</x>"
    
    def doc_to_combined(self, doc):
        return doc["question"] + self.sep + doc["answer"]
    
    def combined_to_doc(self, docstr, doc):
        splitstr = docstr.split(self.sep)
        outdoc = doc.copy()
        outdoc["question"] = splitstr[0]
        outdoc["answer"] = self._process_answer(splitstr[1])
        return outdoc
    
    def _process_answer(self, ans: str):
        ans = ans.replace("&lt;","<")
        ans = ans.replace("&gt;",">")
        return ans
    
    def get_docs(self, split):
        return [self._preprocess_doc(i, doc) for i, doc in enumerate(self.dataset[split])]
    
    def doc_id(self, doc):
        return doc["id"]
    
    def _preprocess_doc(self, i, doc):
        doc["id"] = i
        return doc
    
    def sample_strat(self) -> str:
        return "simple"