import d_trans.base as base
            
class Winogrande(base.Task):
    def __init__(self) -> None:
        self.DATASET_PATH = "winogrande"
        self.DATASET_NAME = "winogrande_xl"
        super().__init__()
        self.sep = "<x>SEP</x>"     #TODO: set separator as desired
    
    def doc_to_combined(self, doc):
        sen = doc["sentence"]
        outsen = sen.split("_")[0] + "<x>_</x>" + sen.split("_")[1]
        return outsen
    
    def combined_to_doc(self, docstr, doc):
        #TODO (optional): separate translated string into original format
        pass
        
    def doc_to_stack(self, doc):
        #TODO (optional): return a list of document substrings without separation
        pass
    
    def stack_to_doc(self, doc):
        #TODO (optional): combine translated stack of document substrings into original format
        pass
    
    def get_docs(self, split):
        #TODO: get the documents belonging to split
        pass
    
    def doc_id(self, doc):
        #TODO: return unique identifier of document
        return doc["id"]