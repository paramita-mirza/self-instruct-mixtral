import d_trans.base as base

def generate_tasks(args):
    return {arg:make_task(arg) for arg in args}

def make_task(arg):
    class newsubtask(newtask):
        def __init__(self) -> None:
            self.DATASET_NAME = arg
            super().__init__()
            
    return newsubtask
            
class newtask(base.Task):
    def __init__(self) -> None:
        self.DATASET_PATH = None    #TODO: set correct download path
        super().__init__()
        self.sep = "<x>SEP</x>"     #TODO: set separator as desired
    
    def doc_to_combined(self, doc):
        #TODO (optional): combine a document into a single translatable string
        pass
    
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