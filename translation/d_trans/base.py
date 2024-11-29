from abc import ABC, abstractmethod
from datasets import load_dataset

class Task(ABC):
    DATASET_PATH = None
    DATASET_NAME = None
    def __init__(self) -> None:
        self.download()
    
    def download(self):
        self.dataset = load_dataset(path=self.DATASET_PATH,
                                    name=self.DATASET_NAME)
    
    @abstractmethod
    def get_docs(self, split: str):
        """
        Return the task documents belonging to the specified split.
        """
        pass

    @abstractmethod
    def doc_to_combined(self, doc: dict) -> str:
        """
        Return the combined translation text for a specified document.
        """
        pass
    
    @abstractmethod
    def combined_to_doc(self, docstr: str) -> dict:
        """
        Inverse operation of doc_to_combined.
        """
        pass
    
    @abstractmethod
    def doc_id(self, doc: dict) -> str:
        """
        Return the id of a document.
        """
        pass
    
    @abstractmethod
    def sample_strat(self) -> str:
        """
        Return sampling strategy of the task
        Choices: 'topics','simple'
        """
        pass