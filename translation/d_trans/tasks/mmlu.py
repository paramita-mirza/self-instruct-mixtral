import d_trans.base as base

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

def generate_tasks():
    return {f"hendrycks_{arg}":make_task(arg) for arg in SUBJECTS}

def make_task(arg):
    class MMLUsub(MMLU):
        def __init__(self) -> None:
            super().__init__(arg)
            
    return MMLUsub
            
class MMLU(base.Task):
    def __init__(self, subject) -> None:
        self.DATASET_PATH = "hendrycks_test"
        self.DATASET_NAME = subject
        super().__init__()
        self.sep = "<x>SEP</x>"
    
    def doc_to_combined(self, doc):
        combined = doc["question"] + self.sep
        for el in doc["choices"]:
            combined += el + self.sep
        return combined
    
    def combined_to_doc(self, docstr, doc):
        splitstr = docstr.split(self.sep)[:-1]
        outdoc = doc
        outdoc["question"] = splitstr[0]
        outdoc["choices"] = splitstr[1:]
        return outdoc
    
    def get_docs(self, split):
        return [self._preprocess_doc(i, doc) for i, doc in enumerate(self.dataset[split])]

    def _preprocess_doc(self, i, doc):
        doc["id"] = f"{self.DATASET_NAME}/{i}"
        return doc
    
    def doc_id(self, doc):
        #TODO: return unique identifier of document
        return doc["id"]

    def sample_strat(self) -> str:
        return "simple"