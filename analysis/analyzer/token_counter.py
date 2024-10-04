import logging
from transformers import AutoTokenizer
from typing import List, Dict
from analyzer.utils import plot_histogram, plot_histogram_per_category

logger = logging.getLogger(__name__)

class TokenCounter(object):

    def __init__(self):
        self.model_name = "hkust-nlp/deita-quality-scorer"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def infer_length(self, user_request: str, system_response: str):
        return (len(self.tokenizer(user_request, return_tensors="pt")['input_ids'][0]),
                len(self.tokenizer(system_response, return_tensors="pt")['input_ids'][0]))

    def run(self, instructions: List[str], responses: List[str], dataset_name: str, dataset_title: str, output_dir: str, request_batch_size: int=16):
        instruction_length = []
        response_length = []
        for i, inst in enumerate(instructions):
            inst_length, resp_length = self.infer_length(instructions[i], responses[i])
            instruction_length.append(inst_length)
            response_length.append(resp_length)

        # Write token counts to file
        with open(f"{output_dir}/{dataset_name}_instructions.csv", "w") as fout:
            for len in instruction_length:
                fout.write(f"{len}\n")

        with open(f"{output_dir}/{dataset_name}_responses.csv", "w") as fout:
            for len in response_length:
                fout.write(f"{len}\n")

        return {"instruction_length": instruction_length, "response_length": response_length}

    def plot(self, scores: Dict, dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 0.0
        max_ylim = 0.0
        instruction_length = scores["instruction_length"]
        response_length = scores["response_length"]
        plot_histogram(f"{output_dir}/{dataset_name}_instructions.png", dataset_title + " (instruction length)", instruction_length, min_ylim, max_ylim)
        plot_histogram(f"{output_dir}/{dataset_name}_responses.png", dataset_title + " (response length)", response_length, min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_instructions_category.png",
                                        dataset_title + " (instruction length per category)",
                                        instruction_length, categories)
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_response_category.png",
                                        dataset_title + " (response length per category)",
                                        response_length, categories)


