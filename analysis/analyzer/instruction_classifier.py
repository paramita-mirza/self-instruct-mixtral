import logging
from typing import List
from tqdm import tqdm
import json
from analyzer.utils import plot_categories
from setfit import SetFitModel

logger = logging.getLogger(__name__)

class InsClassifier(object):

    COLOR_MAP = {
        'Brainstorm': 'C0',
        'Classify': 'C1',
        'Closed QA': 'C2',
        'Coding': 'C3',
        'Extract': 'C4',
        'Generation': 'C5',
        'Math': 'C6',
        'Open QA': 'C7',
        'Rewrite': 'C8',
        'Summarize': 'C9',
        'Chat': 'C10'
    }

    def __init__(self):
        self.model_path = "paramitopia/sigma-cls"

    def run(self, instructions: List[str], responses: List[str], dataset_name: str, dataset_title: str, output_dir: str, request_batch_size: int=16):
        self.model = SetFitModel.from_pretrained(self.model_path)
        instances = [f"USER: {request}\n\nASSISTANT: {responses[i]}"
                        for i, request in enumerate(instructions)]
        predicted_categories = self.model.predict(instances, show_progress_bar=True)

        # Write categories to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            for category in predicted_categories:
                fout.write(f"{category}\n")

        return predicted_categories

    def plot(self, predicted_categories: List[str], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the pie chart
        label_stats = {}
        for label in predicted_categories:
            if label not in label_stats: label_stats[label] = 0
            label_stats[label] += 1
        labels = sorted(list(label_stats.keys()))
        x = [label_stats[label] for label in labels]
        plot_categories(f"{output_dir}/{dataset_name}.png", dataset_title + " (category)", x, labels, self.COLOR_MAP)

