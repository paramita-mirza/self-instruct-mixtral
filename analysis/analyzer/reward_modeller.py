import logging
import os
import torch
from tqdm import tqdm
from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyzer.utils import plot_histogram, plot_histogram_per_category


attributes_ArmoRM = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
               'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
               'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
               'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
               'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
               'code-style','code-explanation','code-instruction-following','code-readability']

logger = logging.getLogger(__name__)

class RewardModeller(object):
    def __init__(self,
                 model_path: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1", 
                 from_scratch: bool = True, 
                ):
        self.model_path = model_path
        self.from_scratch = from_scratch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_rewards(self, instructions, responses, output_file, return_results=False):
        """
        Generate rewards from a given reward model
        """

        gating_outputs = []
        multi_obj_rewards = []
        preference_scores = []
        multi_obj_coeffs = []

        with torch.no_grad():
            # Run reward model
            for instruction, response in tqdm(zip(instructions, responses)):

                messages = [{"role": "user", "content": instruction },
                            {"role": "assistant", "content": response}]
                input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
                output = self.llm(input_ids)
                multi_obj_rewards.append(output.rewards.cpu().float())
                gating_outputs.append(output.gating_output.cpu().float())
                preference_scores.append(output.score.cpu().float())
                obj_transform = self.llm.reward_transform_matrix.data.cpu().float()
                multi_obj_coeffs.append(gating_outputs[-1] @ obj_transform.T)

        # Stack and save results
        multi_obj_rewards = torch.cat(multi_obj_rewards, dim=0)
        preference_scores = torch.cat(preference_scores, dim=0)
        gating_outputs = torch.cat(gating_outputs, dim=0)
        multi_obj_coeffs = torch.cat(multi_obj_coeffs, dim=0)

        rewards = torch.stack([multi_obj_rewards, multi_obj_coeffs, gating_outputs], dim=0)

        results = {'multi_obj_rewards': rewards, 
                    'preference_scores': preference_scores, 
                    'reward_dimensions': attributes_ArmoRM}

        torch.save(results, output_file)
        
        if return_results:
            return results
    

    def run(self, instructions, responses, dataset_name, dataset_title, output_dir, request_batch_size):
        self.llm = AutoModelForSequenceClassification.from_pretrained(self.model_path,
                                                                      device_map=self.device,
                                                                      trust_remote_code=True,
                                                                      torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        output_file = f"rewards_{self.model_path.split('/')[-1]}_{dataset_name}.pt"
        output_path = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)
                
        # Generate rewards from given reward model
        if not os.path.exists(output_path) or self.from_scratch:
            rewards = self.generate_rewards(
                                        instructions=instructions,
                                        responses=responses, 
                                        output_file=output_path, 
                                        return_results=True)
        else:
            rewards = torch.load(output_path)

        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            for score in rewards['preference_scores']:
                fout.write(f"{score}\n")

        return rewards['preference_scores']

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 0
        max_ylim = 0
        plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (reward model preferences)", scores,
                       min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_category.png",
                                        dataset_title + " (reward model preferences per category)",
                                        scores, categories)

