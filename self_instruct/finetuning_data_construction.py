import argparse
import json
import random
import os
import torch
from chromadb_utils import create_collection, populate_collection, add_to_collection
from tqdm import tqdm
import yaml
from time import time
from collections import Counter
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to yaml file containing dataset configs.",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the final dataset.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.985,
        help="Fraction of train split (the rest as val).",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="/raid/s3/opengptx/lucas/data/analysis",
        help="Where to store the analysis results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/raid/s3/opengptx/lucas/data",
        help="Where to store the generated data."
    )
    return parser.parse_args()

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
        'Chat': 'C10',
        np.nan: 'lightgrey'
    }

# TODO:
# 1. add more elaborate sampling strategies


class Sampler:
    def __init__(self, dataset_name):
        self.all_data = []
        self.all_tags = set()
        self.dataset_name = dataset_name
        self.metric_names = {'embedding_distance': 'embedding_distance', # how the folder is called vs. how the key is called
                             'complexity_scores': 'complexity', 
                             'quality_scores': 'quality', 
                             'instagger': 'ins_tags', 
                             'reward_scores': 'reward',
                             'categories': 'setfit_label'
                             }
        
        self._check_data_presence(config['data'])
        self.metric_normalisation_factor = self._get_metric_normalisation_factors(config['data'], ['complexity', 'quality', 'reward'])
        self.chromadb_collection = None
        self.rewards_weight = 0.4
        self.quality_weight = 0.4
        self.complexity_weight = 0.2
    
    def load_dataset(self, dataset_config):
        with open(dataset_config['data_path']) as finst:
            samples = [json.loads(line) for line in finst.readlines()]
        samples = self._add_metrics(samples, dataset_config['name'])
        samples = self._add_origin(samples, dataset_config['name'])
        if 'multi_turn' not in dataset_config or dataset_config['multi_turn'] == False:
            samples = self._add_conversation(samples)
        if 'language' in dataset_config:
            samples = self._add_language(samples, dataset_config['language'])
        return samples

    @staticmethod
    def _identify_language(model, text):
        # Use the model to predict the language of the text
        predictions = model.predict(text, k=1)  # k=1 means returning top 1 prediction
        language_code = predictions[0][0].replace("__label__", "")  # Extract the language code
        confidence = predictions[1][0]  # Extract the confidence score
        return language_code, confidence

    def filter_dataset_by_language(self, samples, language_codes = ["en"]):
        import fasttext
        model = fasttext.load_model('models/lid.176.bin')

        filtered_samples = []
        for sample in samples:
            if 'language' not in sample:
                if 'instruction' in sample:
                    language_code, confidence = self._identify_language(model, sample['instruction'].replace("\n", " "))
                else:
                    language_code, confidence = self._identify_language(model, sample['messages'][0]['content'].replace("\n", " "))
                sample['language'] = language_code
                sample['lang_confidence'] = confidence
            if sample['language'] in language_codes:
                filtered_samples.append(sample)
        return filtered_samples

    def _add_conversation(self, samples):
        conversational_samples = []
        for sample in samples:
            if 'input' in sample:
                sample['messages'] = [
                    {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
                    {"role": "assistant", "content": sample['output']}
                ]
                del sample['input']
            else:
                sample['messages'] = [
                    {"role": "user", "content": sample['instruction']},
                    {"role": "assistant", "content": sample['output']}
                ]
            del sample['instruction']
            del sample['output']
            conversational_samples.append(sample)
        return conversational_samples
    
    def _add_origin(self, samples, origin):
        return [{**sample, 'origin': origin} if 'origin' not in sample else {**sample} for sample in samples]

    def _add_language(self, samples, lang):
        return [{**sample, 'language': lang} if 'language' not in sample else {**sample} for sample in samples]
        
    def _add_metrics(self, samples, dataset_name):
        for metric_name, metric_clean_name in self.metric_names.items():
            try:
                if metric_name == "instagger":
                    with open(f"{args.analysis_dir}/{metric_name}/{dataset_name}.jsonl") as f:
                        for sample, line in zip(samples, f.readlines()): 
                            sample.update({metric_clean_name: json.loads(line) if line.strip() != 'None' else None}) 
                else:     
                    with open(f"{args.analysis_dir}/{metric_name}/{dataset_name}.csv") as f:
                        for sample, line in zip(samples, f.readlines()): 
                            if metric_name == "categories": #if line.strip() != 'None':
                                sample.update({metric_clean_name: line.strip() if line.strip() != 'None' else None})
                            else:
                                sample.update({metric_clean_name: float(line.strip()) if line.strip() != 'None' else None})
            except FileNotFoundError:
                print(f"Could not find {metric_clean_name} for {dataset_name} at default path.")
        return samples
  
    
    def save_metrics(self):
        """Save collective metrics of constructed dataset"""
        for metric_name, metric_clean_name in self.metric_names.items():
            if metric_name == "instagger":
                with open(f"{args.analysis_dir}/{metric_name}/{self.dataset_name}.jsonl", "w") as f:
                    metric_values = [inst.get(metric_clean_name, None) for inst in self.all_data]
                    for value in metric_values:
                        f.write(json.dumps(value) + "\n")
            else:
                with open(f"{args.analysis_dir}/{metric_name}/{self.dataset_name}.csv", "w") as f:
                    metric_values = [inst.get(metric_clean_name, None) for inst in self.all_data]
                    for value in metric_values:
                        f.write(f"{value}\n")
                    
    def save_dataset(self, shuffle=True, train_split=None):
        """Save constructed dataset"""

        if shuffle:
            self.all_data = random.sample(self.all_data, len(self.all_data))

        with open(f"{args.output_dir}/{self.dataset_name}.jsonl", "w") as f:
            for inst in sampler.all_data:
                f.write(json.dumps(inst)+"\n")

        if train_split >= 0.0:

            random.shuffle(self.all_data)
            path = f'{args.output_dir}/{self.dataset_name}'
            os.makedirs(path, exist_ok=True)

            if train_split > 0.0 and train_split < 1.0:
                train_split_size = int(train_split * len(self.all_data))
                with open(path + '/train.json', 'w') as file:
                    json.dump(self.all_data[:train_split_size], file, indent=4)
                with open(path + '/val.json', 'w') as file:
                    json.dump(self.all_data[train_split_size:], file, indent=4)
            elif train_split == 1.0:
                with open(path + '/train.json', 'w') as file:
                    json.dump(self.all_data, file, indent=4)
            else:
                with open(path + '/val.json', 'w') as file:
                    json.dump(self.all_data, file, indent=4)
                
    def plot_final_composition(self, by='origin'):
        """ TODO: plot exact amounts of samples, not just percentages"""
        used_keys = [by] 
        results = {key: [instruction.get(key, None) for instruction in self.all_data] for key in used_keys}
        results_df = pd.DataFrame(results)
        counts = Counter(results_df[by])
        plt.figure()
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}, textprops={'size': 'small'})
        plt.title({self.dataset_name}, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"dataset_configs/{self.dataset_name}_{by}.png", bbox_inches="tight")
             
    def plot_metrics(self, by='origin'):
        """Plot metrics of constructed dataset"""
        used_keys = ['origin', 'overall_preference'] + list(self.metric_names.values())
        results = {key: [instruction.get(key, None) for instruction in self.all_data] for key in used_keys}
        results['categories'] = [inst.get('setfit_label', 'no_setfit_label') for inst in self.all_data]
        results_df = pd.DataFrame(results)

        # Scores
        for metric_name, metric_name_clean in self.metric_names.items():
            if metric_name in ["instagger", "categories"]:
                continue
            plt.figure(figsize=(8, 6))
            sns.histplot(data=results_df, x=metric_name_clean, hue=by, multiple="stack")
            plt.axvline(results_df[metric_name_clean].mean(), color='grey', linestyle='--')
            plt.text(results_df[metric_name_clean].mean(), 0, f"mean: {results_df[metric_name_clean].mean():.2f}", rotation=90)
            plt.title(self.dataset_name)
            plt.savefig(f"{args.analysis_dir}/{metric_name}/{self.dataset_name}_by_{by}.png")
        
        # Overall preference
        plt.figure(figsize=(8, 6))
        sns.histplot(data=results_df, x='overall_preference', hue=by, multiple="stack")
        plt.title(self.dataset_name)
        os.makedirs(f"{args.analysis_dir}/overall_preferences", exist_ok=True)
        plt.savefig(f"{args.analysis_dir}/overall_preferences/{self.dataset_name}_by_{by}.png")
        
        # Categories
        counts = Counter(results_df['categories'])
        plt.figure()
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
            textprops={'size': 'small'}, colors=[COLOR_MAP[key] for key in counts.keys()])
        plt.title({self.dataset_name}, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{args.analysis_dir}/categories/{self.dataset_name}.png", bbox_inches="tight")
        

    
    def init_chromadb(self, embedding_model):
        documents = []
        for sample in self.all_data:
            user_messages = ""
            for turn in sample['messages']:
                if turn['role'] == "user":
                    user_messages += f"{turn['content']} "
            documents.append(user_messages.strip())
        self.chromadb_collection = create_collection(collection_name="instruction_embeddings",
                                            embedding_model=embedding_model, device=device)
        populate_collection(collection=self.chromadb_collection, documents=documents, batch_size=100)
                

    def calculate_overall_preference(self, samples, scores_exist = True):
        """Calculate a simple preference score"""
        for sample in samples:
            if scores_exist:
                if 'reward' in sample:
                    sample['overall_preference'] = self.rewards_weight      * sample['reward'] / self.metric_normalisation_factor['reward'] + \
                                                   self.quality_weight      * sample['quality'] / self.metric_normalisation_factor['quality'] + \
                                                   self.complexity_weight   * sample['complexity'] / self.metric_normalisation_factor['complexity']
                elif 'edu_score' in sample:
                    sample['overall_preference'] = sample['edu_score']
            else:
                # raise ValueError(f"Please calculate missing dataset metrics. See sample: \n{sample}")
                sample['overall_preference'] = 1
        return samples
    
    
    def process_data(self, dataset_config):
        """Sample data and add them to the all_data"""
        data = self.load_dataset(dataset_config)
        dataset_name = dataset_config['name']
        if 'subset_name' in dataset_config: dataset_name += f"_{dataset_config['subset_name']}"

        sample_scored = True
        if "sample_scored" in dataset_config: sample_scored = dataset_config['sample_scored']
        data = self.calculate_overall_preference(data, sample_scored)
        
        if 'filter_key' in dataset_config:
            # in case of subsets, filter the data according to subset name
            filter_key = dataset_config['filter_key']
            data = [instruction for instruction in data if instruction.get(filter_key) == dataset_config['subset_name']]

        self.init_setfit_limits(dataset_config)
        sorted_samples = sorted(data, key=lambda x: x['overall_preference'], reverse=True)

        if 'language_filter' in dataset_config:
            sorted_samples = self.filter_dataset_by_language(sorted_samples, dataset_config['language_filter'])

        if self.chromadb_collection is None:
            # Base dataset (self instruct)
            i = -1
            while len(self.all_data) < dataset_config['sample_size'] and i < len(sorted_samples) - 1:
                i += 1
                sample = sorted_samples[i]
                if 'setfit_limit' in dataset_config:
                    if not self.check_setfit_limits(sample.get('setfit_label')):
                        continue
                    self.update_setfit_limits(sample.get('setfit_label'))
                self.all_data.append(sample)
            
            # initialize ChromaDB
            self.init_chromadb(config.pop("chromadb_embedding_model"))
            for sample in self.all_data:
                if "ins_tags" in sample and sample["ins_tags"] is not None:
                    tags = set([tag['tag'] for tag in sample["ins_tags"] if "tag" in tag])
                    self.all_tags.update(tags)

            print(f"{dataset_name}; Sampled: {len(self.all_data)}; InsTags: {len(self.all_tags)}; Total: {len(self.all_data)}")

        else:
            # Additional datasets
            t = 0.3
            if 'min_distance' in dataset_config: t = dataset_config['min_distance']
            sampled_data = []
            i = -1
            pbar = tqdm(total=dataset_config['sample_size'])
            while len(sampled_data) < dataset_config['sample_size'] and i < len(sorted_samples) - 1:
                i += 1
                sample = sorted_samples[i]
                         
                # Check setfit limits
                if 'setfit_limit' in dataset_config:
                    if not self.check_setfit_limits(sample.get('setfit_label')):
                        continue
                
                # Check for new instags
                new_tags = set()
                if "ins_tags" in sample and sample["ins_tags"] is not None:
                    tags = set([tag['tag'] for tag in sample["ins_tags"] if "tag" in tag])
                    new_tags = tags.difference(self.all_tags)

                user_messages = ""
                for turn in sample['messages']:
                    if turn['role'] == "user":
                        user_messages += f"{turn['content']} "
                query_text = user_messages.strip()

                if t > 0.0:
                    # Query ChromaDB for similar instructions
                    results = self.chromadb_collection.query(
                        query_texts=[query_text],
                        n_results=1)
                    distance = results['distances'][0][0]
                    sample["embedding_distance"] = distance

                    # Add samples based on distance or new tags
                    if distance > t or len(new_tags) > 0: # or sample.get('setfit_label') == 'Math' or dataset_config['name'] == 'fewshot':
                        sampled_data.append(sample)
                        pbar.update(1)
                        self.all_tags.update(new_tags)
                        add_to_collection(
                            collection=self.chromadb_collection,
                            documents=[query_text])
                        if 'setfit_limit' in dataset_config:
                            self.update_setfit_limits(sample.get('setfit_label'))

                else:   # min_distance is 0.0, no need to check with Chroma
                    sampled_data.append(sample)
                    pbar.update(1)
                    self.all_tags.update(new_tags)
                    if 'setfit_limit' in dataset_config:
                        self.update_setfit_limits(sample.get('setfit_label'))
                    
            self.all_data.extend(sampled_data)
            print(f"+{dataset_name}; Sampled: {len(sampled_data)}; InsTags: {len(self.all_tags)}; Total: {len(self.all_data)}")
        
    def _check_data_presence(self, data_configs):
        # check that data exists
        for dataset in data_configs:
            dataset_name = dataset['name']
            assert os.path.exists(dataset['data_path']), f"Data path {dataset['data_path']} does not exist."
        
            # check that metrics exist
            for metric_name in self.metric_names.keys():
                    if metric_name == "instagger":
                        if not os.path.exists(f"{args.analysis_dir}/{metric_name}/{dataset_name}.jsonl"):
                            print(f"Could not find {metric_name} for {dataset_name} at default path.")
                    else:
                        if not os.path.exists(f"{args.analysis_dir}/{metric_name}/{dataset_name}.csv"):
                            print(f"Could not find {metric_name} for {dataset_name} at default path.")   
                        

    def _get_metric_normalisation_factors(self, data_configs, metric_names): 
        """loads metrics from all datasets and calculates 95th percentile for normalisation"""
        
        all_metric_values = {metric: [] for metric in metric_names}
        # Load metrics from all datasets
        for dataset in data_configs:
            dataset_name = dataset['name']
            for metric_name in metric_names:
                try:
                    with open(f"{args.analysis_dir}/{metric_name}_scores/{dataset_name}.csv") as f:
                        all_metric_values[metric_name] += [float(line.strip()) for line in f.readlines() if line.strip() != 'None']
                except FileNotFoundError:
                    pass
        
        all_metrics_percentiles = {}
        # Calculate 95th percentile for normalisation
        for metric_name in metric_names:
            if all_metric_values[metric_name]:
                all_metrics_percentiles[metric_name] = torch.tensor(all_metric_values[metric_name]).quantile(0.95).item()
        
        return all_metrics_percentiles       
        
    def _final_setfit_filtering(self, setfit_proportions):
        """ Take dict of setfit labels and proportions, determine maximal possible dataset-size and filter the dataset accordingly""" 
        # sort data by setfit label
        samples_by_setfit = {}
        for setfit_label in setfit_proportions.keys():
            samples_by_setfit[setfit_label] = [sample for sample in self.all_data if sample['setfit_label'] == setfit_label]

        # determine maximally possible final size based on fixed proportions
        max_possible_size = int(min([len(samples_by_setfit[setfit_label]) / proportion if proportion > 0 else 10**9 for setfit_label, proportion in setfit_proportions.items()]))

        print(f"Shrinking dataset from {len(self.all_data)} to {max_possible_size} samples.")

        # filter data based on overall preference following setfit labels
        orig_data_size = len(self.all_data)
        old_size_variable_proportions = sum([len(samples) for setfit_label, samples in samples_by_setfit.items() if setfit_proportions[setfit_label] < 0]) / orig_data_size
        new_size_variable_proportions = (1 - sum([proportion for proportion in setfit_proportions.values() if proportion > 0]))
                        
        self.all_data = []
        for setfit_label, proportion in setfit_proportions.items():
            if proportion < 0:
                # if proportion is not specified, keep original proportion of samples
                proportion = (len(samples_by_setfit[setfit_label]) / orig_data_size / old_size_variable_proportions) * new_size_variable_proportions

            samples_by_setfit[setfit_label] = sorted(samples_by_setfit[setfit_label], key=lambda x: x['overall_preference'], reverse=True)
            self.all_data.extend(samples_by_setfit[setfit_label][:int(max_possible_size  * proportion)])

        self.all_tags = set()
        for sample in self.all_data:
            if "ins_tags" in sample and sample["ins_tags"] is not None:
                tags = set([tag['tag'] for tag in sample["ins_tags"] if "tag" in tag])
                self.all_tags.update(tags)

        print(f"InsTags: {len(self.all_tags)}; Total: {len(self.all_data)}")
    
    def init_setfit_limits(self, dataset_config):
        """Set max numbers to be added to dataset per setfit label"""
        if 'setfit_limits' in dataset_config:
            self.setfit_limits = {category: limit * dataset_config['sample_size'] 
                                       for setfit_label in dataset_config['setfit_limits'] 
                                       for category, limit in setfit_label.items()}

    def check_setfit_limits(self, setfit_label):
        """Check if there are more samples that can be added to the dataset for a given setfit label"""
        return self.setfit_limits[setfit_label] > 0
    
    def update_setfit_limits(self, setfit_label):
        """Update the number of samples that can be added to the dataset for a given setfit label"""
        self.setfit_limits[setfit_label] -= 1
                
if __name__ == "__main__":
    args = parse_args()
    
    with open(os.path.join('dataset_configs', args.config), 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    dataset_name, config = config.popitem()
    sampler = Sampler(dataset_name)

    for dataset_config in tqdm(config['data'], desc="Processed datasets"):
        # if sampler.chromadb_collection is None:
        #     assert dataset_config['type'] == 'self_instruct', "First dataset in yaml must be the self_instruct base dataset."

        # Check if there are subsets to handle
        if 'subsets' in dataset_config:
            for subset_config in tqdm(dataset_config['subsets'], desc="Processed subsets"):
                subset_dataset_config = {
                    **dataset_config,  # inherit base config for the dataset
                    **subset_config    # override with subset-specific values
                }
                sampler.process_data(subset_dataset_config)
        else:
            sampler.process_data(dataset_config)
    
    if 'final_setfit_proportions' in config:
        sampler._final_setfit_filtering(config['final_setfit_proportions'])

    sampler.save_dataset(shuffle=args.shuffle, train_split=args.train_split)
    sampler.save_metrics()
    sampler.plot_final_composition(by='origin')
    sampler.plot_final_composition(by='language')
    sampler.plot_metrics(by='origin')
    sampler.plot_metrics(by='categories')

