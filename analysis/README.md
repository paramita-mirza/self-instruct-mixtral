# SIGMA-analysis: A Repository for Analyzing Generated Instructions

This repository contains code and data for analyzing instruction-following data.

- We use [SetFit](https://github.com/huggingface/setfit) for efficient few-shot learning with Sentence Transformers
for classifying instructions into 10 categories: [Classify, Extract, Closed QA, Generation, Open QA, Rewrite,
Brainstorm, Coding, Summarize, Math]. We construct the training data (``data/no_robots_gsm8k_train.jsonl``) by sampling 50 samples per category from 
[No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) dataset and 
[GSM8K](https://github.com/google-research/FLAN/blob/main/flan/v2/cot_data/gsm8k_train.tsv) dataset. 
Each sample is manually checked whether it really belongs to the category. Evaluated on [No Robots test split](https://huggingface.co/datasets/HuggingFaceH4/no_robots/viewer/default/test) 
of 500 instructions, the classifier gets 94% macro F1-score.
- We use [Deita complexity scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer) and 
[Deita quality scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer) for scoring (instruction) complexity and 
(response) quality w.r.t a given instruction
- We use [InsTagger](https://huggingface.co/OFA-Sys/InsTagger) for tagging instructions, following 
[this paper](https://arxiv.org/pdf/2308.07074)
- We use [vllm](https://github.com/vllm-project/vllm) for faster inference

### Running Analysis

#### Prerequisites

1. Create a virtual environment ``python -m venv analysis-env`` then activate `source analysis-env/bin/activate`
2. Install the dependencies with ``pip install -r requirements.txt``

#### Running SIGMA-cls for classifying instructions into 10 categories

1. Run ``python run_analysis.py --analysis categories``, which will run the classifier on all considered datasets. 
Otherwise, run ``python run_analysis.py --analysis categories --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python run_analysis.py --analysis categories --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The categories (and pie chart) will be saved in ``categories/``.

#### Running Complexity Analysis

1. Run ``python run_analysis.py --analysis complexity``, which will run the complexity scorer on all considered datasets. 
Otherwise, run ``python run_analysis.py --analysis complexity --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python run_analysis.py --analysis complexity --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The scores (and histogram) will be saved in ``complexity_scores/``.

#### Running Quality Analysis

1. Run ``python run_analysis.py --analysis quality``, which will run the quality scorer on all considered datasets. 
Otherwise, run ``python run_analysis.py --analysis quality --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python run_analysis.py --analysis quality --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The scores (and histogram) will be saved in ``quality_scores/``.

#### Complexity/Quality analysis on existing datasets

Run ``python compute_correlations.py``

![Alt text](correlations/avg_complexity_quality.png?raw=true "Title")

#### Running InsTagger

1. Run ``python run_analysis.py --analysis tagging``, which will run the quality scorer on all considered datasets. 
Otherwise, run ``python run_analysis.py --analysis tagging --dataset <dataset_name(s)>`` to run the classifier on 
specific dataset(s), e.g., ``python run_analysis.py --analysis tagging --dataset sigma_v2_evol,deita_10k``. 
List of available datasets below.
2. The scores (and histogram) will be saved in ``instagger/``.

#### Diversity analysis on existing datasets, based on #InsTag and embedding distance

Run ``python compute_correlations.py``

![Alt text](correlations/num_unique_tags_10k.png?raw=true "Title")
![Alt text](correlations/num_unique_tags_10k_avg_embedding.png?raw=true "Title")

#### Correlation between scores, token length and #tags

![Alt text](correlations/instlen_complexity.png?raw=true "Title")
![Alt text](correlations/resplen_quality.png?raw=true "Title")
![Alt text](correlations/numtag_complexity.png?raw=true "Title")

#### List of Analyzed Datasets
- sigma_v1
- sigma_v2_evol
- sigma_v3
- deita_10k
- no_robots
- flan_v2_cot
- dolly_15k
- alpaca
- alpaca_gpt4
- lima
- longform
- bactrian-x_en
- wizardlm_evol_instruct
- wizardlm_orca
- sharegpt
- oasst2
- ultrachat

#### Add a new dataset to be analyzed
1. Add a dataset processing in ``utils.py`` with `dataset_name` as the key, making sure that 
`instructions` contains all user requests in the dataset and `responses` contains the corresponding 
system responses. Please add also `dataset_title` to show the dataset name on the charts, e.g., 
```
    elif dataset_name == "longform":
        dataset_title = "LongForm"
        dataset = load_dataset("akoksal/LongForm")  # from Huggingface Hub
        longform_train = dataset["train"]
        instructions = [sample["input"] for sample in longform_train]
        responses = [sample["output"] for sample in longform_train]
```


