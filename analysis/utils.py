import json
import random
from datasets import Dataset, load_dataset


def load_sft_dataset(dataset_name: str):
    if dataset_name == "sigma_v1":
        dataset_title = "SIGMA v1"
        samples = [json.loads(l) for l in open("../data/sigma_10k.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "sigma_v1_regen":
        dataset_title = "SIGMA v1 (output regenerated)"
        samples = [json.loads(l) for l in open("../data/sigma_10k_regenerated.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "sigma_10k_llama_8b":
        dataset_title = "SIGMA 10k (with LLama3.1-8B)"
        samples = [json.loads(l) for l in open("../data/sigma_10k_llama3-1_8B.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "sigma_10k_llama_70b":
        dataset_title = "SIGMA 10k (with LLama3.1-70B)"
        samples = [json.loads(l) for l in open("../data/sigma_10k_llama3-1_70B.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "llama_sigma2_10k":
        dataset_title = "SIGMA2 10k (with LLama3.1-8B)"
        samples = [json.loads(l) for l in open("../data/llama_sigma2_10k.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "llama_sigma2_10k_regenerated":
        dataset_title = "SIGMA2 10k (with LLama3.1-8B, output regenerated)"
        samples = [json.loads(l) for l in open("../data/llama_sigma2_10k_regenerated.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]
    
    elif dataset_name == "nemotron-4-340B_self_instruct":
        dataset_title = "Self-instruct (with Nemotron-4-340B)"
        samples = [json.loads(l) for l in open("../data/nemotron-4-340B_self_instruct.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]
    
    elif dataset_name == "code-search-net":
        dataset_title = "CodeSearchNet 20k"
        samples = [json.loads(l) for l in open("../data/cs_net_python_20k.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]
        
    elif dataset_name == "sigma_v2":
        dataset_title = "SIGMA v2"
        samples = [json.loads(l) for l in open("../data/sigma_v2.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "sigma_v2_regen":
        dataset_title = "SIGMA v2 (output regenerated)"
        samples = [json.loads(l) for l in open("../data/sigma_v2_regenerated.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "sigma_v2_evol":
        dataset_title = "SIGMA v2 (auto evol)"
        samples = [json.loads(l) for l in open("../data/sigma_v2_auto_evol_filtered.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]
        
    elif dataset_name == "magicoder_4k":
        dataset_title = "OSS-Instruct (with DeepSeek V2 Coder Instruct)"
        samples = [json.loads(l) for l in open("../data/magicoder_4k.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]
        
    elif dataset_name == "sigma_v2_evol_math":
        dataset_title = "SIGMA v2 (auto evol) + GSM8K + MATH"
        samples = [json.loads(l) for l in open("../data/sigma_v2_auto_evol_math.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" if "input" in sample and sample["input"] != ""
                        else f"{sample['instruction']}"
                        for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "sigma_v3":
        dataset_title = "SIGMA v3"
        samples = [json.loads(l) for l in open("../data/sigma_v3.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" if "input" in sample and sample["input"] != ""
                        else f"{sample['instruction']}"
                        for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "self_instruct_mix_v1":
        dataset_title = "Self-Instruct Mix v1"
        samples = [json.loads(l) for l in open("../data/self_instruct_mix_v1.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" if "input" in sample and sample["input"] != ""
                        else f"{sample['instruction']}"
                        for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "no_robots":
        dataset_title = "No Robots"
        dataset = load_dataset("HuggingFaceH4/no_robots")
        train_ds = dataset["train"]
        instructions = [example['messages'][0]["content"] for example in train_ds]
        responses = [example['messages'][1]["content"] for example in train_ds]

    elif dataset_name == "flan_v2_cot":
        # Download Flan V2 CoT data from https://github.com/google-research/FLAN/tree/main/flan/v2/cot_data
        dataset_title = "Flan V2 CoT"
        samples = [json.loads(l) for l in open("../data/flan_v2_cot.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" if "input" in sample and sample["input"] != ""
                        else f"{sample['instruction']}"
                        for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "dolly_15k":
        # Download Dolly data from https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl
        dataset_title = "Databricks Dolly 15k"
        samples = [json.loads(l) for l in open("../data/dolly_15k.jsonl", "r")]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" if "input" in sample and sample["input"] != ""
                        else f"{sample['instruction']}"
                        for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "alpaca":
        # Download Alpaca data from https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
        dataset_title = "Stanford Alpaca"
        samples = json.load(open('../data/alpaca_data.json', 'r'))
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in samples]
        responses = [sample['output'] for sample in samples]

    elif dataset_name == "alpaca_gpt4":
        dataset_title = "Stanford Alpaca (with GPT4)"
        dataset = load_dataset("vicgalle/alpaca-gpt4")
        alpaca_train = dataset["train"]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in alpaca_train]
        responses = [sample['output'] for sample in alpaca_train]

    elif dataset_name == "lima":
        dataset_title = "LIMA"
        dataset = load_dataset("GAIR/lima")
        lima_train = dataset["train"]
        instructions = [f"{sample[0]}" for sample in lima_train["conversations"]]
        responses = [f"{sample[1]}" for sample in lima_train["conversations"]]

    elif dataset_name == "bactrian-x_en":
        dataset_title = "Bactrian-X-en"
        dataset = load_dataset("MBZUAI/Bactrian-X", "en")
        bactrian_train = dataset["train"]
        instructions = [f"{sample['instruction']}\n\n{sample['input']}" for sample in bactrian_train]
        responses = [sample['output'] for sample in bactrian_train]

    elif dataset_name == "wizardlm_orca":
        dataset = load_dataset("pankajmathur/WizardLM_Orca")
        dataset_title = "WizardLM-Orca"
        wizardlm_train = dataset["train"]
        instructions = [f"{sample['instruction']}" for sample in wizardlm_train]
        responses = [sample['output'] for sample in wizardlm_train]

    elif dataset_name == "sharegpt":
        # Download ShareGPT data from https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
        dataset_title = "ShareGPT"
        samples = json.load(open('../data/ShareGPT_V3_unfiltered_cleaned_split.json', 'r'))
        instructions = [f"{sample['conversations'][0]['value']}" for sample in samples if
                        len(sample['conversations']) > 1]
        responses = [f"{sample['conversations'][1]['value']}" for sample in samples if len(sample['conversations']) > 1]

    elif dataset_name == "oasst2":
        # Download OASST2 data from https://huggingface.co/datasets/OpenAssistant/oasst2/blob/main/2023-11-05_oasst2_all.trees.jsonl.gz
        dataset_title = "OASST2"
        samples = [json.loads(l) for l in open("../data/2023-11-05_oasst2_all.trees.jsonl", "r")]
        instructions = [f"{sample['prompt']['text']}"
                        for sample in samples if
                        len(sample['prompt']['replies']) > 0 and sample['prompt']['lang'] == "en"]
        responses = [f"{sample['prompt']['replies'][0]['text']}"
                     for sample in samples if len(sample['prompt']['replies']) > 0 and sample['prompt']['lang'] == "en"]

    elif dataset_name == "ultrachat":
        dataset_title = "UltraChat"
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
        ultrachat_train = dataset["train_sft"]
        instructions = [f"{sample['messages'][0]['content']}" for sample in ultrachat_train if
                        len(sample['messages']) > 1]
        responses = [f"{sample['messages'][1]['content']}" for sample in ultrachat_train if
                     len(sample['messages']) > 1]

    elif dataset_name == "deita_10k":
        dataset_title = "Deita 10K V0"
        dataset = load_dataset("hkust-nlp/deita-10k-v0")
        deita_train = dataset["train"]
        instructions = [sample['conversations'][0]['value']
                        for sample in deita_train if len(sample['conversations']) > 1]
        responses = [sample['conversations'][1]['value']
                     for sample in deita_train if len(sample['conversations']) > 1]

    elif dataset_name == "wizardlm_evol_instruct":
        dataset_title = "WizardLM Evol Instruct V2"
        dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k")
        wizardlm_train = dataset["train"]
        instructions = [sample['conversations'][0]['value']
                        for sample in wizardlm_train if len(sample['conversations']) > 1]
        responses = [sample['conversations'][1]['value']
                     for sample in wizardlm_train if len(sample['conversations']) > 1]

    elif dataset_name == "longform":
        dataset_title = "LongForm"
        dataset = load_dataset("akoksal/LongForm")
        longform_train = dataset["train"]
        instructions = [sample["input"] for sample in longform_train]
        responses = [sample["output"] for sample in longform_train]

    elif dataset_name == "gsm8k":
        dataset_title = "GSM8K"
        samples = [json.loads(l) for l in open("../data/gsm8k.jsonl", "r")]
        instructions = [f"{sample['instruction']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    elif dataset_name == "open_math_instruct":
        dataset_title = "MATH"
        samples = [json.loads(l) for l in open("../data/open_math_instruct.jsonl", "r")]
        instructions = [f"{sample['instruction']}" for sample in samples]
        responses = [f"{sample['output']}" for sample in samples]

    else:
        print("No dataset specification found (in utils.py)!")
        exit(0)

    return (dataset_title, instructions, responses)