from utils import load_sft_dataset
import argparse
import os
import glob
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="The dataset name. The default is 'all', all datasets will be processed.",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        help="The analysis type. The default is 'all', all analysis will be conducted.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=16,
        help="The number of requests to send to LLM at a time."
    )
    parser.add_argument(
        "--repeat_analysis",
        action="store_true",
        help="Whether to force repeat running the analyzer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Where to store the analysis results."
    )
    parser.set_defaults(repeat_analysis=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dataset_names = ["sigma_v1", "sigma_v2_evol", "sigma_v2_evol_math", "sigma_v3", "self_instruct_mix_v1",
                     "deita_10k", "no_robots", "lima", "longform",
                     "flan_v2_cot", "dolly_15k",
                     "alpaca", "alpaca_gpt4",
                     "bactrian-x_en", "wizardlm_evol_instruct", "wizardlm_orca",
                     "sharegpt", "oasst2", "ultrachat"]
    if args.dataset != "all": dataset_names = args.dataset.split(",")

    analysis_types = ["categories", "complexity", "quality", "tagging", "tokens", "embeddings", "reward_modelling"]
    if args.analysis != "all": analysis_types = args.analysis.split(",")

    for analysis_type in analysis_types:
        analysis_type = analysis_type.strip()

        if analysis_type == "complexity":
            from analyzer.complexity_scorer import ComplexityScorer
            analyzer = ComplexityScorer()
            output_dir = os.path.join(args.output_dir, "./complexity_scores")

        elif analysis_type == "quality":
            from analyzer.quality_scorer import QualityScorer
            analyzer = QualityScorer()
            output_dir = os.path.join(args.output_dir, "./quality_scores")

        elif analysis_type == "tagging":
            from analyzer.instruction_tagger import InsTagger
            analyzer = InsTagger()
            output_dir = os.path.join(args.output_dir, "./instagger")

        elif analysis_type == "tokens":
            from analyzer.token_counter import TokenCounter
            analyzer = TokenCounter()
            output_dir = os.path.join(args.output_dir, "./token_length")

        elif analysis_type == "embeddings":
            from analyzer.embedding_distance import EmbeddingDistance
            analyzer = EmbeddingDistance()
            output_dir = os.path.join(args.output_dir, "./embedding_distance")

        elif analysis_type == "reward_modelling":
            from analyzer.reward_modeller import RewardModeller
            analyzer = RewardModeller(from_scratch=True)
            output_dir = os.path.join(args.output_dir, "./reward_scores")

        elif analysis_type == "categories":
            from analyzer.instruction_classifier import InsClassifier
            analyzer = InsClassifier()
            output_dir = os.path.join(args.output_dir, "./categories")

        else:
            print("No analysis type found!")
            exit(0)

        try:
            os.makedirs(output_dir)
        except FileExistsError:
            # directory already exists
            pass

        if analyzer:

            for dataset_name in dataset_names:
                dataset_name = dataset_name.strip()

                # Load SFT Dataset
                (dataset_title, instructions, responses) = load_sft_dataset(dataset_name)

                # Check if the analysis result exist already
                result_exists = False
                if glob.glob(f"{output_dir}/{dataset_name}.*") \
                        or glob.glob(f"{output_dir}/{dataset_name}_instructions.*") \
                        or glob.glob(f"{output_dir}/{dataset_name}_responses.*"):
                    result_exists = True
                    if analysis_type == "complexity":
                        with open(f"{output_dir}/{dataset_name}.csv") as fcompl:
                            results = [float(line.strip()) for line in fcompl.readlines()]
                    elif analysis_type == "quality":
                        with open(f"{output_dir}/{dataset_name}.csv") as fqual:
                            results = [float(line.strip()) for line in fqual.readlines()]
                    elif analysis_type == "embeddings":
                        with open(f"{output_dir}/{dataset_name}.csv") as fembed:
                            results = [float(line.strip()) for line in fembed.readlines()]
                    elif analysis_type == "reward_modelling":
                        with open(f"{output_dir}/{dataset_name}.csv") as freward:
                            results = [float(line.strip()) for line in freward.readlines()]
                    elif analysis_type == "tagging":
                        with open(f"{output_dir}/{dataset_name}.jsonl") as ftags:
                            results = [len(json.loads(line)) for line in ftags.readlines()]
                    elif analysis_type == "categories":
                        with open(f"{output_dir}/{dataset_name}.csv") as fcats:
                            results = [line.strip() for line in fcats.readlines()]
                    elif analysis_type == "tokens":
                        with open(f"{output_dir}/{dataset_name}_instructions.csv") as finst, \
                                open(f"{output_dir}/{dataset_name}_responses.csv") as fresp:
                            results = {
                                "instruction_length": [int(line.strip()) for line in finst.readlines()],
                                "response_length": [int(line.strip()) for line in fresp.readlines()]
                            }

                # Categories, if exists
                categories = None
                if glob.glob(f"{os.path.join(args.output_dir, './categories')}/{dataset_name}.csv"):
                    with open(f"{os.path.join(args.output_dir, './categories')}/{dataset_name}.csv") as fcats:
                        categories = [line.strip() for line in fcats.readlines()]

                if not result_exists or args.repeat_analysis:
                    print(f"Run {analysis_type} on {dataset_title}...")
                    # Run the analyzer
                    results = analyzer.run(instructions, responses, dataset_name, dataset_title, output_dir, args.request_batch_size)

                # Plot the charts
                print(f"Plot {analysis_type} on {dataset_title}...")
                analyzer.plot(results, dataset_name, dataset_title, output_dir, categories)



