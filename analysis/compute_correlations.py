import argparse
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import scipy.stats
import json
import random
import os

def compute_correlations(title: str, filename: str, x_array: List, y_array: List, x_label: str, y_label: str, xmax: int, ymax: int):
    # Fit linear regression via least squares with numpy.polyfit
    # It returns an slope (b) and intercept (a)
    # deg=1 means linear fit (i.e. polynomial of degree 1)
    coef = np.polyfit(x_array, y_array, deg=1)
    poly1d_fn = np.poly1d(coef)

    # Scatter plot for instruction length and complexity, and regression line
    plt.figure()
    plt.plot(x_array, y_array, 'yo', x_array, poly1d_fn(x_array), '--k')
    if xmax > 0: plt.xlim(0, xmax)
    if ymax > 0: plt.ylim(0, ymax)

    # Add the correlation coefficient in the title
    plt.title(title)

    # Add labels for x and y axis
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([], [], ' ', label='Pearson correlation=' + str(
        round(scipy.stats.pearsonr(x_array, y_array)[0], 2)))
    plt.plot([], [], ' ', label='Spearman correlation=' + str(
        round(scipy.stats.spearmanr(x_array, y_array)[0], 2)))
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="The dataset name. The default is 'all', all datasets will be processed.",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="./",
        help="Where to store the analysis results."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dataset_names = [
        "sigma_v1",
        # "sigma_10k_regenerated",
        # "llama_sigma2_10k",
        # "llama_sigma2_10k_regenerated",
        "sigma_v1_regen", "sigma_v2_regen",
        "sigma_v2", "sigma_v2_evol", "sigma_v2_evol_math", "sigma_v3",
        "deita_10k", "no_robots", "flan_v2_cot", "dolly_15k", "longform",
        # "alpaca",
        "alpaca_gpt4", # "lima",
        "bactrian-x_en", "wizardlm_evol_instruct",
        # "wizardlm_orca",
        "sharegpt", "oasst2", "ultrachat"
    ]
    if args.dataset != "all": dataset_names = args.dataset.split(",")

    inst_length = []
    resp_length = []
    complexity = []
    quality = []
    num_tags = []

    dataset_size = []

    tags_x = []
    tags_y = []
    tags_x_10k = []

    avg_complexity = []
    avg_quality = []
    avg_embedding = []

    for i, dataset_name in enumerate(dataset_names):
        dataset_name = dataset_name.strip()

        unique_tags = {}
        unique_tags_10k = {}

        with open(f"{os.path.join(args.analysis_dir, 'token_length')}/{dataset_name}_instructions.csv") as finst_len, \
                open(f"{os.path.join(args.analysis_dir, 'token_length')}/{dataset_name}_responses.csv") as fresp_len, \
                open(f"{os.path.join(args.analysis_dir, 'complexity_scores')}/{dataset_name}.csv") as fcomplexity, \
                open(f"{os.path.join(args.analysis_dir, 'quality_scores')}/{dataset_name}.csv") as fquality, \
                open(f"{os.path.join(args.analysis_dir, 'instagger')}/{dataset_name}.jsonl") as ftags, \
                open(f"{os.path.join(args.analysis_dir, 'embedding_distance')}/{dataset_name}.csv") as fembedding:

            inst_length += [int(line.strip()) for line in finst_len.readlines()]
            resp_length += [int(line.strip()) for line in fresp_len.readlines()]
            compl = [float(line.strip()) for line in fcomplexity.readlines()]
            complexity += compl
            qual = [float(line.strip()) for line in fquality.readlines()]
            quality += qual
            tags_per_line = ftags.readlines()
            num_tags += [len(json.loads(line.strip())) for line in tags_per_line]
            emb = [float(line.strip()) for line in fembedding.readlines()]

            for line in tags_per_line:
                tags = json.loads(line.strip())
                for tag in tags:
                    if 'tag' in tag:
                        if tag['tag'] not in unique_tags: unique_tags[tag['tag']] = 0
                        unique_tags[tag['tag']] += 1

            sampled_tags_per_line = tags_per_line
            if len(tags_per_line) > 10000: sampled_tags_per_line = random.sample(tags_per_line, 10000)
            for line in sampled_tags_per_line:
                tags = json.loads(line.strip())
                for tag in tags:
                    if 'tag' in tag:
                        if tag['tag'] not in unique_tags_10k: unique_tags_10k[tag['tag']] = 0
                        unique_tags_10k[tag['tag']] += 1

        num_unique_tags = len(unique_tags) / len(tags_per_line)
        num_unique_tags_10k = len(unique_tags_10k) / len(sampled_tags_per_line)
        avg_tag_usage = 1 / (np.average(np.array([(v / len(tags_per_line) * 100) for k,v in unique_tags.items()])))

        tags_x.append(num_unique_tags)
        tags_y.append(avg_tag_usage)
        tags_x_10k.append(num_unique_tags_10k)

        avg_complexity.append(np.mean(compl))
        avg_quality.append(np.mean(qual))
        avg_embedding.append(np.mean(emb))

        dataset_size.append(len(tags_per_line))

        print(dataset_name, len(tags_per_line), num_unique_tags, avg_tag_usage)

    # Plot complexity/quality...
    plt.figure()
    plt.figure(figsize=(18, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(dataset_names)))
    for i, dataset_name in enumerate(dataset_names):
        if "sigma" in dataset_name:
            plt.text((avg_complexity[i] * (1 - 0.01)), (avg_quality[i] * (1 - 0.01)), dataset_name, fontsize=12,
                     fontweight="bold")
        else:
            plt.text((avg_complexity[i] * (1 - 0.01)), (avg_quality[i] * (1 - 0.01)), dataset_name, fontsize=12)

    avg_complexity = np.array(avg_complexity)
    avg_quality = np.array(avg_quality)
    plt.title("Complexity / Quality", fontsize=18)
    plt.xlabel("Avg. Complexity Scores", fontsize=18)
    plt.ylabel("Avg. Quality Scores", fontsize=18)
    plt.xlim([1, 4])
    plt.ylim([2, 5])
    plt.scatter(avg_complexity, avg_quality, c=colors, alpha=0.5, s=[(s // 100) * 2 for s in dataset_size])
    plt.savefig("correlations/avg_complexity_quality.png", bbox_inches="tight")

    # Plot #InsTag diversity...
    plt.figure()
    plt.figure(figsize=(18, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(dataset_names)))
    for i, dataset_name in enumerate(dataset_names):
        if "sigma" in dataset_name:
            plt.text((tags_x[i] * (1 - 0.01)), (tags_y[i] * (1 - 0.01)), dataset_name, fontsize=12, fontweight="bold")
        else:
            plt.text((tags_x[i] * (1 - 0.01)), (tags_y[i] * (1 - 0.01)), dataset_name, fontsize=12)

    tags_x = np.array(tags_x)
    tags_y = np.array(tags_y)
    plt.title("#InsTag diversity", fontsize=18)
    plt.xlabel("Num Unique #InsTags / Dataset Size", fontsize=18)
    plt.ylabel("1 / Avg. Tag Frequency", fontsize=18)
    plt.scatter(tags_x, tags_y, c=colors, alpha=0.5, s=[(s // 100)*2 for s in dataset_size])
    plt.savefig("correlations/num_unique_tags.png", bbox_inches="tight")

    # Plot #InsTag diversity (on 10k randomly sampled)...
    plt.figure()
    plt.figure(figsize=(18, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(dataset_names)))
    for i, dataset_name in enumerate(dataset_names):
        if "sigma" in dataset_name:
            plt.text((tags_x_10k[i] * (1 - 0.01)), (tags_y[i] * (1 - 0.01)), dataset_name, fontsize=12, fontweight="bold")
        else:
            plt.text((tags_x_10k[i] * (1 - 0.01)), (tags_y[i] * (1 - 0.01)), dataset_name, fontsize=12)

    tags_x_10k = np.array(tags_x_10k)
    tags_y = np.array(tags_y)
    plt.title("#InsTag diversity", fontsize=18)
    plt.xlabel("Num Unique #InsTags (per 10k samples)", fontsize=18)
    plt.ylabel("1 / Avg. Tag Frequency", fontsize=18)
    plt.scatter(tags_x_10k, tags_y, c=colors, alpha=0.5, s=[(s // 100) * 2 for s in dataset_size])
    plt.savefig("correlations/num_unique_tags_10k.png", bbox_inches="tight")

    # Plot #InsTag / embedding diversity...
    plt.figure()
    plt.figure(figsize=(18, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(dataset_names)))
    for i, dataset_name in enumerate(dataset_names):
        if "sigma" in dataset_name:
            plt.text((tags_x[i] * (1 - 0.01)), (avg_embedding[i] * (1 - 0.01)), dataset_name, fontsize=12, fontweight="bold")
        else:
            plt.text((tags_x[i] * (1 - 0.01)), (avg_embedding[i] * (1 - 0.01)), dataset_name, fontsize=12)


    tags_x = np.array(tags_x)
    avg_embedding = np.array(avg_embedding)
    plt.title("#InsTag / Embedding diversity", fontsize=18)
    plt.xlabel("Num Unique #InsTags / Dataset Size", fontsize=18)
    plt.ylabel("Avg. Embedding Distance", fontsize=18)
    # plt.ylim([0, 1])
    plt.scatter(tags_x, avg_embedding, c=colors, alpha=0.5, s=[(s // 100)*2 for s in dataset_size])
    plt.savefig("correlations/num_unique_tags_avg_embedding.png", bbox_inches="tight")

    # Plot #InsTag (on 10k randomly sampled) / embedding diversity...
    plt.figure()
    plt.figure(figsize=(18, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(dataset_names)))
    for i, dataset_name in enumerate(dataset_names):
        if "sigma" in dataset_name:
            plt.text((tags_x_10k[i] * (1 - 0.01)), (avg_embedding[i] * (1 - 0.01)), dataset_name, fontsize=12,
                     fontweight="bold")
        else:
            plt.text((tags_x_10k[i] * (1 - 0.01)), (avg_embedding[i] * (1 - 0.01)), dataset_name, fontsize=12)

    tags_x_10k = np.array(tags_x_10k)
    avg_embedding = np.array(avg_embedding)
    plt.title("#InsTag / Embedding diversity", fontsize=18)
    plt.xlabel("Num Unique #InsTags (per 10k samples)", fontsize=18)
    plt.ylabel("Avg. Embedding Distance", fontsize=18)
    # plt.ylim([0, 1])
    plt.scatter(tags_x_10k, avg_embedding, c=colors, alpha=0.5, s=[(s // 100) * 2 for s in dataset_size])
    plt.savefig("correlations/num_unique_tags_10k_avg_embedding.png", bbox_inches="tight")



    # inst_length = np.array(inst_length)
    # resp_length = np.array(resp_length)
    # complexity = np.array(complexity)
    # quality = np.array(quality)
    # num_tags = np.array(num_tags)
    #
    # # Compute correlations...
    # compute_correlations("Instruction Length - Complexity", "correlations/instlen_complexity.png",
    #                      inst_length, complexity, "Instruction Length", "Complexity", 4096, 6)
    # compute_correlations("Response Length - Quality", "correlations/resplen_quality.png",
    #                      resp_length, quality, "Response Length", "Quality", 4096, 6)
    # compute_correlations("Num Tags - Complexity", "correlations/numtag_complexity.png",
    #                      num_tags, complexity, "Num Tags", "Complexity", 20, 6)