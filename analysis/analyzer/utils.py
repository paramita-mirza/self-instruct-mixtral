from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_histogram(filename: str, title: str, scores: List[float], min_ylim: float=1.0, max_ylim: float=6.0):
    scores = np.array(scores)
    plt.figure()
    if min_ylim > 0 and max_ylim > 0:
        plt.hist(scores, bins="auto", range=(min_ylim, max_ylim))  # arguments are passed to np.histogram
    else:
        plt.hist(scores, bins="auto")
    plt.axvline(scores.mean(), color='k', linestyle='dashed', linewidth=1)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(scores.mean() * 1.1, ymax * 0.9, 'Mean: {:.2f}'.format(scores.mean()))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")

def plot_histogram_per_category(filename: str, title: str, scores: List[float], categories: List[str]):
    scores_per_category = {
        'score': scores,
        'category': categories,
    }
    df = pd.DataFrame(scores_per_category)
    plt.figure()
    sns.histplot(data=df, x='score', bins=200, hue='category', multiple="stack")
    scores = np.array(scores)
    plt.axvline(scores.mean(), color='k', linestyle='dashed', linewidth=1)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(scores.mean() * 1.05, ymax * 0.9, 'Mean: {:.2f}'.format(scores.mean()))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")

def plot_categories(filename: str, title: str, x: List, labels: List, colors: Dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(x, labels=labels, autopct='%.1f%%',
           wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
           textprops={'size': 'small'},
           colors=[colors[key] for key in labels])
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")