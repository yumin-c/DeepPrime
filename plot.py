import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_spearman(pred = None, y = None, filename = None, title = "Evaluation of DeepPE2"):

    sr, _ = stats.spearmanr(pred, y)
    pr, _ = stats.pearsonr(pred, y)

    _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y, pred, s=0.1)
    
    ax.set_xlim([-1, 51])
    ax.set_ylim([-1, 51])
    ax.set_xticks(range(0, 55, 5))
    ax.set_yticks(range(0, 55, 5))

    ax.set_title(title)
    ax.set_xlabel("Measured PE2 efficiency (%)")
    ax.set_ylabel("DeepPE prediction score (%)")

    ax.annotate('R = {:.4f}\nr = {:.4f}\nn = {}'.format(sr, pr, len(y)),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')
    
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=600)
    return plt