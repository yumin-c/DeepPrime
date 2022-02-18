import matplotlib.pyplot as plt
import numpy as np
import scipy

def plot_spearman(pred: np.ndarray, y: np.ndarray, filename: str):

    corr = scipy.stats.spearmanr(pred, y).correlation

    _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y, pred, s=0.1)
    ax.set_xlim([-1, 31])
    ax.set_ylim([-1, 31])
    ax.set_xticks(range(0, 35, 5))
    ax.set_yticks(range(0, 35, 5))

    ax.set_title("Evaluation of DeepPE2")
    ax.set_xlabel("Measured PE2 efficiency (%)")
    ax.set_ylabel("DeepPE prediction score (%)")

    ax.annotate('R = {:.4f}'.format(corr),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')

    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()