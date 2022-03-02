import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def plot_spearman(pred: np.ndarray, y: np.ndarray, filename: str):

    sr, _ = scipy.stats.spearmanr(pred, y)
    pr, _ = scipy.stats.pearsonr(pred, y)

    _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y, pred, s=0.1)

    log_scale = False
    
    if log_scale:
        ax.set_xlim([1e-1, 30])
        ax.set_ylim([1e-1, 30])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1e-1, 1, 10])
        ax.set_yticks([1e-1, 1, 10])
    else:
        # ax.plot([-1, 31], [-1, 31])
        ax.set_xlim([-1, 31])
        ax.set_ylim([-1, 31])
        ax.set_xticks(range(0, 35, 5))
        ax.set_yticks(range(0, 35, 5))

    ax.set_title("Evaluation of DeepPE2")
    ax.set_xlabel("Measured PE2 efficiency (%)")
    ax.set_ylabel("DeepPE prediction score (%)")

    ax.annotate('R = {:.4f}\nr = {:.4f}\nn = {}'.format(sr, pr, len(y)),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')

    plt.savefig(filename, bbox_inches="tight", dpi=600)
    plt.close()

def plot_rank(pred: np.ndarray, y: np.ndarray, filename: str):

    sr, _ = scipy.stats.spearmanr(pred, y)
    pr, _ = scipy.stats.pearsonr(pred, y)

    _, ax = plt.subplots(figsize=(6, 6))

    pred = np.argsort(pred)
    y = np.argsort(y)

    print(pred[:10])

    print(y[:10])

    ax.scatter(y, pred, s=0.1)

    ax.set_title("Evaluation of DeepPE2")
    ax.set_xlabel("Measured PE2 efficiency (rank)")
    ax.set_ylabel("DeepPE prediction score (rank))")

    ax.annotate('R = {:.4f}\nr = {:.4f}\nn = {}'.format(sr, pr, len(y)),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')

    plt.savefig(filename, bbox_inches="tight", dpi=600)
    plt.close()
