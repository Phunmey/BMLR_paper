import arviz as az
import numpy as np


def traceplotting(X):
    axes = az.plot_trace(X, var_names=['betaI', 'betaP'], divergences=None, show=False)
    axes[1, 0].set_xlabel("Sample Values")
    axes[1, 1].set_xlabel("Number of Samples")
    axes[0, 0].set_ylabel("Density")
    axes[0, 1].set_ylabel("Sample Values")
    axes[1, 0].set_ylabel("Density")
    axes[1, 1].set_ylabel("Sample Values")

    return


def posterior_predictiveplotting(X):
    ax = az.plot_ppc(X, group='posterior', kind='kde', figsize=(6, 5))

    ax.set_ylim(0, 0.7)
    ax.set_yticks(np.arange(0, 0.7, 0.1))
    ax.set_ylabel("Probability", fontsize=16)
    ax.set_xlabel("Status", fontsize=16)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["Dropout", "Enrolled", "Graduate"], fontsize=12)
    ax.legend(fontsize=12)  # Increase the size of the legend

    return
