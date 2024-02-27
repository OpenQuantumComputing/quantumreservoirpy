import numpy as np
import matplotlib.pyplot as plt


def state_plotter(x, target):
    n_features = len(x[0])
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        figsize=(10 * n_cols, 10 * n_rows),
        sharex="col",
        sharey="row",
    )
    cmap = plt.get_cmap("jet", len(np.unique(target)))

    try:
        axes = axes.flatten()
    except:
        axes = [axes]

    for i, t in enumerate(np.unique(target)):
        mask = t == target
        for idx in range(n_features):
            ax = axes[idx]
            ax.plot(
                np.arange(len(x))[mask],
                x[mask][:, idx],
                marker="o",
                lw=0,
                label=f"{t}",
                c=cmap(i),
            )
            ax.set_title(f"Feature {idx+1}")
            ax.legend(loc="upper right")
    return fig, axes


def feature_plotter(x, target):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("jet", len(np.unique(target)))
    ax.scatter(x[:, 0], x[:, 1], cmap=cmap, c=target)
    ax.set_xlabel("state 1")
    ax.set_ylabel("state 2")
    return fig, ax
