import numpy as np

def create_violin_plot(ax, data, remove_outliers=False, colors=None):
    """
    Creates a violin plot on the given instantiated axes.
    Arguments:
        `ax`: MatPlotLib `Axes` object
        `data`: a list of N 1D NumPy arrays (each can be a different length),
            where each of the N arrays is a distribution to plot
        `remove_outliers`: if True, remove outliers in each distribution using
            the Outlier Rule before plotting
        `colors`: if True, a list of N colors, one for each distribution
    """
    num_dists = len(data)
    if colors:
        assert len(colors) == num_dists

    q1, med, q3 = np.stack([
        np.nanpercentile(vec, [25, 50, 70], axis=0) for vec in data
    ], axis=1)

    if remove_outliers:
        iqr = q3 - q1
        lower_outlier = q1 - (1.5 * iqr)
        upper_outlier = q3 + (1.5 * iqr)

        sorted_clipped_data = [
            np.sort(vec[(vec >= lower_outlier[i]) & (vec <= upper_outlier[i])])
            for i, vec in enumerate(data)
        ]
    else:
        sorted_clipped_data = [np.sort(vec) for vec in data]

    plot_parts = ax.violinplot(
        sorted_clipped_data, showmeans=False, showmedians=False,
        showextrema=False
    )
    violin_parts = plot_parts["bodies"]
    if colors:
        for i in range(num_dists):
            violin_parts[i].set_facecolor(colors[i])
            violin_parts[i].set_edgecolor(colors[i])

    inds = np.arange(1, num_dists + 1)
    ax.vlines(inds, q1, q3, color="black", linewidth=5, zorder=1)
    ax.scatter(inds, med, marker="o", color="white", s=30, zorder=2)
