import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches
import logomaker

def create_dna_logo(matrix, ax=None, highlights=[]):
    """
    Creates a DNA logo using the logomaker package.
    Arguments:
        `matrix`: an L x 4 array of values to plot, in ACGT order
        `ax`: Axes object on which to plot
        `highlights`: if given, a list of pairs, where each pair denotes the
            start and end indices within `L` of a rectangle to highlight; the
            end point is not included
    Returns the resulting Logo object.
    """
    data = pd.DataFrame(matrix, columns=["A", "C", "G", "T"])
    logo = logomaker.Logo(data, ax=ax)

    for start, end in highlights:
        top = np.max(np.sum(np.clip(matrix[start : end], 0, None), axis=1))
        bot = np.min(np.sum(np.clip(matrix[start : end], None, 0), axis=1))
        rect = matplotlib.patches.Rectangle(
            (start - 0.5, bot), end - start, top - bot,
            linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    return logo


def plot_motif_grid(
    motifs, titles=None, num_cols=2, same_y=True, show_x=True, show_y=True
):
    """
    Displays motifs in a grid of Logos.
    Arguments:
        `motifs`: a list of F motifs, where each motif is a W x 4 NumPy array
            (can be different Ws), or an F x W x 4 NumPy array
        `titles`: if provided, a list of F titles for each motif plot
        `num_cols`: number of columns to show motifs in grid
        `same_y`: if True, y-axis will be the same for all motifs in the grid
        `show_x`: if True, show x-tick labels for position
        `show_y`: if True, show y-tick labels for height
    Returns the figure object.
    """
    if titles:
        assert len(titles) == len(motifs)
    num_rows = int(np.ceil(len(motifs) / num_cols))
    fig, ax = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 4)
    )
    if type(ax) is plt.Axes:
        # If there is only one motif
        ax = np.array([ax])

    if len(ax.shape) == 1 and num_cols == 1:
        # If there is only one column
        ax = ax[:, None]
    elif len(ax.shape) == 1 and num_rows == 1:
        # If there is only one row
        ax = ax[None]
    
    if same_y: 
        min_height = np.min(np.concatenate([
            np.sum(np.minimum(motif, 0), axis=1) for motif in motifs
        ]))
        max_height = np.max(np.concatenate([
            np.sum(np.maximum(motif, 0), axis=1) for motif in motifs
        ]))
    ylims = (min_height, max_height)
    for motif_i in range(len(motifs)):
        i, j = motif_i % num_rows, motif_i // num_rows
        create_dna_logo(motifs[motif_i], ax=ax[i, j])
        if same_y:
            ax[i, j].set_ylim(ylims)
        if titles:
            ax[i, j].set_title(titles[motif_i])
        if not show_x:
            ax[i, j].set_xticks([])
            ax[i, j].set_xticklabels([])
        if not show_y:
            ax[i, j].set_yticks([])
            ax[i, j].set_yticklabels([])
    return fig
