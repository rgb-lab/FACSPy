from matplotlib.axes import Axes
import numpy as np
import pandas as pd

def calculate_nrows(ncols: int, 
                    dataset: pd.DataFrame):
    return int(
            np.ceil(
                len(dataset.columns)/ncols
            )
        )

def calculate_fig_size(ncols: int,
                       nrows: int,
                       groupby_list: list = None) -> tuple[int, int]:
    
    x_dim_scale_factor = (1 + (0.07 * len(groupby_list))) if groupby_list is not None else 1
    x_dimension = 2 * ncols * x_dim_scale_factor
    y_dimension = 1.5 * nrows if groupby_list is None else 1.8 * nrows
    return (x_dimension, y_dimension)

def turn_off_missing_plot(ax: Axes) -> Axes:
    ax.axis("off")
    return ax

def turn_off_missing_plots(ax: Axes) -> Axes:
    for axs in ax:
        if not axs.lines:
            turn_off_missing_plot(axs)
    return ax