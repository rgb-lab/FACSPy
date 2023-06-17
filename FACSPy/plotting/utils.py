from matplotlib.axes import Axes
import numpy as np
import pandas as pd

import seaborn as sns

from anndata import AnnData

from matplotlib.axis import Axis

def prep_uns_dataframe(adata: AnnData,
                       data: pd.DataFrame) -> pd.DataFrame:
    data = data.T
    data.index = data.index.set_names(["sample_ID", "gate_path"])
    data = data.reset_index()
    data["sample_ID"] = data["sample_ID"].astype("int64")
    return append_metadata(adata, data)


def map_obs_to_cmap(data: pd.DataFrame,
                    parameter_to_map: str,
                    cmap: str = "Set1") -> dict[str, tuple[float, float, float]]:
    obs = data[parameter_to_map].unique()
    cmap = sns.color_palette(cmap, len(obs))
    mapping = {obs_entry: cmap[i] for i, obs_entry in enumerate(obs)}
    return data[parameter_to_map].map(mapping)

def append_metadata(adata: AnnData,
                    dataframe_to_merge: pd.DataFrame) -> pd.DataFrame:
    metadata = adata.uns["metadata"].to_df()
    return pd.merge(dataframe_to_merge, metadata, on = "sample_ID")

def create_boxplot(ax: Axis,
                   grouping: str,
                   plot_params: dict) -> Axis:
    
    if grouping is None or grouping == "sample_ID":
        sns.barplot(**plot_params,
                    ax = ax)
    
    else:
        sns.stripplot(**plot_params,
                      dodge = True,
                      jitter = True,
                      linewidth = 1,
                      ax = ax)
        sns.boxplot(**plot_params,
                    boxprops = dict(facecolor = "white"),
                    whis = (0,100),
                    ax = ax)
    
    return ax

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