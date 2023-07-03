from matplotlib.axes import Axes
import numpy as np
import pandas as pd

import seaborn as sns

from anndata import AnnData

from typing import Literal

from sklearn.preprocessing import MinMaxScaler, RobustScaler

from scipy.cluster.hierarchy import cut_tree

def remove_unused_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ handles the case where categorical variables are still there that are not present anymore """
    categorical_columns = [col for col in dataframe.columns if dataframe[col].dtype == "category"]
    for col in categorical_columns:
        dataframe[col] = dataframe[col].cat.remove_unused_categories()
    return dataframe

def prep_uns_dataframe(adata: AnnData,
                       data: pd.DataFrame) -> pd.DataFrame:
    data = data.T
    data.index = data.index.set_names(["sample_ID", "gate_path"])
    data = data.reset_index()
    data["sample_ID"] = data["sample_ID"].astype("int64")
    return append_metadata(adata, data)

def select_gate_from_multiindex_dataframe(dataframe: pd.DataFrame,
                               gate: str) -> pd.DataFrame:
    return dataframe.loc[(slice(None), gate), :]

def select_gate_from_singleindex_dataframe(dataframe: pd.DataFrame,
                               gate: str) -> pd.DataFrame:
    return dataframe[dataframe["gate_path"] == gate]

def scale_data(dataframe: pd.DataFrame,
               scaling: Literal["MinMaxScaler", "RobustScaler"]) -> np.ndarray:
    if scaling == "MinMaxScaler":
        return MinMaxScaler().fit_transform(dataframe)
    if scaling == "RobustScaler":
        return RobustScaler().fit_transform(dataframe)

def map_obs_to_cmap(data: pd.DataFrame,
                    parameter_to_map: str,
                    cmap: str = "Set1",
                    return_mapping: bool = False) -> dict[str, tuple[float, float, float]]:
    obs = data[parameter_to_map].unique()
    cmap = sns.color_palette(cmap, len(obs))
    mapping = {obs_entry: cmap[i] for i, obs_entry in enumerate(obs)}
    if return_mapping:
        return mapping
    return data[parameter_to_map].astype("object").map(mapping)

def calculate_metaclusters(linkage: np.ndarray,
                           n_clusters: int) -> dict[int: list[int]]:
    ### stackoverflow https://stackoverflow.com/questions/65034792/print-all-clusters-and-samples-at-each-step-of-hierarchical-clustering-in-python
    linkage_matrix = linkage
    clusters = cut_tree(linkage_matrix, n_clusters=n_clusters)
    # transpose matrix
    clusters = clusters.T
    for row in clusters[::-1]:
        # create empty dictionary
        groups = {}
        for i, g in enumerate(row):
            if g not in groups:
                # add new key to dict and assign empty set
                groups[g] = set([])
            # add to set of certain group
            groups[g].add(i)

    return groups

def map_metaclusters_to_sample_ID(metaclusters: dict,
                                  sample_IDs: list) -> pd.DataFrame:
    sample_IDs = pd.DataFrame(sample_IDs, columns = ["sample_ID"])
    for i, sample_ID in enumerate(sample_IDs["sample_ID"].to_list()):
        sample_IDs.loc[sample_IDs["sample_ID"] == sample_ID, "metacluster"] = int([metacluster
                                                                                   for metacluster in metaclusters
                                                                                   if i in metaclusters[metacluster]][0])
    
    return sample_IDs

def merge_metaclusters_into_dataframe(data, metacluster_mapping) -> pd.DataFrame:
    if "metacluster" in data.columns:
        data = data.drop(["metacluster"], axis = 1)
    return pd.merge(data, metacluster_mapping, on = "sample_ID")

def append_metadata(adata: AnnData,
                    dataframe_to_merge: pd.DataFrame) -> pd.DataFrame:
    metadata = adata.uns["metadata"].to_df()
    return remove_unused_categories(pd.merge(dataframe_to_merge, metadata, on = "sample_ID"))

def create_boxplot(ax: Axes,
                   grouping: str,
                   plot_params: dict) -> Axes:
    
    if grouping is None or grouping == "sample_ID":
        sns.barplot(**plot_params,
                    ax = ax)
    
    else:
        sns.stripplot(**plot_params,
                      dodge = False,
                      jitter = True,
                      linewidth = 1,
                      ax = ax)
        plot_params["hue"] = None
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
        if not axs.lines and not axs.collections:
            turn_off_missing_plot(axs)
    return ax