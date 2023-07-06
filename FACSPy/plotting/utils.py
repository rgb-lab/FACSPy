from matplotlib.axes import Axes
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

import seaborn as sns

from anndata import AnnData

from typing import Literal, Union, Optional

from sklearn.preprocessing import MinMaxScaler, RobustScaler

from ..exceptions.exceptions import AnalysisNotPerformedError

from ..utils import find_gate_path_of_gate

from scipy.cluster.hierarchy import cut_tree
import scipy
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix

ANNOTATION_CMAPS = ["Set1", "Set2", "tab10", "hls", "Paired"]

def remove_ticks(ax: Axes,
                 which: Literal["x", "y", "both"]) -> None:
    if which == "x":
        ax.set_xticks([])
    if which == "y":
        ax.set_yticks([])
    if which == "both":
        ax.set_xticks([])
        ax.set_yticks([])

def remove_ticklabels(ax: Axes,
                      which: Literal["x", "y", "both"]) -> None:
    if which == "x":
        ax.set_xticklabels([])
    if which == "y":
        ax.set_yticklabels([])
    if which == "both":
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def label_metaclusters_in_dataset(adata: AnnData,
                                  data: pd.DataFrame,
                                  label_metaclusters_key: Optional[str] = None) -> None:
    if "metacluster" in adata.uns["metadata"].dataframe:
        print("warninig... overwriting metaclusters")
        adata.uns["metadata"].dataframe = adata.uns["metadata"].dataframe.drop(["metacluster"], axis = 1)
    adata.uns["metadata"].dataframe = pd.merge(adata.uns["metadata"].dataframe,
                                                data[["sample_ID", "metacluster"]],
                                                on = "sample_ID")
    if label_metaclusters_key is not None:
        adata.uns["metadata"].dataframe[label_metaclusters_key] = adata.uns["metadata"].dataframe["metacluster"]
        adata.uns["metadata"].dataframe = adata.uns["metadata"].dataframe.drop(["metacluster"], axis = 1)

def remove_dendrogram(clmap: sns.matrix.ClusterGrid,
                      which: Literal["x", "y", "both"]) -> None:
    if which == "x":
        clmap.ax_col_dendrogram.set_visible(False)
    if which == "y":
        clmap.ax_row_dendrogram.set_visible(False)
    if which == "both":
        clmap.ax_col_dendrogram.set_visible(False)
        clmap.ax_row_dendrogram.set_visible(False)

def add_metaclusters(adata: AnnData,
                     data: pd.DataFrame,
                     row_linkage: np.ndarray,
                     n_clusters: int,
                     sample_IDs: Union[pd.Index, pd.Series, list[int], list[str]],
                     label_metaclusters: bool,
                     label_metaclusters_key: str
                     ):
    metaclusters = calculate_metaclusters(row_linkage, n_clusters = n_clusters)
    metacluster_mapping = map_metaclusters_to_sample_ID(metaclusters, sample_IDs)
    data = merge_metaclusters_into_dataframe(data, metacluster_mapping)
    
    
    if label_metaclusters:
        label_metaclusters_in_dataset(adata = adata,
                                      data = data,
                                      label_metaclusters_key = label_metaclusters_key)
    data = data.set_index("sample_ID")
    return data


def calculate_linkage(dataframe: pd.DataFrame) -> np.ndarray:
    """calculates the linkage"""
    return hierarchy.linkage(distance.pdist(dataframe), method='average')

def calculate_sample_distance(dataframe: pd.DataFrame) -> np.ndarray:
    """ returns sample distance matrix of given dataframe"""
    return distance_matrix(dataframe.to_numpy(),
                           dataframe.to_numpy())

def remove_unused_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ handles the case where categorical variables are still there that are not present anymore """
    categorical_columns = [col for col in dataframe.columns if dataframe[col].dtype == "category"]
    for col in categorical_columns:
        dataframe[col] = dataframe[col].cat.remove_unused_categories()
    return dataframe

def extract_uns_dataframe(adata: AnnData,
                       data: pd.DataFrame,
                       column_identifier_name: str) -> pd.DataFrame:
    data = data.T
    data.index = data.index.set_names([column_identifier_name, "gate_path"])
    data = data.reset_index()
    data[column_identifier_name] = data[column_identifier_name].astype("str")
    return data

def append_metadata(adata: AnnData,
                    dataframe_to_merge: pd.DataFrame) -> pd.DataFrame:
    metadata = adata.uns["metadata"].to_df().copy()
    metadata["sample_ID"] = metadata["sample_ID"].astype("str")

    return remove_unused_categories(pd.merge(dataframe_to_merge, metadata, on = "sample_ID"))

def get_uns_dataframe(adata: AnnData,
                      gate: str,
                      table_identifier: str,
                      column_identifier_name: Literal["sample_ID", "cluster"]) -> pd.DataFrame:
    
    if table_identifier not in adata.uns:
        raise AnalysisNotPerformedError(table_identifier)
    
    data = adata.uns[table_identifier]
    data = extract_uns_dataframe(adata,
                                 data,
                                 column_identifier_name)
    data = select_gate_from_singleindex_dataframe(data, find_gate_path_of_gate(adata, gate))

    if column_identifier_name == "sample_ID":
        data = append_metadata(adata, data)
    
    data = data.set_index(column_identifier_name)
    return data


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
    return

def calculate_correlation_data(data: pd.DataFrame,
                               corr_method: Literal["pearson", "spearman", "kendall"]) -> pd.DataFrame:
    return data.corr(method = corr_method)

def add_annotation_plot(adata: AnnData,
                        annotate: str,
                        annot_frame: pd.DataFrame,
                        indices: list[Union[str, int]],
                        clustermap: sns.matrix.ClusterGrid,
                        y_label_fontsize: Optional[int],
                        y_label: Optional[str]
                        ) -> None:
    #TODO: add sampleID/clusterID on top?
    annot_frame = annot_frame.loc[indices]
    divider = make_axes_locatable(clustermap.ax_heatmap)
    ax: Axes = divider.append_axes("top", size = "20%", pad = 0.05)

    annot_frame.plot(kind = "bar",
                        stacked = annotate == "frequency",
                        legend = True,
                        ax = ax,
                        subplots = False,
                        )
    ax.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left")
    ax.set_ylabel(y_label)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = y_label_fontsize)
    remove_ticklabels(ax, which = "x")
    remove_ticks(ax, which = "x")
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.4)
    ax.set_xlabel("")

def add_categorical_legend_to_clustermap(clustermap: sns.matrix.ClusterGrid,
                                         heatmap: Axes,
                                         data: pd.DataFrame,
                                         groupby: list[str]) -> None:
    next_legend = 0
    for i, group in enumerate(groupby):
        group_lut = map_obs_to_cmap(data, group, ANNOTATION_CMAPS[i], return_mapping = True)
        handles = [Patch(facecolor = group_lut[name]) for name in group_lut]
        legend_space = 0.1 * (len(handles) + 1)
        group_legend = heatmap.legend(handles,
                                 group_lut,
                                 title = group,
                                 loc = "upper left",
                                 bbox_to_anchor = (1.02, 1 - next_legend, 0, 0),
                                 bbox_transform = heatmap.transAxes
                                 )

        next_legend += legend_space
        clustermap.fig.add_artist(group_legend)

def scale_cbar_to_heatmap(clustermap: sns.matrix.ClusterGrid,
                          heatmap_position: Bbox,
                          cbar_padding: Optional[float] = 0.7,
                          cbar_height: Optional[float] = 0.02,
                          loc: Literal["bottom", "right"] = "bottom") -> None:
    if loc == "bottom":
        clustermap.ax_cbar.set_position([heatmap_position.x0,
                                         heatmap_position.y0 * cbar_padding,
                                         heatmap_position.x1 - heatmap_position.x0,
                                         cbar_height])
    if loc == "right":
        clustermap.ax_cbar.set_position([heatmap_position.x1 * cbar_padding,
                                         heatmap_position.y0,
                                         cbar_height,
                                         heatmap_position.x1 - heatmap_position.x0])
    return

def map_obs_to_cmap(data: pd.DataFrame,
                    parameter_to_map: str,
                    cmap: str = "Set1",
                    return_mapping: bool = False,
                    as_series: bool = True) -> dict[str, tuple[float, float, float]]:
    obs = data[parameter_to_map].unique()
    cmap = sns.color_palette(cmap, len(obs))
    mapping = {obs_entry: cmap[i] for i, obs_entry in enumerate(obs)}
    if return_mapping:
        return mapping
    if as_series:
        return pd.Series(data[parameter_to_map].astype("object").map(mapping), name = parameter_to_map)
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