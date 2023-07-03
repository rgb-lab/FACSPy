from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Literal, Optional, Union

from ..utils import subset_gate, find_gate_path_of_gate
from .utils import (prep_uns_dataframe,
                    select_gate_from_multiindex_dataframe,
                    select_gate_from_singleindex_dataframe,
                    scale_data,
                    map_obs_to_cmap,
                    calculate_metaclusters,
                    map_metaclusters_to_sample_ID,
                    merge_metaclusters_into_dataframe
                    )
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix

from ._frequency_plots import prep_dataframe_cluster_freq
from ..exceptions.exceptions import AnalysisNotPerformedError

def cluster_mfi(adata: AnnData,
                marker: Union[str, list[str]],
                groupby: Union[str, list[str]] = None,
                on: Literal["mfi", "fop", "gate_frequency"] = "mfi_c",
                colorby: Optional[str] = None,
                order: list[str] = None,
                gate: str = None,
                overview: bool = False,
                return_dataframe: bool = False) -> Optional[Figure]:

    try:
        data = adata.uns[on]
        data = select_gate_from_multiindex_dataframe(data.T, find_gate_path_of_gate(adata, gate))
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]

    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e

    data.index = data.index.set_names(["cluster", "gate"])
    raw_data = data.reset_index()

    sns.barplot(data = raw_data,
                x = "cluster",
                y = marker)
    plt.show()

def cluster_heatmap(adata: AnnData,
                    groupby: Optional[Union[str, list[str]]],
                    gate: str,
                    scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                    on: Literal["mfi", "fop", "gate_frequency"] = "mfi_c",
                    corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                    cluster_method: Literal["correlation", "distance"] = "distance",
                    cmap: str = "inferno",
                    return_fig: bool = False,
                    annotation_kwargs: dict = {},
                    metaclusters: Optional[int] = None,
                    label_metaclusters_in_dataset: bool = True,
                    label_metaclusters_key: Optional[str] = "metacluster_sc",
                    figsize: Optional[tuple[int, int]] = (5,3.8),
                    y_label_fontsize: Optional[Union[int, float]] = 4) -> Optional[Figure]:
    
    
    try:
        data = adata.uns[on]
        data = select_gate_from_multiindex_dataframe(data.T, find_gate_path_of_gate(adata, gate))
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]
        if scaling is not None:
            data[fluo_columns] = scale_data(data[fluo_columns], scaling)

    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e

    data.index = data.index.set_names(["cluster", "gate"])
    raw_data = data.reset_index()
    cluster_annot = raw_data["cluster"].astype("object").to_list()

    annotation_cmaps = ["Set1", "Set2", "tab10", "hls", "Paired"]
    sample_IDs = raw_data["cluster"].to_list()

    raw_data = raw_data[fluo_columns].T
    raw_data.columns = cluster_annot


    if cluster_method == "correlation":
        correlation_data = raw_data.corr(method = corr_method)
        col_linkage = hierarchy.linkage(distance.pdist(correlation_data.to_numpy()), method='average')
        row_linkage = hierarchy.linkage(distance.pdist(correlation_data.to_numpy().T), method='average')

    elif cluster_method == "distance":
        distance_data = distance_matrix(data[fluo_columns].to_numpy(), data[fluo_columns].to_numpy())
        row_linkage = hierarchy.linkage(distance.pdist(distance_data), method='average')
        col_linkage = row_linkage

    # if metaclusters is not None:
    #     metaclusters = calculate_metaclusters(col_linkage, n_clusters = metaclusters)
    #     metacluster_mapping = map_metaclusters_to_sample_ID(metaclusters, sample_IDs)
    #     data = merge_metaclusters_into_dataframe(data, metacluster_mapping)
    #     groupby += ["metacluster"]
    #     if label_metaclusters_in_dataset:
    #         if "metacluster" in adata.uns["metadata"].dataframe:
    #             print("warninig... overwriting metaclusters")
    #             adata.uns["metadata"].dataframe = adata.uns["metadata"].dataframe.drop(["metacluster"], axis = 1)
    #         adata.uns["metadata"].dataframe = pd.merge(adata.uns["metadata"].dataframe, data[["sample_ID", "metacluster"]], on = "sample_ID")
    #         if label_metaclusters_key is not None:
    #             adata.uns["metadata"].dataframe[label_metaclusters_key] = adata.uns["metadata"].dataframe["metacluster"]
    #             adata.uns["metadata"].dataframe = adata.uns["metadata"].dataframe.drop(["metacluster"], axis = 1)



    # fig = plt.figure(figsize = figsize)
    # gs = GridSpec(3,3)
    # gs.update(wspace = 0.015, hspace = 0.05)
    clustermap = sns.clustermap(
        data = raw_data,
        col_linkage = col_linkage,
        row_cluster = True,
        vmin = 0 if scaling is not None else None,
        vmax = 1 if scaling is not None else None,
        cmap = cmap,
        dendrogram_ratio = (0.1, 0.1),
        annot_kws = {"size": 4},
        figsize = figsize,
        cbar_kws = {"label": "scaled expression", "orientation": 'vertical'},
        yticklabels = True,
        xticklabels = True
    )
    
    cluster_freqs = prep_dataframe_cluster_freq(
        adata,
        groupby = annotation_kwargs.get("groupby", "sample_ID"),
        cluster_key = annotation_kwargs.get("cluster_key", "leiden"),
        normalize = annotation_kwargs.get("normalize", True),
    )
    indices = [t.get_text() for t in np.array(clustermap.ax_heatmap.get_xticklabels())]
    cluster_freqs = cluster_freqs.loc[indices]
    
    clustermap.fig.subplots_adjust(right=0.7)

    clustermap.ax_cbar.set_position([0.16, 0, 0.53, 0.02])
    clustermap.ax_cbar.set_position([0.75, 0.25, 0.02, 0.53])
    ax = clustermap.ax_heatmap
    ax.set_xticklabels([])
    ax.tick_params(left=False, bottom=False)
    ax.yaxis.set_ticks_position("left")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = y_label_fontsize)
    ax.set_ylabel("")
    clustermap.ax_row_dendrogram.set_visible(False)



    divider = make_axes_locatable(clustermap.ax_heatmap)
    ax3: Axes = divider.append_axes("bottom", size = "15%", pad = 0.02)

    cluster_freqs.plot(kind = "bar",
                       stacked = True,
                       legend = True,
                       ax = ax3,
                       subplots = False,
                       )
    ax3.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left")
    ax3.set_ylabel("Frequency")
    ax3.invert_yaxis()
    ax3.set_yticklabels(ax3.get_yticklabels(), fontsize = y_label_fontsize)


    if return_fig:
        return clustermap
    plt.show()






