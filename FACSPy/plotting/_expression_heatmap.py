from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib.patches import Patch

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

from ..exceptions.exceptions import AnalysisNotPerformedError

def expression_heatmap(adata: AnnData,
                       groupby: Optional[Union[str, list[str]]],
                       gate: str,
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cluster_method: Literal["correlation", "distance"] = "distance",
                       cmap: str = "inferno",
                       return_fig: bool = False,
                       metaclusters: Optional[int] = None,
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc",
                       figsize: Optional[tuple[int, int]] = (5,3.8),
                       y_label_fontsize: Optional[Union[int, float]] = 4) -> Optional[Figure]:
    
    try:
        if on == "all_cells":
            adata = subset_gate(adata, gate, as_view = True)
            data = adata.to_df(layer = "transformed")

        else:
            data = adata.uns[on]
            data = prep_uns_dataframe(adata, data)
            data = select_gate_from_singleindex_dataframe(data, find_gate_path_of_gate(adata, gate))
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]
        if scaling is not None:
            data[fluo_columns] = scale_data(data[fluo_columns], scaling)
    
    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e


    if not isinstance(groupby, list):
        groupby = [groupby]

    #fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5,5))
    annotation_cmaps = ["Set1", "Set2", "tab10", "hls", "Paired"]
    sample_IDs = data["sample_ID"].to_list()

    raw_data = data[fluo_columns].T

    if cluster_method == "correlation":
        correlation_data = raw_data.corr(method = corr_method)
        col_linkage = hierarchy.linkage(distance.pdist(correlation_data.to_numpy()), method='average')
        row_linkage = hierarchy.linkage(distance.pdist(correlation_data.to_numpy().T), method='average')
    
    elif cluster_method == "distance":
        distance_data = distance_matrix(data[fluo_columns].to_numpy(), data[fluo_columns].to_numpy())
        row_linkage = hierarchy.linkage(distance.pdist(distance_data), method='average')
        col_linkage = row_linkage

    if metaclusters is not None:
        metaclusters = calculate_metaclusters(col_linkage, n_clusters = metaclusters)
        metacluster_mapping = map_metaclusters_to_sample_ID(metaclusters, sample_IDs)
        data = merge_metaclusters_into_dataframe(data, metacluster_mapping)
        groupby += ["metacluster"]
        if label_metaclusters_in_dataset:
            if "metacluster" in adata.uns["metadata"].dataframe:
                print("warninig... overwriting metaclusters")
                adata.uns["metadata"].dataframe = adata.uns["metadata"].dataframe.drop(["metacluster"], axis = 1)
            adata.uns["metadata"].dataframe = pd.merge(adata.uns["metadata"].dataframe, data[["sample_ID", "metacluster"]], on = "sample_ID")
            if label_metaclusters_key is not None:
                adata.uns["metadata"].dataframe[label_metaclusters_key] = adata.uns["metadata"].dataframe["metacluster"]
                adata.uns["metadata"].dataframe = adata.uns["metadata"].dataframe.drop(["metacluster"], axis = 1)


    clustermap = sns.clustermap(
        data = raw_data,
        col_colors = [
            map_obs_to_cmap(data, group, annotation_cmaps[i])
            for i, group in enumerate(groupby)
        ],
        col_linkage = col_linkage,

        row_cluster = True,
        vmin = 0 if scaling is not None else None,
        vmax = 1 if scaling is not None else None,
        cmap = cmap,
        dendrogram_ratio = (0.1, 0.1),
        annot_kws = {"size": 4},
        figsize = figsize,
        cbar_kws = {"label": "scaled expression", "orientation": 'horizontal'},
        yticklabels = True,
        xticklabels = False,
    )
    clustermap.fig.subplots_adjust(right=0.7)

    clustermap.ax_cbar.set_position([0.16, 0, 0.53, 0.02])
    ax = clustermap.ax_heatmap
    ax.set_xticklabels([])
    ax.yaxis.set_ticks_position("left")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = y_label_fontsize)
    clustermap.ax_row_dendrogram.set_visible(False)

    next_legend = 0.8
    for i, group in enumerate(groupby):
        group_lut = map_obs_to_cmap(data, group, annotation_cmaps[i], return_mapping = True)
        handles = [Patch(facecolor = group_lut[name]) for name in group_lut]
        legend_space = 0.05 * (len(data[group].unique()) + 1)
        group_legend = plt.legend(handles,
                                  group_lut,
                                  title = group,
                                  bbox_to_anchor = (1.01,
                                                    next_legend),
                                  bbox_transform=clustermap.fig.transFigure
                                  )
        next_legend -= legend_space
        clustermap.fig.add_artist(group_legend)
    if return_fig:
        return clustermap
    plt.show()