import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Literal, Union, Optional

from .utils import (prep_uns_dataframe,
                    scale_data,
                    select_gate_from_singleindex_dataframe,
                    map_obs_to_cmap,
                    calculate_sample_distance,
                    calculate_linkage,
                    add_metaclusters,
                    remove_ticklabels,
                    remove_ticks,
                    scale_cbar_to_heatmap,
                    add_categorical_legend_to_clustermap,
                    calculate_correlation_data,
                    ANNOTATION_CMAPS)
from ..utils import find_gate_path_of_gate, reduction_names, subset_gate

from ..exceptions.exceptions import AnalysisNotPerformedError

from scipy.spatial import distance
from scipy.cluster import hierarchy

from ._clustermap import create_clustermap

def marker_correlation(adata: AnnData,
                       gate: str,
                       scaling: Literal["MinMaxScaler", "RobustScaler"] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency", "all_cells"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno",
                       figsize: tuple[float, float] = (4,4),
                       return_fig: bool = False) -> Optional[Figure]:
    
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

    correlation_data = calculate_correlation_data(data[fluo_columns], corr_method = corr_method)

    clustermap = create_clustermap(data = correlation_data,
                                   cmap = cmap,
                                   figsize = figsize,
                                   vmin = -1,
                                   vmax = 1,
                                   cbar_kws = {"label": f"{corr_method} correlation",
                                               "orientation": 'horizontal'}
                                   )

    ax = clustermap.ax_heatmap
    heatmap_position = ax.get_position()
    
    scale_cbar_to_heatmap(clustermap,
                          heatmap_position = heatmap_position,
                          cbar_padding = 0.8)
    remove_ticklabels(ax, which = "x")
    remove_ticks(ax, which = "x")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 5)
    if return_fig:
        return clustermap
    plt.show()




def sample_correlation(adata: AnnData,
                       groupby: Optional[Union[str, list[str]]],
                       gate: str,
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno",
                       return_fig: bool = False,
                       metaclusters: Optional[int] = None,
                       figsize: tuple[float, float] = (4,4),
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc") -> Optional[Figure]:
    
    try:
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

    sample_IDs = data["sample_ID"].to_list()
    correlation_data = calculate_correlation_data(data[fluo_columns].T, corr_method = corr_method)

    row_linkage = calculate_linkage(correlation_data)

    if metaclusters is not None:
        groupby += ["metacluster"]
        data = add_metaclusters(adata = adata,
                                row_linkage = row_linkage,
                                n_clusters = metaclusters,
                                sample_IDs = sample_IDs,
                                label_metaclusters = label_metaclusters_in_dataset,
                                label_metaclusters_key = label_metaclusters_key)
    
    clustermap = create_clustermap(data = correlation_data,
                                   row_colors = [
                                       map_obs_to_cmap(data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(groupby)
                                   ],
                                   col_colors = [
                                       map_obs_to_cmap(data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(groupby)
                                   ],
                                   row_linkage = row_linkage,
                                   col_linkage = row_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   vmin = -1,
                                   vmax = 1,
                                   cbar_kws = {"label": f"{corr_method} correlation",
                                               "orientation": 'horizontal'}
                                   )
    ax = clustermap.ax_heatmap
    heatmap_position = ax.get_position()
    
    scale_cbar_to_heatmap(clustermap,
                          heatmap_position = heatmap_position)
    remove_ticklabels(ax, which = "both")
    remove_ticks(ax, which = "both")
    add_categorical_legend_to_clustermap(clustermap,
                                         heatmap = ax,
                                         data = data,
                                         groupby = groupby)
    if return_fig:
        return clustermap
    plt.show()